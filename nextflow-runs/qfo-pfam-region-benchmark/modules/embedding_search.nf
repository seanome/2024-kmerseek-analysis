/*
 * G3, embedding search: TM-Vec and PLMSearch, the 998 human queries against each QfO
 * target proteome of the midi-plus run. Call = the best hit's Pfam domains, from the
 * target's domain map (the same map the main run's transfers read):
 *
 *   TM-Vec     best hit by predicted TM-score; when a DeepBLAST model is given, the
 *              domains are placed through DeepBLAST's structural alignment of the pair
 *              (annotator_best_hit.place_domains); without one, or for a pair DeepBLAST
 *              does not align, no position is claimed.
 *   PLMSearch  best hit by PLMSearch's score (its embedding search, the "SS-predictor";
 *              the Pfam-based prefilter of its full pipeline is not run). It reports no
 *              alignment, so every call spans the whole query and the position scores
 *              are "not applicable".
 *
 * GPU: one per task (-G 1, Sherlock's rule for GPU jobs), `--nv` for the CUDA runtime.
 * No history: 1 GPU, 8 CPUs, 32 GB, 2 h per target proteome (prompt G's ask is 1 GPU,
 * 2 h). GPU hours are realtime of these tasks, from the trace; record peaks with sacct.
 *
 * Images are built from Dockerfile.tmvec and Dockerfile.plmsearch (`make
 * push-embedding-images`); their tags are params so a rebuilt image is a new tag.
 */

include { retryOnKill } from './annotator_common'

process shortHeaders {
    tag "${name}"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 2.GB * 2 ** (task.attempt - 1) }
    time { 1.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    tuple val(name), path(fasta)

    output:
    tuple val(name), path("${name}.short.fasta")

    script:
    """
    embedding_calls.py short-headers --fasta ${fasta} --out ${name}.short.fasta
    """

    stub:
    """
    sed 's/^>[^|]*|\\([^|]*\\)|.*/>\\1/' ${fasta} > ${name}.short.fasta
    """
}

process tmvecSearch {
    tag "tmvec ${species}"
    container "${params.tmvec_image}"
    containerOptions '--nv'
    // Sherlock only (the standard profile ignores both): a GPU partition and -G 1.
    queue 'gpu,hns'
    clusterOptions '--account=ayeletv -G 1'
    cpus 8
    memory { 32.GB * 2 ** (task.attempt - 1) }
    time { 2.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/hits/tmvec", mode: 'copy'

    input:
    tuple val(species), path(targets), path(queries), path(model), path(config),
          path(prott5), path(deepblast)

    output:
    tuple val("tmvec"), val(params.tmvec_setting), val(species),
          path("tmvec.${species}.tabular.tsv"), path("tmvec.${species}.alignments.txt"),
          path(targets), emit: hits
    path "tmvec.${species}.run.txt", emit: run_info

    script:
    def has_deepblast = deepblast.name != 'NO_DEEPBLAST'
    def build = "tmvec-build-database --input-fasta ${targets} --tm-vec-model ${model} " +
                "--tm-vec-config-path ${config} --protrans-model ${prott5} " +
                "--device gpu --output db"
    def common = "--query ${queries} --tm-vec-model ${model} --tm-vec-config ${config} " +
                 "--protrans-model ${prott5} --database \$DB --metadata \$META " +
                 "--database-fasta ${targets} --device gpu --k-nearest-neighbors 1"
    def tab = "tmvec-search ${common} --output-format tabular --output tmvec.${species}.tabular.tsv"
    def aln = "tmvec-search ${common} --database-faidx targets.fai --deepblast-model ${deepblast} " +
              "--output-format alignment --output tmvec.${species}.alignments.txt"
    """
    set -euo pipefail
    ${build}
    # The build writes a database and a metadata array; their exact file names differ
    # between the DeepBLAST wiki (db.npy, meta.npy) and its prose (.db, .meta).
    DB=\$( (ls db/*db* db*db* 2>/dev/null || true) | grep -v meta | head -1 || true)
    META=\$( (ls db/*meta* db*meta* 2>/dev/null || true) | head -1)
    [ -n "\$DB" ] && [ -n "\$META" ] || { ls -la db* >&2; echo "no TM-Vec database files" >&2; exit 1; }
    ${tab}
    if ${has_deepblast}; then
        build-fasta-index --fasta ${targets} --faidx targets.fai
        ${aln}
    else
        : > tmvec.${species}.alignments.txt
    fi
    {
        echo "command: ${build} && ${tab}${has_deepblast ? ' && ' + aln : ''}"
        echo "version: tm-vec \$(pip show tm-vec 2>/dev/null | sed -n 's/^Version: //p'), deepblast \$(pip show deepblast 2>/dev/null | sed -n 's/^Version: //p')"
        echo "cpu_model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
        echo "gpu_model: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    } > tmvec.${species}.run.txt
    """

    stub:
    def fx = "${projectDir}/tests/fixtures/embedding"
    """
    cp ${fx}/tmvec.tabular.tsv tmvec.${species}.tabular.tsv
    cp ${fx}/tmvec.alignments.txt tmvec.${species}.alignments.txt
    printf 'command: stub\\nversion: stub\\ncpu_model: stub\\ngpu_model: stub\\n' > tmvec.${species}.run.txt
    """
}

process plmsearchEmbed {
    tag "plmsearch embed ${name}"
    container "${params.plmsearch_image}"
    containerOptions '--nv'
    // Sherlock only (the standard profile ignores both): a GPU partition and -G 1.
    queue 'gpu,hns'
    clusterOptions '--account=ayeletv -G 1'
    cpus 8
    memory { 32.GB * 2 ** (task.attempt - 1) }
    time { 2.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    tuple val(name), path(fasta), path(model_dir)

    output:
    tuple val(name), path("${name}.embedding.pkl")

    script:
    // ESM-1b reads at most 1022 residues per protein (PLMSearch's batch converter);
    // longer proteins are embedded from their first 1022, as the authors run it.
    """
    python \$PLMSEARCH_HOME/plmsearch/embedding_generate.py \\
        -emp ${model_dir}/esm/esm1b_t33_650M_UR50S.pt \\
        -f ${fasta} -e ${name}.embedding.pkl
    """

    stub:
    """
    touch ${name}.embedding.pkl
    """
}

process plmsearchSearch {
    tag "plmsearch ${species}"
    container "${params.plmsearch_image}"
    containerOptions '--nv'
    // Sherlock only (the standard profile ignores both): a GPU partition and -G 1.
    queue 'gpu,hns'
    clusterOptions '--account=ayeletv -G 1'
    cpus 8
    memory { 32.GB * 2 ** (task.attempt - 1) }
    time { 2.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/hits/plmsearch", mode: 'copy', pattern: '*.run.txt'

    input:
    tuple val(species), path(target_emb), path(query_emb), path(model_dir)

    output:
    tuple val("plmsearch"), val(params.plmsearch_setting), val(species),
          path("plmsearch.${species}.results.tsv"), emit: hits
    path "plmsearch.${species}.run.txt", emit: run_info

    script:
    def cmd = "python \$PLMSEARCH_HOME/plmsearch/main_similarity.py -iqe ${query_emb} " +
              "-ite ${target_emb} -smp ${model_dir}/plmsearch.sav " +
              "-osr plmsearch.${species}.results.tsv"
    """
    set -euo pipefail
    ${cmd}
    {
        echo "command: ${cmd}"
        echo "version: PLMSearch \$(cd \$PLMSEARCH_HOME && git rev-parse --short HEAD 2>/dev/null || echo d6df1dc)"
        echo "cpu_model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
        echo "gpu_model: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    } > plmsearch.${species}.run.txt
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/embedding/plmsearch.results.tsv plmsearch.${species}.results.tsv
    printf 'command: stub\\nversion: stub\\ncpu_model: stub\\ngpu_model: stub\\n' > plmsearch.${species}.run.txt
    """
}

process embeddingCalls {
    tag "${tool} ${species}"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 8.GB * 2 ** (task.attempt - 1) }
    time { 1.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/regions", mode: 'copy'

    input:
    tuple val(tool), val(setting), val(species), path(results), path(alignments),
          path(targets), path(domain_map), path(queries)

    output:
    tuple val(tool), val(setting), val(species), path("${tool}.${setting}.${species}.regions.tsv.gz"), emit: regions
    path "${tool}.${setting}.${species}.calls_qc.json", emit: qc

    script:
    def out = "--out ${tool}.${setting}.${species}.regions.tsv.gz --qc-out ${tool}.${setting}.${species}.calls_qc.json"
    if (tool == 'tmvec')
        """
        embedding_calls.py tmvec --tabular ${results} --alignments ${alignments} \\
            --targets ${targets} --queries ${queries} --domain-map ${domain_map} \\
            --setting ${setting} --species ${species} --min-overlap ${params.min_overlap} ${out}
        """
    else
        """
        embedding_calls.py plmsearch --results ${results} --queries ${queries} \\
            --domain-map ${domain_map} --setting ${setting} --species ${species} ${out}
        """

    stub:
    def fx = "${projectDir}/tests/fixtures/embedding"
    """
    cp ${fx}/${tool}.${setting}.yeast.regions.tsv.gz ${tool}.${setting}.${species}.regions.tsv.gz
    cp ${fx}/${tool}.${setting}.yeast.calls_qc.json ${tool}.${setting}.${species}.calls_qc.json
    """
}

workflow TMVEC {
    take:
    targets      // tuple(species, short fasta, domain map)
    queries      // short-header query FASTA
    models       // tuple(model, config, prott5 dir, deepblast model or NO_DEEPBLAST)

    main:
    tmvecSearch(targets.map { s, f, m -> [s, f] }.combine(queries).combine(models))
    calls = embeddingCalls(
        tmvecSearch.out.hits
            .map { tool, setting, species, tab, aln, tfa -> [species, tool, setting, tab, aln, tfa] }
            .combine(targets.map { s, f, m -> [s, m] }, by: 0)
            .map { species, tool, setting, tab, aln, tfa, map -> [tool, setting, species, tab, aln, tfa, map] }
            .combine(queries)
    )

    emit:
    regions = calls.regions
}

workflow PLMSEARCH {
    take:
    targets      // tuple(species, short fasta, domain map)
    queries      // short-header query FASTA
    model_dir

    main:
    emb = plmsearchEmbed(
        targets.map { s, f, m -> [s, f] }
            .mix(queries.map { ['query', it] })
            .combine(model_dir)
    )
    q_emb = emb.filter { it[0] == 'query' }.map { it[1] }
    t_emb = emb.filter { it[0] != 'query' }
    plmsearchSearch(t_emb.combine(q_emb).combine(model_dir))
    calls = embeddingCalls(
        plmsearchSearch.out.hits
            .map { tool, setting, species, res -> [species, tool, setting, res] }
            .combine(targets, by: 0)
            .map { species, tool, setting, res, tfa, map ->
                [tool, setting, species, res, file("${projectDir}/assets/NO_ALIGNMENTS"), tfa, map] }
            .combine(queries)
    )

    emit:
    regions = calls.regions
}
