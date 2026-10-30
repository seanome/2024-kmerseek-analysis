#!/usr/bin/env nextflow
/*
 * Notebook 361: Domen et al. 2026 (PNAS) PEC RNA-seq without a genome.
 *
 * Per contrast (from the frozen design in analysis/pec-domen2026-splash/design/):
 *   ENA download (md5 + read count checked) -> fastp (paired + interleaved copy)
 *   -> SPLASH 2.11.9, supervised (PEC +1, control -1)  -> compactors on every anchor with
 *      corrected p <= 0.05  -> six-frame peptides -> sourmash manysearch -> parasail pair check
 *   -> SPLASH anchor counts -> PyDESeq2
 *   -> DE-kupl 1.3.5 (limma test) -> contigs -> the same peptide search
 * plus the known-answer control (simulated reads, same stages, scored against the truth).
 *
 * Every result is kept with storeDir under params.pec_root, in the same places the earlier
 * Slurm stages wrote, so results that already exist are found and not recomputed.
 * The stage logic is the same as analysis/pec-domen2026-splash/slurm/*.sbatch.
 */
nextflow.enable.dsl = 2

params.pec_root     = null      // $SCRATCH/nb361-pec
params.code_dir     = null      // .../analysis/pec-domen2026-splash
params.contrasts    = 'young_2h,young_24h,old_24h,young_13_49d,young_132d,old_41_93d'
params.control      = true
params.dekupl_method = 'limma'
params.reuse_dir    = ''        // other run's verified FASTQs, hard-linked when the MD5 matches

if (!params.pec_root || !params.code_dir) {
    error "set --pec_root and --code_dir"
}

ENV      = "${params.pec_root}/env/bin"
SPLASH   = "${params.pec_root}/splash"
PECTOOLS = "${params.pec_root}/pec-tools/target/release/pec-tools"
SCRIPTS  = "${params.code_dir}/scripts"
DB       = "${params.pec_root}/db"

// ---------------------------------------------------------------- reads and trimming

process download {
    tag "${run}"
    label 'download'
    storeDir "${params.pec_root}/reads"

    input:
    val run

    output:
    tuple val(run), path("${run}_1.fastq.gz"), path("${run}_2.fastq.gz"), path("${run}_1.json"), path("${run}_2.json"), path("${run}.done")

    script:
    """
    export PATH=${ENV}:\$PATH
    python ${SCRIPTS}/download_run.py ${params.pec_root}/meta/ena_filereport.tsv ${run} . ${params.reuse_dir}
    """
}

process fastp {
    tag "${sample}"
    label 'fastp'
    storeDir "${store}"

    input:
    tuple val(sample), val(store), path(r1), path(r2)

    output:
    tuple val(sample), path("${sample}_1.fastq.gz"), path("${sample}_2.fastq.gz"), path("${sample}.il.fastq.gz"), path("${sample}.fastp.json"), path("${sample}.fastp.html"), path("${sample}.done")

    script:
    """
    export PATH=${ENV}:\$PATH
    # fastp picks gzip from the extension, so temporary names end in .fastq.gz
    fastp -i ${r1} -I ${r2} -o t1.tmp.fastq.gz -O t2.tmp.fastq.gz --compression 4 \\
      --detect_adapter_for_pe --trim_poly_g --length_required 50 --thread ${task.cpus} \\
      -j ${sample}.fastp.json -h ${sample}.fastp.html 2> ${sample}.fastp.log
    WANT=\$(python -c "import json;print(json.load(open('${sample}.fastp.json'))['summary']['after_filtering']['total_reads'])")
    N1=\$(pigz -dc t1.tmp.fastq.gz | awk 'END{print NR/4}'); N2=\$(pigz -dc t2.tmp.fastq.gz | awk 'END{print NR/4}')
    [ "\$N1" = "\$N2" ] && [ \$(( N1 + N2 )) = "\$WANT" ] || { echo "R1 \$N1 R2 \$N2 fastp \$WANT"; exit 3; }
    mv t1.tmp.fastq.gz ${sample}_1.fastq.gz; mv t2.tmp.fastq.gz ${sample}_2.fastq.gz
    # interleaved copy for SPLASH: R1 record, then its R2 record
    paste <(pigz -dc ${sample}_1.fastq.gz | paste - - - -) <(pigz -dc ${sample}_2.fastq.gz | paste - - - -) \\
      | tr '\\t' '\\n' | pigz -p ${task.cpus} -1 > ${sample}.il.fastq.gz
    N=\$(pigz -dc ${sample}.il.fastq.gz | awk 'END{print NR/4}')
    [ "\$N" = "\$WANT" ] || { echo "interleaved \$N != fastp \$WANT"; exit 3; }
    echo "\$N" > ${sample}.done
    """
}

process simulate_control {
    label 'small'
    storeDir "${params.pec_root}/control"

    output:
    path("reads")

    script:
    """
    export PATH=${ENV}:\$PATH
    mkdir reads
    ${PECTOOLS} simulate ${params.code_dir}/control/control_transcripts.fa ${params.code_dir}/control/design.tsv reads 1000000 11 2> simulate.log
    pigz -p ${task.cpus} reads/*.fastq
    """
}

// ---------------------------------------------------------------- SPLASH channel

process splash {
    tag "${contrast}"
    label 'splash'
    storeDir "${params.pec_root}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(cjs), path(il)

    output:
    tuple val(contrast), path("splash_out")

    script:
    """
    export PATH=${ENV}:\$PATH
    for f in *.il.fastq.gz; do echo "\${f%.il.fastq.gz} \$PWD/\$f"; done > input.txt
    [ \$(wc -l < input.txt) = \$(wc -l < ${cjs}) ] || { echo "samples and cjs differ"; exit 2; }
    mkdir splash_out && cd splash_out
    ${SPLASH}/splash --bin_path ${SPLASH} --Cjs_samplesheet ../${cjs} \\
      --pvals_correction_col_name Cjs_0_pval --with_effect_size_cts --n_most_freq_targets 4 \\
      --dump_sample_anchor_target_count_binary --without_compactors \\
      --n_threads_stage_1 4 --n_threads_stage_1_internal \$(( ${task.cpus} / 4 > 0 ? ${task.cpus} / 4 : 1 )) \\
      --n_threads_stage_2 ${task.cpus} --kmc_max_mem_GB ${(task.memory.toGiga() / 3) as int} \\
      --tmp_dir \$PWD/../splash_tmp --logs_dir logs --outname_prefix result ../input.txt \\
      > splash.stdout 2> splash.stderr
    rm -rf ../splash_tmp
    test -s result.after_correction.all_anchors.tsv && test -s result.after_correction.scores.tsv
    """
}

process compactors {
    tag "${contrast}"
    label 'compactors'
    storeDir "${params.pec_root}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(splash_out), path(il)

    output:
    tuple val(contrast), path("compactors")

    script:
    // every anchor with corrected supervised p <= 0.05 (all rows of scores.tsv), no top-N cut
    """
    mkdir compactors && cd compactors
    for f in ../*.il.fastq.gz; do readlink -f \$f; done > input_list.txt
    echo "significant anchors: \$(( \$(wc -l < ../${splash_out}/result.after_correction.scores.tsv) - 1 ))" > n_input_anchors.txt
    ${SPLASH}/compactors --max_length 186 --num_threads ${task.cpus} --epsilon 0.001 --beta 0.5 \\
      --num_kmers 2 --input_format fastq input_list.txt ../${splash_out}/result.after_correction.scores.tsv \\
      compactors.tsv > compactors.log 2>&1
    test -s compactors.tsv
    """
}

process anchor_counts {
    tag "${contrast}"
    label 'anchor_counts'
    storeDir "${params.pec_root}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(splash_out), path(design)

    output:
    tuple val(contrast), path("anchor_counts.tsv"), path("deseq")

    script:
    """
    export PATH=${ENV}:\$PATH
    tail -n +2 ${design} | cut -f1 > samples.txt
    ALL=${splash_out}/result.after_correction.all_anchors.tsv
    for b in ${splash_out}/result_satc/bin*.satc; do
      ${SPLASH}/satc_dump --sample_names ${splash_out}/sample_name_to_id.mapping.txt \$b /dev/stdout \\
        | ${PECTOOLS} anchor-counts samples.txt \$ALL >> anchor_counts.tsv 2>> anchor_counts.log
    done
    N=\$(wc -l < anchor_counts.tsv); W=\$(( \$(wc -l < \$ALL) - 1 ))
    [ "\$N" = "\$W" ] || { echo "anchor count rows \$N != tested anchors \$W"; exit 3; }
    python ${SCRIPTS}/anchor_deseq.py anchor_counts.tsv samples.txt ${design} deseq ${task.cpus} > deseq.log 2>&1
    test -s deseq/deseq_summary.json
    """
}

// ---------------------------------------------------------------- DE-kupl channel

process dekupl {
    tag "${contrast}"
    label 'dekupl'
    storeDir "${params.pec_root}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(design), path(pairs)

    output:
    tuple val(contrast), path("dekupl_${params.dekupl_method}")

    script:
    def W = "dekupl_${params.dekupl_method}"
    """
    mkdir -p ${W}/fastq
    # absolute targets, so the links still resolve after storeDir moves the folder
    for f in *_1.fastq.gz *_2.fastq.gz; do ln -s \$(readlink -f \$f) ${W}/fastq/\$f; done
    cd ${W}
    ${ENV}/python ${SCRIPTS}/dekupl_config.py ../${design} \$PWD/fastq \$PWD/../dekupl_tmp config.json ${params.dekupl_method}
    # the image's limma script loads pryr (never used) but the image lacks it; supplied beside it
    export APPTAINERENV_R_LIBS=${params.pec_root}/Rlib_dekupl
    apptainer run --bind /scratch ${params.pec_root}/images/dekupl-run_1.3.5.sif \\
      --configfile config.json -j ${task.cpus} --resources ram=${(task.memory.toMega() * 0.9) as long} -p > dekupl.log 2>&1
    rm -rf ../dekupl_tmp
    test -s DEkupl_result/control_vs_PEC_kmer_counts/merged-diff-counts.tsv.gz
    """
}

// ---------------------------------------------------------------- protein search

process search {
    tag "${contrast} ${src}"
    label 'search'
    storeDir "${params.pec_root}/contrasts/${contrast}"

    input:
    tuple val(contrast), val(src), path(seqdir)

    output:
    tuple val(contrast), val(src), path("search_${src}")

    script:
    // compactors: compactors/compactors.tsv, column compactor; DE-kupl: its merged contigs, column contig
    def prep = src == 'compactors'
        ? "ln -s ../${seqdir}/compactors.tsv seqs.tsv; COL=compactor"
        : "zcat ../${seqdir}/DEkupl_result/control_vs_PEC_kmer_counts/merged-diff-counts.tsv.gz > dekupl_contigs.tsv; ln -s dekupl_contigs.tsv seqs.tsv; COL=contig"
    """
    export PATH=${ENV}:\$PATH
    mkdir search_${src} && cd search_${src}
    ${prep}
    ${PECTOOLS} translate seqs.tsv pep 20 \$COL 2> translate.log
    if [ -s pep.peptides.faa ]; then
      sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\\npep,,%s\\n' "\$PWD/pep.peptides.faa") \\
        --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c ${task.cpus} -o pep.scaled5.zip
      for db in proteome swissprot botznik; do
        for spec in protein:7 dayhoff:12 hp:30; do
          m=\${spec%:*}; k=\${spec#*:}
          for s in 10 100 5; do
            sourmash scripts manysearch pep.scaled5.zip ${DB}/\$db.\$m.k\$k.s\$s.rocksdb -m \$m -k \$k -s \$s \\
              -t 0.000001 -N -c ${task.cpus} -o ms.\$db.\$m.k\$k.s\$s.csv 2>> manysearch.log
          done
        done
      done
      python ${SCRIPTS}/make_pairs.py pairs.tsv hits.parquet ms.*.csv
      for db in proteome swissprot botznik; do
        python ${SCRIPTS}/pair_align.py pairs.\$db.tsv pep.peptides.faa ${DB}/\$db.faa align.\$db.parquet ${task.cpus}
      done
    fi
    rm -f seqs.tsv
    touch search.done
    """
}

// ---------------------------------------------------------------- known-answer control scoring

process control_truth {
    label 'small_mem'
    storeDir "${params.pec_root}/control_eval"

    input:
    tuple path(splash_out), path(anchor_counts), path(deseq), path(dekupl_dir)

    output:
    path("control_truth_summary.json")
    path("control_genes_found.tsv")

    script:
    """
    mkdir -p cdir && cd cdir && ln -s ../${splash_out} splash_out && ln -s ../${deseq} deseq && ln -s ../${dekupl_dir} ${dekupl_dir}
    ${ENV}/python ${SCRIPTS}/control_truth.py ${params.code_dir}/control/control_transcripts.fa \\
      ${params.code_dir}/control/truth_design.tsv . .. > ../control_truth.log 2>&1
    """
}

// ---------------------------------------------------------------- workflow

workflow {
    def contrasts = params.contrasts.tokenize(',')
    // frozen design: sample, run, treatment, genotype, age_days, contrast, ...
    def rows = file("${params.code_dir}/design/samples.tsv").splitCsv(header: true, sep: '\t')
    def used = rows.findAll { it.contrast in contrasts }
    def run2sample = used.collectEntries { [(it.run): it.sample] }
    def sample2contrast = used.collectEntries { [(it.sample): it.contrast] }

    reads = download(Channel.fromList(used.collect { it.run }))
        .map { run, r1, r2, j1, j2, d -> tuple(run2sample[run], "${params.pec_root}/trimmed", r1, r2) }

    // one fastp call for real and simulated samples (a DSL2 process runs once per workflow)
    if (params.control) {
        sim = simulate_control().flatMap { dir ->
            def fs = file("${dir}/*_{1,2}.fastq.gz").sort { it.name }
            fs.groupBy { it.name.replaceAll(/_[12]\.fastq\.gz$/, '') }
              .collect { s, pair -> tuple(s, "${params.pec_root}/control/trimmed", pair[0], pair[1]) }
        }
        reads = reads.mix(sim)
    }
    trimmed = fastp(reads)

    // group trimmed samples by contrast; the control's samples are ctrl1-6 and pec1-6
    keyed = trimmed.map { s, p1, p2, il, js, ht, d -> tuple(sample2contrast.get(s, 'control'), s, il, [p1, p2]) }
    all_contrasts = contrasts + (params.control ? ['control'] : [])
    per_contrast = keyed.groupTuple().map { c, ss, ils, pairs ->
        def dsg = c == 'control' ? file("${params.code_dir}/control/control.design.tsv")
                                 : file("${params.code_dir}/design/contrast_${c}.design.tsv")
        def cjs = c == 'control' ? file("${params.code_dir}/control/control.cjs")
                                 : file("${params.code_dir}/design/contrast_${c}.cjs")
        tuple(c, cjs, dsg, ils, pairs.flatten())
    }

    sp = splash(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, cjs, il) })
    comp = compactors(sp.join(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, il) }))
    ac = anchor_counts(sp.join(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, dsg) }))
    dk = dekupl(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, dsg, pairs) })

    search(comp.map { c, d -> tuple(c, 'compactors', d) }
               .mix(dk.map { c, d -> tuple(c, "dekupl_${params.dekupl_method}", d) }))

    if (params.control) {
        control_truth(sp.join(ac).join(dk).filter { it[0] == 'control' }
                        .map { c, s, a, d, k -> tuple(s, a, d, k) })
    }
}
