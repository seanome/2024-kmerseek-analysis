#!/usr/bin/env nextflow
/*
 * Pilot: does an index of Swiss-Prot's annotated regions alone let kmerseek call short and
 * disordered regions that it cannot call against whole proteins, and does it help kmerseek
 * more than MMseqs2?
 *
 * Queries: reviewed human Swiss-Prot entries on chromosome 6 (assets/chr6_queries.*.tsv).
 * Truth:   each query's own Swiss-Prot features (bin/build_query_truth.py).
 * Indexes, both from reviewed Swiss-Prot with every Mammalia entry removed:
 *   the whole-protein index  every remaining entry, whole
 *   the regions index        every DOMAIN, REGION, MOTIF, REPEAT, ZN_FING, COMPBIAS, COILED,
 *                            TRANSMEM and INTRAMEM of those entries, cut out with
 *                            (k_max - 1) residues each side; 30 aa or more clustered with
 *                            MMseqs2 (50% identity, 80% coverage of both), shorter ones
 *                            deduplicated by exact sequence
 * Each index is searched together with its decoy copy (residues shuffled within 10-residue
 * windows, names prefixed DECOY_), so calls can be compared at the same decoy error rate.
 * A second decoy copy, shuffled within 20-residue windows, repeats everything as a check.
 *
 * Entries:
 *   -entry build   the queries, truth, both indexes and their decoys
 *   (default)      the build, then kmerseek (four settings), MMseqs2 and the
 *                  composition-only classifier against both indexes, the scoring at 5%
 *                  decoy error, and the comparison tables (bin/compare_indexes.py)
 * --query_limit 20 runs the searches on 20 queries spread over the accession list.
 */

nextflow.enable.dsl = 2

def home = System.getenv('HOME')

params.swissprot_dat = "${home}/data/uniprot/uniprot_sprot.2026_03.dat.gz"
params.chr6_queries  = "${projectDir}/assets/chr6_queries.2026_03.tsv"
params.disprot       = "${projectDir}/assets/disprot_human_disorder.tsv"
params.outdir        = null
params.exclude_clade = 'Mammalia'
// The largest k-size among the kmerseek settings searched. A regions-index entry carries
// k_max - 1 residues each side of its feature, so a k-mer of that size can start on the
// feature's first residue and end on its last. 21 is the k of the two H/P settings
// (hp_pbotc_1st_ed2 and hp_lehninger2), chosen to carry about the same bits per k-mer as
// protein20 k = 5 (see PREDICTIONS.md).
params.k_max          = 21
params.min_cluster_aa = 30
params.cluster_min_seq_id = 0.5
params.cluster_coverage   = 0.8
// The decoys' shuffle window. The first is the headline; the others repeat everything as a
// check (Olga, 2026-10-07: keep 10, add 20).
params.decoy_windows = '10,20'
params.decoy_seed    = 20261006

// kmerseek settings, alphabet:k:scaled (Olga, 2026-10-06 and 2026-10-07; PREDICTIONS.md).
// Each searches with extension at its alphabet's own mismatch penalty, c_opt in
// assets/kappa_by_alphabet.tsv (copied from olgabot/dark-set-kmerseek-0.4), X-drop 4 x C,
// low-complexity mask off.
params.kmerseek_settings = 'polarity4:11:1,hp_pbotc_1st_ed2:21:1,hp_lehninger2:21:1,protein20:5:1'
params.kappa_table       = "${projectDir}/assets/kappa_by_alphabet.tsv"
params.xdrop_per_penalty = 4
params.chain_max_gap     = 30
params.chain_max_shift   = 10
params.ka_queries        = 500
// The dark set's search filters (olgabot/dark-set-kmerseek-0.4 main.nf, 7913404).
params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_query_pvalue = 0.05
params.min_region_score = 1.3
// Rows above this region_evalue are dropped in the search's own pipe, as in the dark set.
// The 5% decoy thresholds are expected far below it; compare_indexes.py checks.
params.max_region_evalue = 10000
// The dark set's mmseqs2Search: -s 7, 3 iterations, report E <= 10.
params.mmseqs_sensitivity = 7
params.mmseqs_iterations  = 3
params.mmseqs_evalue      = 10
params.composition_window = 30
params.composition_step   = 10
params.max_decoy_rate = 0.05
params.min_landing    = 0.5
params.query_chunk_size = 100
// 0 searches every query. A positive number searches that many, spread evenly over the
// accession-sorted list, for a test run.
params.query_limit = 0
// A command put in front of each kmerseek and MMseqs2 call to record its peak memory. Empty on
// Sherlock, where the trace has it; `/usr/bin/time -l` on the laptop (profile local), where
// Nextflow reports no memory on macOS. Its report goes to the task's log.
params.time_cmd = ''

// The interpreter for every bin/ script, and the MMseqs2 binary. The container's on
// Sherlock; full paths on the laptop (nextflow.config, profile local), because a
// `#!/usr/bin/env python3` script inside Nextflow finds the system Python, not the conda env.
def py(String script) { "${params.python} \$(command -v ${script})" }

process parseFeatures {
    label 'python'
    publishDir "${params.outdir}/swissprot", mode: 'copy'

    input:
    path dat

    output:
    path 'sprot_features.parquet', emit: features
    path 'sprot_sequences.parquet', emit: sequences
    path 'parse_features.log'

    script:
    """
    ${py('parse_swissprot_features.py')} --swissprot-dat ${dat} --out-prefix sprot 2> parse_features.log
    cat parse_features.log >&2
    """
}

// Reused from invertebrate-dark-set (olgabot/dark-set-kmerseek-0.4 at 7913404), changed only
// by black formatting.
process buildWholeReference {
    label 'python'
    publishDir "${params.outdir}/whole_protein_index", mode: 'copy'

    input:
    path dat

    output:
    path 'reference.fasta', emit: fasta
    path 'reference_accessions.txt', emit: accessions
    path 'reference_summary.json'

    script:
    """
    ${py('build_clade_excluded_reference.py')} --swissprot-dat ${dat} \\
        --exclude-clade ${params.exclude_clade} --out-prefix reference \\
        --summary-out reference_summary.json
    """
}

process buildQueryTruth {
    label 'python'
    publishDir "${params.outdir}/queries", mode: 'copy'

    input:
    path features
    path sequences
    path chr6
    path disprot

    output:
    path 'queries.fasta', emit: queries
    path 'truth.parquet', emit: truth
    path 'truth_summary.tsv'
    path 'truth.log'

    script:
    """
    ${py('build_query_truth.py')} --features ${features} --sequences ${sequences} \\
        --chr6 ${chr6} --disprot ${disprot} --out-dir . 2> truth.log
    cat truth.log >&2
    """
}

process extractRegions {
    label 'python'
    publishDir "${params.outdir}/regions_index", mode: 'copy', pattern: 'regions.parquet'

    input:
    path features
    path sequences
    path accessions

    output:
    path 'regions.parquet', emit: regions
    path 'regions_long.fasta', emit: long_fasta
    path 'regions_short.fasta', emit: short_fasta

    script:
    """
    ${py('extract_regions.py')} --features ${features} --sequences ${sequences} \\
        --accessions ${accessions} --k-max ${params.k_max} \\
        --min-cluster-aa ${params.min_cluster_aa} --out-dir .
    """
}

process clusterRegions {
    label 'mmseqs'
    publishDir "${params.outdir}/regions_index", mode: 'copy', pattern: '*.log'

    input:
    path long_fasta

    output:
    path 'clu_cluster.tsv', emit: tsv
    path 'cluster.log'

    script:
    """
    set -euo pipefail
    ${params.mmseqs} easy-cluster ${long_fasta} clu tmp \\
        --min-seq-id ${params.cluster_min_seq_id} -c ${params.cluster_coverage} --cov-mode 0 \\
        --threads ${task.cpus} > cluster.log 2>&1
    ${params.mmseqs} version >> cluster.log
    """
}

process finishRegionsIndex {
    label 'python'
    publishDir "${params.outdir}/regions_index", mode: 'copy'

    input:
    path regions
    path cluster_tsv

    output:
    path 'regions_index.fasta', emit: fasta
    path 'cluster_members.parquet'
    path 'cluster_summary.tsv'

    script:
    """
    ${py('finish_regions_index.py')} --regions ${regions} --cluster-tsv ${cluster_tsv} --out-dir .
    """
}

process makeDecoys {
    label 'python'
    tag "${name}.w${window}"
    publishDir "${params.outdir}/${name}", mode: 'copy', pattern: '*.{md5,log}'

    input:
    tuple val(name), path(fasta), val(window)

    output:
    tuple val(name), val(window), path("${name}.w${window}.target_decoy.fasta"), emit: combined
    path "${name}.w${window}.decoy.fasta.md5"
    path "${name}.w${window}.decoys.log"

    script:
    """
    ${py('make_window_decoys.py')} --fasta ${fasta} --decoy-out ${name}.w${window}.decoy.fasta \\
        --combined-out ${name}.w${window}.target_decoy.fasta \\
        --window ${window} --seed ${params.decoy_seed} 2> ${name}.w${window}.decoys.log
    cat ${name}.w${window}.decoys.log >&2
    """
}

process indexSummary {
    label 'python'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path queries
    tuple val(name_a), path(fasta_a), val(name_b), path(fasta_b)

    output:
    path 'index_summary.json'
    path 'index_summary.log'

    script:
    """
    ${py('index_summary.py')} --queries ${queries} --index ${name_a} ${fasta_a} \\
        --index ${name_b} ${fasta_b} --out index_summary.json 2> index_summary.log
    cat index_summary.log >&2
    """
}

// ---------------------------------------------------------------------------------------
// Searches, scoring and comparison
// ---------------------------------------------------------------------------------------

// alphabet -> c_opt, from the kappa table's header-named columns.
def kappaPenalties() {
    def rows = file(params.kappa_table).readLines().findAll { it && !it.startsWith('#') }
    def head = rows[0].split('\t') as List
    def ia = head.indexOf('alphabet'), ic = head.indexOf('c_opt')
    if (ia < 0 || ic < 0) error "${params.kappa_table} has no alphabet or c_opt column"
    rows.drop(1).collectEntries { r -> def f = r.split('\t'); [(f[ia]): f[ic]] }
}

// The dark set's penaltyString: two decimals, trailing zeros dropped.
def penaltyString(double c) {
    def sf = String.format('%.2f', c)
    sf.contains('.') ? sf.replaceAll(/0+$/, '').replaceAll(/\.$/, '') : sf
}

def kmerseekSettings() {
    def pens = kappaPenalties()
    params.kmerseek_settings.toString().tokenize(',')*.trim().collect { spec ->
        def (alphabet, k, scaled) = spec.tokenize(':')
        def c = pens[alphabet]
        if (c == null) error "no c_opt for ${alphabet} in ${params.kappa_table}"
        def x = penaltyString((params.xdrop_per_penalty as double) * (c as double))
        [alphabet, k as int, scaled as int, c, x, "${alphabet}_k${k}_s${scaled}".toString()]
    }
}

// whole_protein_index -> whole, regions_index -> regions: the index part of a label.
def indexKind(String name) { name.startsWith('whole') ? 'whole' : 'regions' }

process splitQueries {
    label 'python'

    input:
    path queries

    output:
    path 'chunk_*.fasta', emit: chunks
    path 'searched_queries.fasta', emit: searched

    script:
    """
    ${params.python} - <<'PYEOF'
    recs, name, seq = [], None, []
    for line in open("${queries}"):
        line = line.rstrip("\\n")
        if line.startswith(">"):
            if name: recs.append((name, "".join(seq)))
            name, seq = line[1:].split()[0], []
        elif line:
            seq.append(line)
    if name: recs.append((name, "".join(seq)))
    recs.sort()
    limit = ${params.query_limit}
    if limit and limit < len(recs):
        step = len(recs) / limit
        recs = [recs[int(i * step)] for i in range(limit)]
    size = ${params.query_chunk_size}
    with open("searched_queries.fasta", "w") as all_fh:
        for i in range(0, len(recs), size):
            with open(f"chunk_{i // size:03d}.fasta", "w") as fh:
                for n, s in recs[i:i + size]:
                    fh.write(f">{n}\\n{s}\\n")
                    all_fh.write(f">{n}\\n{s}\\n")
    PYEOF
    """
}

process kmerseekIndex {
    label 'kmerseek_index'
    tag "${name}.w${window}.${slug}"

    input:
    tuple val(name), val(window), path(fasta), val(alphabet), val(ksize), val(scaled),
          val(penalty), val(xdrop), val(slug)

    output:
    tuple val(name), val(window), val(alphabet), val(ksize), val(penalty), val(xdrop),
          val(slug), path("${name}.w${window}.${slug}.kmerseek.rocksdb"), emit: index
    path "${name}.w${window}.${slug}.index.log", emit: log

    script:
    def idx = "${name}.w${window}.${slug}.kmerseek.rocksdb"
    """
    set -euo pipefail
    ${params.time_cmd} ${params.kmerseek} index --alphabet ${alphabet} --ksize ${ksize} --scaled ${scaled} \\
        --input ${fasta} --output ${idx} \\
        --extend-mismatch-penalty ${penalty} --extend-xdrop ${xdrop} \\
        --ka-queries ${params.ka_queries} --ka-survival-out ${idx}/ka_survival.C${penalty}.csv \\
        --kmer-stats-out ${idx}/spectrum.csv.gz > ${name}.w${window}.${slug}.index.log 2>&1
    ${params.kmerseek} --version >> ${name}.w${window}.${slug}.index.log
    """
}

process kmerseekSearch {
    label 'kmerseek_search'
    tag "${name}.w${window}.${slug}.${chunk.simpleName}"

    input:
    tuple val(name), val(window), val(alphabet), val(ksize), val(penalty), val(xdrop),
          val(slug), path(index_dir), path(chunk)

    output:
    tuple val("kmerseek.${indexKind(name)}.w${window}.${slug}"),
          path("${chunk.simpleName}.calls.tsv.gz"), emit: calls
    path "${chunk.simpleName}.search.log"

    script:
    // kmerseek writes regions 0-based and end-exclusive on both proteins; the shared call
    // format is 1-based and inclusive, so start + 1 and the end as written. Columns are
    // found by header name. Names hold no commas (queries are bare accessions; regions-index
    // descriptions had commas replaced), so splitting on commas is safe.
    """
    set -uo pipefail
    ${params.time_cmd} ${params.kmerseek} search --alphabet ${alphabet} --ksize ${ksize} \\
        --query ${chunk} --target ${index_dir} \\
        --extend-mismatch-penalty ${penalty} --extend-xdrop ${xdrop} --ka-queries 0 \\
        --chain-max-gap ${params.chain_max_gap} --chain-max-shift ${params.chain_max_shift} \\
        --threshold ${params.threshold} --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} \\
        --min-region-score ${params.min_region_score} 2> ${chunk.simpleName}.search.log \\
      | awk -F, -v OFS='\\t' -v max=${params.max_region_evalue} '
          NR == 1 { for (i = 1; i <= NF; i++) c[\$i] = i
                    split("query_name target_name region_start region_end target_start target_end region_evalue region_subseq target_subseq", need, " ")
                    for (n in need) if (!(need[n] in c)) { print "missing column " need[n] > "/dev/stderr"; exit 3 }
                    print "query", "qstart", "qend", "target", "tstart", "tend", "evalue", "qseq", "tseq"; next }
          (\$c["region_evalue"] + 0) <= max {
                    print \$c["query_name"], \$c["region_start"] + 1, \$c["region_end"],
                          \$c["target_name"], \$c["target_start"] + 1, \$c["target_end"],
                          \$c["region_evalue"], \$c["region_subseq"], \$c["target_subseq"] }' \\
      | gzip -c > ${chunk.simpleName}.calls.tsv.gz
    status=(\${PIPESTATUS[@]})
    if [ "\${status[0]}" -ne 0 ] && grep -q "no Karlin-Altschul fit" ${chunk.simpleName}.search.log; then
        # The index-time fit was refused, so this setting has no E-value on this index.
        # Recorded as such, never as zero calls: the scorer reads the #nofit line.
        { printf '#nofit\\n'; printf 'query\\tqstart\\tqend\\ttarget\\ttstart\\ttend\\tevalue\\tqseq\\ttseq\\n'; } \\
            | gzip -c > ${chunk.simpleName}.calls.tsv.gz
        exit 0
    fi
    for s in "\${status[@]}"; do [ "\$s" -eq 0 ] || exit 1; done
    """
}

process mmseqsSearch {
    label 'mmseqs'
    tag "${name}.w${window}"

    input:
    tuple val(name), val(window), path(fasta), path(queries)

    output:
    tuple val("mmseqs2.${indexKind(name)}.w${window}.mmseqs2"), path("calls.tsv.gz"), emit: calls
    path "mmseqs.${name}.w${window}.log"

    script:
    """
    set -euo pipefail
    ${params.mmseqs} createdb ${queries} qdb > mmseqs.${name}.w${window}.log 2>&1
    ${params.mmseqs} createdb ${fasta} tdb >> mmseqs.${name}.w${window}.log 2>&1
    ${params.time_cmd} ${params.mmseqs} search qdb tdb res tmp -s ${params.mmseqs_sensitivity} \\
        --num-iterations ${params.mmseqs_iterations} -e ${params.mmseqs_evalue} \\
        --threads ${task.cpus} >> mmseqs.${name}.w${window}.log 2>&1
    ${params.mmseqs} convertalis qdb tdb res out.tsv \\
        --format-output 'query,target,qstart,qend,tstart,tend,evalue,qaln,taln' \\
        >> mmseqs.${name}.w${window}.log 2>&1
    ${params.mmseqs} version >> mmseqs.${name}.w${window}.log
    { printf 'query\\tqstart\\tqend\\ttarget\\ttstart\\ttend\\tevalue\\tqseq\\ttseq\\n'; cat out.tsv; } \\
        | gzip -c > calls.tsv.gz
    """
}

process compositionClassifier {
    label 'python'
    tag "w${window}"

    input:
    tuple val(name), val(window), path(fasta), path(queries)

    output:
    tuple val("composition.regions.w${window}.composition"), path("calls.tsv.gz"), emit: calls

    script:
    """
    ${py('composition_classifier.py')} --queries ${queries} --index ${fasta} \\
        --window-aa ${params.composition_window} --step ${params.composition_step} \\
        --out calls.tsv.gz
    """
}

process scoreCalls {
    label 'python'
    tag "${label}"
    publishDir "${params.outdir}/scored", mode: 'copy'

    input:
    tuple val(label), path(calls, stageAs: 'calls_??????.tsv.gz'), path(truth),
          path(searched), path(target_features)

    output:
    path "${label}.features.parquet", emit: features
    path "${label}.threshold.json", emit: threshold

    script:
    def kind = label.tokenize('.')[1]
    """
    set -euo pipefail
    # One header, then every chunk's rows; a #nofit line anywhere marks the whole set.
    nofit=0
    for f in calls_*.tsv.gz; do gzip -dc "\$f" | grep -q '^#nofit' && nofit=1 || true; done
    { gzip -dc \$(ls calls_*.tsv.gz | head -1) | grep -v '^#' | head -1
      for f in calls_*.tsv.gz; do gzip -dc "\$f" | grep -v '^#' | tail -n +2; done; } > calls.tsv
    ${py('score_calls.py')} --calls calls.tsv --truth ${truth} --searched ${searched} \\
        --index-kind ${kind} --target-features ${target_features} \\
        --flank \$(( ${params.k_max} - 1 )) --max-decoy-rate ${params.max_decoy_rate} \\
        --min-landing ${params.min_landing} --label ${label} --prefix ${label} \\
        \$( [ \$nofit -eq 1 ] && echo --nofit )
    """
}

process compareIndexes {
    label 'python'
    publishDir "${params.outdir}/comparison", mode: 'copy'

    input:
    path scored
    path truth
    path searched
    path regions_fasta

    output:
    path '*.tsv'
    path 'feature_calls.parquet'

    script:
    """
    ${py('compare_indexes.py')} --scored ${scored} --truth ${truth} --queries ${searched} \\
        --regions-fasta ${regions_fasta} \\
        --headline-window w${params.decoy_windows.toString().tokenize(',')[0].trim()}
    """
}

workflow build {
    main:
    if (!params.outdir) error "--outdir is required"
    dat = file(params.swissprot_dat, checkIfExists: true)
    windows = params.decoy_windows.toString().tokenize(',')*.trim()*.toInteger()

    parseFeatures(dat)
    buildWholeReference(dat)
    buildQueryTruth(parseFeatures.out.features, parseFeatures.out.sequences,
                    file(params.chr6_queries, checkIfExists: true),
                    file(params.disprot, checkIfExists: true))
    extractRegions(parseFeatures.out.features, parseFeatures.out.sequences,
                   buildWholeReference.out.accessions)
    clusterRegions(extractRegions.out.long_fasta)
    finishRegionsIndex(extractRegions.out.regions, clusterRegions.out.tsv)

    indexes = buildWholeReference.out.fasta.map { ['whole_protein_index', it] }
        .mix(finishRegionsIndex.out.fasta.map { ['regions_index', it] })
    makeDecoys(indexes.combine(Channel.fromList(windows)))
    // The size summary reads the headline window's files.
    pair = makeDecoys.out.combined.filter { it[1] == windows[0] }
        .map { n, w, f -> [n, f] }
        .toSortedList { a, b -> b[0] <=> a[0] }
        .map { l -> l.flatten() }
    indexSummary(buildQueryTruth.out.queries, pair)

    emit:
    combined  = makeDecoys.out.combined
    queries   = buildQueryTruth.out.queries
    truth     = buildQueryTruth.out.truth
    features  = parseFeatures.out.features
    regions   = finishRegionsIndex.out.fasta
}

workflow {
    build()
    splitQueries(build.out.queries)

    settings = Channel.fromList(kmerseekSettings())
    kmerseekIndex(build.out.combined.combine(settings))
    kmerseekSearch(kmerseekIndex.out.index.combine(splitQueries.out.chunks.flatten()))

    mmseqsSearch(build.out.combined.combine(splitQueries.out.searched))
    compositionClassifier(build.out.combined.filter { it[0] == 'regions_index' }
        .combine(splitQueries.out.searched))

    calls = kmerseekSearch.out.calls.groupTuple()
        .mix(mmseqsSearch.out.calls.map { l, f -> [l, [f]] })
        .mix(compositionClassifier.out.calls.map { l, f -> [l, [f]] })
    scoreCalls(calls.combine(build.out.truth).combine(splitQueries.out.searched)
                    .combine(build.out.features))
    compareIndexes(scoreCalls.out.features.mix(scoreCalls.out.threshold).collect(),
                   build.out.truth, splitQueries.out.searched, build.out.regions)
}
