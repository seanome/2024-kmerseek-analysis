#!/usr/bin/env nextflow

/*
 * SCOPe40 all-vs-all with kmerseek 0.4.0: every alphabet at every k from notebook 274's
 * k_min to k_max, each searched with six settings, and every output score column turned
 * into family, superfamily and fold sensitivity (the FoldSeek/TEA
 * sensitivity-up-to-the-first-false-positive AUC of notebooks 066/075).
 *
 * The settings come from assets/settings_per_alphabet.tsv (scripts/scope40_v040_settings.py):
 *   exact            no extension; regions are maximal runs of shared k-mers
 *   c_min            extension at mismatch penalty C = s2 / (1 - s2): no lambda on average,
 *                    so mostly no E-value
 *   c_min_x1.1       1.1 x c_min, just above that floor
 *   c_opt            the alphabet's log-odds penalty at equal class shares
 *   c_max            the largest log-odds penalty over its classes at Swiss-Prot shares
 *   c2               C = 2, kmerseek's own default
 * funcgroups8 has no measured kappa, so no c_opt and no c_max.
 *
 * One index per alphabet-ksize pair, carrying one Karlin-Altschul fit per penalty. A fit
 * kmerseek refuses (too few score bins) stores nothing; that setting's search is then
 * refused and recorded as `nofit`, never as zero hits.
 *
 * Search and scoring are one task. At the low k-sizes one search writes up to ~30 GB of
 * region rows (funcgroups8 k7, measured 2026-10-02), so the rows are never stored: they
 * stream through bin/collapse_pairs.awk, which keeps one row per protein pair with each
 * score at its best region, and the task deletes that table once it is scored.
 *
 * Usage (see the Makefile):
 *   nextflow run main.nf -profile sherlock --fasta /path/astral-...-40-2.08.fa --outdir ...
 *   nextflow run main.nf -profile standard --pairs protein20:5 --settings_only exact,c2
 */

def home = System.getProperty('user.home')

params.fasta    = "${home}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir   = "${home}/data/scope/results-scope40-alphabet-ksize-v040"
params.settings = "${projectDir}/assets/settings_per_alphabet.tsv"
// The FoldSeek SCOPe40 result from the TEA paper (Weiss et al. 2023). Its queries are the
// reference set for the _ref AUCs: every one of them counts, a query kmerseek returns
// nothing for scores 0. Committed in the repo's data/ directory.
params.ref_rocx = "${projectDir}/../../data/tea_scope40_rocx_files/foldseek.rocx"
// The shared scoring module the notebooks use, staged into the task so the pipeline and
// notebooks 066/075 compute the AUC with the same code.
params.scope_utils = "${projectDir}/../../notebooks/scope_kmerseek_utils.py"
// Also publish each search's one-row-per-pair table (parquet, tens to hundreds of MB at
// the low k-sizes). Off by default: the AUC table is the result.
params.publish_pairs = false

// Subsets, for a test run: comma lists, empty = everything in the settings table.
params.alphabets     = ''
params.ksizes        = ''
params.settings_only = ''
// Exact alphabet:k pairs, e.g. protein20:5,gbmr7:35. Combined with the three above.
params.pairs         = ''

// The build of kmerseek's v0.4.0 tag (0aed5ca), the image the dark-set 0.4 runs use.
params.kmerseek_image = 'docker.io/olgabot/kmerseek:0.4.0-rc5'

// kmerseek search's own defaults, written out so the command line says what ran.
params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_query_pvalue = 0.05
params.min_region_score = 1.3

// X-drop scales with C so the walk ends after the same run of mismatches: X = 4 C,
// 8 at C = 2 (as in the dark-set pipeline).
params.xdrop_per_penalty = 4
params.chain_max_gap     = 30
params.chain_max_shift   = 10
// Karlin-Altschul fit queries, the dark-set rule: 500 up to 30 bits per seed
// (k x log2 classes), doubling every 3 bits above, capped at 3000.
params.ka_queries                   = 500
params.ka_queries_bits_base         = 30
params.ka_queries_bits_per_doubling = 3
params.ka_queries_max               = 3000

// Measured peak memory and run time per process and pair (tools/measure_resources.py).
params.resources_measured = "${projectDir}/assets/resources_measured.tsv"

params.max_forks         = 50
params.submit_rate_limit = '15/1min'
params.queue_size        = 200

SETTING_NAMES = ['exact', 'c_min', 'c_min_x1.1', 'c_opt', 'c_max', 'c2']

// ---- the settings table -----------------------------------------------------------------

def settingsTable() {
    def lines = file(params.settings).readLines().findAll { it.trim() && !it.startsWith('#') }
    def header = lines[0].split('\t') as List
    lines.drop(1).collect { line ->
        def f = line.split('\t', -1) as List
        [header, f].transpose().collectEntries { it }
    }
}

def csvList(v) { v.toString().tokenize(',')*.trim().findAll { it } }

// Every [alphabet, ksize, [[setting, penalty string or '' for exact], ...]] the run covers.
def resolvePairs() {
    def table = settingsTable()
    def wantA = csvList(params.alphabets)
    def wantK = csvList(params.ksizes).collect { it as int }
    def wantS = csvList(params.settings_only)
    def wantPairs = csvList(params.pairs)
    def unknownA = wantA - table*.alphabet
    if (unknownA) error "--alphabets: not in ${params.settings}: ${unknownA.join(', ')}"
    def unknownS = wantS - SETTING_NAMES
    if (unknownS) error "--settings_only takes ${SETTING_NAMES.join(', ')}, not ${unknownS.join(', ')}"
    table.findAll { !wantA || it.alphabet in wantA }.collectMany { row ->
        def settings = SETTING_NAMES.findAll { s -> s == 'exact' || row[s] }
                                    .findAll { s -> !wantS || s in wantS }
                                    .collect { s -> [s, s == 'exact' ? '' : row[s]] }
        ((row.k_min as int)..(row.k_max as int))
            .findAll { k -> !wantK || k in wantK }
            .findAll { k -> !wantPairs || "${row.alphabet}:${k}".toString() in wantPairs }
            .collect { k -> [row.alphabet, k, settings] }
    }
}

def alphabetClasses(String alphabet) {
    def m = alphabet =~ /(\d+)$/
    if (!m) error "cannot read the class count off alphabet name '${alphabet}'"
    m[0][1] as int
}

def kaQueriesFor(String alphabet, int ksize) {
    int nq = params.ka_queries as int
    double bits = ksize * Math.log(alphabetClasses(alphabet)) / Math.log(2)
    double base = params.ka_queries_bits_base as double
    if (bits <= base) return nq
    // Capped while still a double: at gbmr7 k35 (98 bits) the doubling alone is 2^23, and
    // rounding that to an int first overflowed to --ka-queries -761938446.
    double n = nq * Math.pow(2.0d, (bits - base) / (params.ka_queries_bits_per_doubling as double))
    (int) (Math.round(Math.min(params.ka_queries_max as double, n) / 50.0d) * 50)
}

def xdropFor(String penalty) {
    String.format('%.3f', (params.xdrop_per_penalty as double) * (penalty as double))
}

// ---- memory and time --------------------------------------------------------------------
//
// Sized from measured runs (params.resources_measured, written by `make measure-resources`
// from every trace file). The first attempt asks for 1.5x the measured peak (see askFor),
// never under the floor; time is 3x the measured run time, never under 1 h. A retry
// doubles both.
def measured() {
    def f = file(params.resources_measured)
    if (!f.exists()) return [:]
    def rows = f.readLines().findAll { it.trim() && !it.startsWith('#') }
    def h = rows[0].split('\t') as List
    rows.drop(1).collect { [h, it.split('\t') as List].transpose().collectEntries { it } }
        .groupBy { it.process + '|' + it.alphabet }
}

// Memory and run time fall as k grows, so the nearest measured k at or below this one is an
// upper bound. With none at or below (this k is under the alphabet's lowest measured k, or
// the alphabet was not measured), the largest value measured for any alphabet is used.
// Only completed tasks count: a memory kill says only "more than it asked", so a pair whose
// own measurement was a kill falls back to the largest completed value.
def askFor(String process, String alphabet, int ksize, String what, double floor, int attempt) {
    def all = measured().collectEntries { key, rows -> [key, rows.findAll { it.evidence == 'completed' }] }
    def rows = all[process + '|' + alphabet] ?: []
    def below = rows.findAll { (it.ksize as int) <= ksize }
    double ref = below ? (below.max { it.ksize as int }[what] as double)
                       : (all.findAll { it.key.startsWith(process + '|') }.values().flatten()
                             .collect { it[what] as double }.max() ?: 0.0d)
    Math.max(floor, 1.5 * ref) * Math.pow(2, attempt - 1)
}

def timeFor(String process, String alphabet, int ksize, int attempt) {
    double h = askFor(process, alphabet, ksize, 'realtime_h', 0.0, 1) * 2
    Math.min(48, (int) Math.ceil(Math.max(1.0d, h) * Math.pow(2, attempt - 1))) + 'h'
}

// ---- processes --------------------------------------------------------------------------

// The SCOP labels of every domain, and the FASTA with each header cut to its domain id:
// kmerseek writes the whole header into query_name, and 3_138 of the 15_177 SCOPe40 headers
// hold a comma, which would shift every column of a comma-split row after it.
process prepareScope {
    publishDir params.outdir, mode: 'copy'
    container params.kmerseek_image

    input:
    path fasta

    output:
    path 'scope_domains.tsv', emit: domains
    path 'scope40.ids.fa',    emit: fasta

    script:
    """
    prepare_scope.py ${fasta} scope_domains.tsv scope40.ids.fa
    """
}

process kmerseekIndex {
    tag "${alphabet}.k${ksize}"
    container params.kmerseek_image
    storeDir "${params.outdir}/indexes"
    memory { askFor('kmerseekIndex', alphabet, ksize as int, 'peak_gb', 8, task.attempt) + ' GB' }
    time   { timeFor('kmerseekIndex', alphabet, ksize as int, task.attempt) }

    input:
    tuple val(alphabet), val(ksize), val(penalties), path(fasta)

    output:
    tuple val(alphabet), val(ksize), path("scope40.${alphabet}.k${ksize}.kmerseek.rocksdb")

    script:
    def idx  = "scope40.${alphabet}.k${ksize}.kmerseek.rocksdb"
    def nq   = kaQueriesFor(alphabet, ksize as int)
    def first = penalties ? "--extend-mismatch-penalty ${penalties[0]} --extend-xdrop ${xdropFor(penalties[0])} " +
                            "--ka-queries ${nq} --ka-survival-out ${idx}/ka_survival.C${penalties[0]}.csv"
                          : '--ka-queries 0'
    def more = penalties.drop(1).collect { c ->
        "kmerseek calibrate --target ${idx} --extend-mismatch-penalty ${c} --extend-xdrop ${xdropFor(c)} " +
        "--ka-queries ${nq} --ka-survival-out ${idx}/ka_survival.C${c}.csv 2>&1 | tee -a ${idx}/index.log"
    }.join('\n    ')
    """
    set -euo pipefail
    mkdir -p ${idx}.tmp
    kmerseek index --alphabet ${alphabet} --ksize ${ksize} --scaled 1 \\
        --input ${fasta} --output ${idx} ${first} 2>&1 | tee ${idx}.tmp/index.log
    mv ${idx}.tmp/index.log ${idx}/index.log && rmdir ${idx}.tmp
    ${more}
    # Read-only files: a search that opened an index read-write would rewrite its
    # CURRENT and MANIFEST and cost every later search on it its cache. The directory
    # stays writable so storeDir can move it (dark-set lesson, 2026-09-19).
    chmod -R a-w ${idx}
    chmod u+w ${idx}
    """

    stub:
    """
    mkdir -p scope40.${alphabet}.k${ksize}.kmerseek.rocksdb
    touch scope40.${alphabet}.k${ksize}.kmerseek.rocksdb/CURRENT
    """
}

process searchAndScore {
    tag "${alphabet}.k${ksize}.${setting}"
    container params.kmerseek_image
    publishDir "${params.outdir}/scores", mode: 'copy', pattern: '*.{auc.tsv,log}'
    publishDir "${params.outdir}/pairs",  mode: 'copy', pattern: '*.pairs.parquet', enabled: params.publish_pairs
    memory { askFor('searchAndScore', alphabet, ksize as int, 'peak_gb', 4, task.attempt) + ' GB' }
    time   { timeFor('searchAndScore', alphabet, ksize as int, task.attempt) }

    input:
    tuple val(alphabet), val(ksize), val(setting), val(penalty), path(index), path(fasta),
          path(domains), path(ref_rocx), path(utils)

    output:
    path "${alphabet}.k${ksize}.${setting}.auc.tsv", emit: auc
    path "${alphabet}.k${ksize}.${setting}.log",     emit: log
    path "${alphabet}.k${ksize}.${setting}.pairs.parquet", optional: true, emit: pairs

    script:
    def slug = "${alphabet}.k${ksize}.${setting}"
    def ext  = penalty ? "--extend-mismatch-penalty ${penalty} --extend-xdrop ${xdropFor(penalty)} --ka-queries 0 " +
                         "--chain-max-gap ${params.chain_max_gap} --chain-max-shift ${params.chain_max_shift}"
                       : ''
    def keep = params.publish_pairs ? "--pairs-parquet ${slug}.pairs.parquet" : ''
    def score = "score_scope40_setting.py --domains ${domains} --ref-rocx ${ref_rocx} --utils ${utils} " +
                "--alphabet ${alphabet} --ksize ${ksize} --setting ${setting} --penalty '${penalty}' " +
                "--out ${slug}.auc.tsv"
    """
    set -euo pipefail
    # Which columns are scores and which way each ranks: one list, in the scoring script.
    score_scope40_setting.py --print-directions > directions.csv
    set +e
    kmerseek search --alphabet ${alphabet} --ksize ${ksize} \\
        --query ${fasta} --target ${index} ${ext} \\
        --threshold ${params.threshold} --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} --min-region-score ${params.min_region_score} \\
        2> ${slug}.log \\
      | collapse_pairs.awk directions.csv - > pairs.csv
    status=(\${PIPESTATUS[@]})
    set -e
    # A refused extension (no fit stored for this penalty) is a result, not a failure.
    if [ "\${status[0]}" -ne 0 ] && grep -q "no Karlin-Altschul fit" ${slug}.log; then
        rm -f pairs.csv
        ${score} --nofit
        exit 0
    fi
    [ "\${status[0]}" -eq 0 ] && [ "\${status[1]}" -eq 0 ] || exit 1
    ${score} --pairs pairs.csv --raw-rows "\$(cat raw_rows.txt)" ${keep}
    rm -f pairs.csv
    """

    stub:
    """
    printf 'alphabet\\tksize\\tsetting\\n${alphabet}\\t${ksize}\\t${setting}\\n' > ${alphabet}.k${ksize}.${setting}.auc.tsv
    echo stub > ${alphabet}.k${ksize}.${setting}.log
    """
}

process collectScores {
    publishDir params.outdir, mode: 'copy'
    container params.kmerseek_image

    input:
    path expected
    path tables

    output:
    path 'scope40_v040_auc.tsv'

    script:
    """
    collect_scores.py ${expected} scope40_v040_auc.tsv ${tables}
    """
}

workflow {
    def pairs = resolvePairs()
    if (!pairs) error "no alphabet-ksize pair matches --alphabets/--ksizes/--pairs"
    log.info "SCOPe40 kmerseek 0.4: ${pairs.size()} alphabet-ksize pairs, " +
             "${pairs.sum { it[2].size() }} searches"

    prepared = prepareScope(channel.fromPath(params.fasta, checkIfExists: true))
    fasta    = prepared.fasta.first()

    index_in = channel.fromList(pairs.collect { a, k, settings ->
        [a, k, settings.findAll { it[1] }.collect { it[1] }]
    }).combine(fasta)
    indexes = kmerseekIndex(index_in)

    settings_ch = channel.fromList(pairs.collectMany { a, k, settings -> settings.collect { s, c -> [a, k, s, c] } })
    scored = searchAndScore(settings_ch
        .combine(indexes, by: [0, 1])
        .combine(fasta)
        .combine(prepared.domains)
        .combine(channel.fromPath(params.ref_rocx, checkIfExists: true))
        .combine(channel.fromPath(params.scope_utils, checkIfExists: true)))

    // Every search the run was asked for, so collectScores can name any that never scored.
    expected = channel.fromList(pairs.collectMany { a, k, settings -> settings.collect { s, c -> "${a}\t${k}\t${s}\t${c}\n" } })
        .collectFile(name: 'expected_searches.tsv', seed: 'alphabet\tksize\tsetting\tpenalty\n', sort: false)
    // ifEmpty: if every search failed, the table is still written, every row `failed`.
    collectScores(expected, scored.auc.collect().ifEmpty([]))
}
