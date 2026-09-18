#!/usr/bin/env nextflow

/*
 * disprot_benchmark.nf
 *
 * Does kmerseek detect homology through intrinsically disordered regions, where a
 * structure-based tool has nothing to encode? Human DisProt proteins are the queries,
 * the nine QfO species proteomes are the targets, and the Pfam QfO pair truth is the
 * answer key, filtered to the DisProt subset. Recall is stratified by each query's
 * predicted disorder level (metapredict).
 *
 * The kmerseek arm sweeps EVERY alphabet over its entropy-derived ksize range, the same
 * matrix qfo-pfam-region-benchmark and invertebrate-dark-set sweep, from the one shared
 * table in ../shared/kmerseek_encodings.nf. 183 combos by default, 203 with
 * --kmerseek_extra_encodings, times nine targets, times the low-complexity toggle.
 *
 * The target indexes are keyed exactly as the region benchmark keys them
 * (<species>.<label>.k<ksize>.lc<lowcomp>.kmerseek.rocksdb under <db_cache>/kmerseek_index),
 * so pointing --db_cache at that pipeline's results directory on Sherlock makes every
 * index a storeDir hit. The only kmerseek work this pipeline then does is the 271-query
 * searches, which is the cheap half.
 *
 * Usage (the Makefile wraps these; see README.md):
 *   nextflow run main.nf -profile sherlock --disprot_tsv <frozen tsv> --db_cache <dir>
 *   nextflow run main.nf -profile standard --kmerseek_combos hp_thomas_dill2:26
 *   nextflow run main.nf -profile stub -stub-run
 */

nextflow.enable.dsl=2

// One copy of the alphabet x ksize matrix, shared with the region benchmark and the dark
// set. A second literal here is a copy nothing checks against the original.
include { allEncodings; extraEncodings; knownEncodings; expandEncodings; keyspaceBits } \
    from '../shared/kmerseek_encodings'

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

def home = System.getProperty('user.home')

params.qfo_dir          = "${home}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.pfam_pairs_dir   = "${projectDir}/../../results/pfam_benchmark/pairs"
params.database         = "disprot"          // "disprot" | "mobidb"
params.outdir           = "${home}/data/disprot-benchmark/${params.database}/results"
params.alphafold_cache  = "${home}/data/alphafold_structures"

// Where the target-side kmerseek indexes live. Defaults to outdir. On Sherlock this is the
// region benchmark's results directory, whose kmerseek_index/ already holds one entry per
// (target, alphabet, ksize, low-complexity) cell of the same matrix.
params.db_cache         = null

// The DisProt query set, three ways. --disprot_tsv is the frozen parse this pipeline
// already published on a machine with internet (disprot/disprot_human.tsv under outdir),
// and is the only one of the three that works on Sherlock, whose compute nodes have no
// outbound internet. --disprot_json is a raw API dump to parse locally; null downloads.
// A frozen TSV also pins the DisProt release: the API's "current" moves.
params.disprot_tsv      = null
params.disprot_json     = null
params.mobidb_json      = null
params.mobidb_source    = "curated"

// ---- kmerseek: the sweep ---------------------------------------------------------------
//
// Bare run: every row of allEncodings() over its own ksize range.
//   --kmerseek_encodings hp_lehninger2,gbmr4   a subset of alphabets, each keeping its
//                                             full ksize range
//   --kmerseek_extra_encodings true            add polarity4 and funcgroups8
//   --kmerseek_combos hp_thomas_dill2:26       named alphabet:ksize pairs, replacing the
//                                             matrix outright (the old single-arm run)
//   --low_complexity_toggle false,true         run the mask off AND on; doubles the sweep
params.kmerseek_encodings       = null
params.kmerseek_extra_encodings = false
params.kmerseek_combos          = null
params.low_complexity_toggle    = [false]

// A subset of the nine targets, by label (`--target_species mouse,ecoli`), for a smoke
// test. Human is the query and is never a target.
params.target_species           = null

// params.kmerseek_image lives in nextflow.config with the other two images: the config
// is parsed before this script, so a param the config's selectors read has to be
// declared there.

// The search flags the region benchmark uses, so a region here and a region there are the
// same object. --min-region-score is OR'd with --max-query-pvalue inside kmerseek.
params.threshold         = 0.0
params.min_shared_kmers  = 2
params.max_query_pvalue  = 0.05
params.min_region_score  = 1.3

// How a protein PAIR is scored from kmerseek's region table. The unit here is the pair,
// not the region, so the default is the whole-pair statistic this benchmark has always
// ranked on (best max_containment per pair). region_enrichment, the region benchmark's
// choice, is available; it is max'd over the pair's regions.
params.kmerseek_rank_by          = 'max_containment'
// Bonferroni filter on the region's Poisson tail, as the region benchmark applies it:
// raw p x region_search_space x db_n_targets < this. 0 disables it.
params.kmerseek_max_bonferroni_p = 0.05

// ---- baselines -------------------------------------------------------------------------
params.mmseqs2_sensitivity = 7
// Foldseek needs the AlphaFold download and a foldseek binary this pipeline only has on
// the laptop, and every make target already passed --skip_foldseek. Off unless asked.
params.skip_foldseek     = true
params.evalue_report     = 10.0

// ---- kmerseek memory, the region benchmark's model -------------------------------------
//
// Fit on 4_299 completed search tasks over these same nine target proteomes (region
// benchmark, 2026-08-25..27): peak GB = headroom x 300 x exp(-0.1017 x keyspace bits),
// scaled by target proteome size with a floor fraction, and a staircase of measured
// floors for the two alphabets the model does not fit. The query set here is 271
// proteins against the region benchmark's 964-19_696, so this over-asks, which is the
// safe direction. See qfo-pfam-region-benchmark/main.nf for the derivation.
params.kmerseek_memory_max            = '128 GB'
params.kmerseek_memory_floor          = '10 GB'
params.kmerseek_index_memory_max      = '16 GB'
params.kmerseek_reference_proteome_mb = 17
params.kmerseek_memory_bits_base      = 300
params.kmerseek_memory_bits_decay     = 0.1017
params.kmerseek_memory_headroom       = 2.3
params.kmerseek_memory_size_floor_frac = 0.55
params.kmerseek_memory_unmodelled     = 'gbmr7:96:14,gbmr7:44:16,gbmr7:22:18,gbmr4:48:13'

// ---------------------------------------------------------------------------
// Species: identical to the Pfam benchmark pipeline
// ---------------------------------------------------------------------------

def ALL_SPECIES = [
    [label: "mouse",       taxon: "10090",  proteome: "UP000000589", subdir: "Eukaryota", mya: 100],
    [label: "chicken",     taxon: "9031",   proteome: "UP000000539", subdir: "Eukaryota", mya: 300],
    [label: "zebrafish",   taxon: "7955",   proteome: "UP000000437", subdir: "Eukaryota", mya: 430],
    [label: "ciona",       taxon: "7719",   proteome: "UP000008144", subdir: "Eukaryota", mya: 550],
    [label: "fly",         taxon: "7227",   proteome: "UP000000803", subdir: "Eukaryota", mya: 600],
    [label: "worm",        taxon: "6239",   proteome: "UP000001940", subdir: "Eukaryota", mya: 650],
    [label: "yeast",       taxon: "559292", proteome: "UP000002311", subdir: "Eukaryota", mya: 900],
    [label: "arabidopsis", taxon: "3702",   proteome: "UP000006548", subdir: "Eukaryota", mya: 1500],
    [label: "ecoli",       taxon: "83333",  proteome: "UP000000625", subdir: "Bacteria",  mya: 2000],
]

def SPECIES = ALL_SPECIES
if (params.target_species) {
    def wanted  = params.target_species.toString().tokenize(',')*.trim().findAll { it }
    def unknown = wanted - ALL_SPECIES*.label
    // Named-but-unknown is an error, not a silent drop.
    if (unknown) {
        error "Unknown target species: ${unknown.join(', ')}. Known: ${ALL_SPECIES*.label.join(', ')}" +
              (unknown.contains('human') ? ". human is the QUERY and is never a target." : "")
    }
    SPECIES = ALL_SPECIES.findAll { it.label in wanted }
}

// ---------------------------------------------------------------------------
// The sweep: which (cli_flag, label, ksize, lowcomp) cells run
// ---------------------------------------------------------------------------

def LC_TOGGLE = (params.low_complexity_toggle instanceof List
                 ? params.low_complexity_toggle
                 : params.low_complexity_toggle.toString().tokenize(','))
    .collect { v ->
        def s = v.toString().trim().toLowerCase()
        if (!(s in ['true', 'false'])) {
            error "--low_complexity_toggle takes true and/or false, not '${v}'"
        }
        s == 'true'
    }.unique()
if (LC_TOGGLE.isEmpty()) {
    error "--low_complexity_toggle is empty; it needs at least one of true, false"
}

def KNOWN = knownEncodings()

def selectedEncodings = {
    if (params.kmerseek_encodings) {
        def wanted  = params.kmerseek_encodings.toString().tokenize(',')*.trim().findAll { it }
        def unknown = wanted - KNOWN*.get(0)
        if (unknown) {
            error "Unknown encoding(s) in --kmerseek_encodings: ${unknown.join(', ')}. " +
                  "Known: ${KNOWN*.get(0).join(', ')}"
        }
        // Naming an extra alphabet selects it; --kmerseek_extra_encodings is not also needed.
        return wanted.collect { w -> KNOWN.find { it[0] == w } }
    }
    params.kmerseek_extra_encodings ? KNOWN : allEncodings()
}

// (cli_flag, label, ksize, lowcomp). An explicit combo list replaces the matrix outright,
// ksizes included.
def COMBOS = (params.kmerseek_combos
    ? params.kmerseek_combos.toString().tokenize(',').collect { spec ->
          def parts = spec.trim().split(':')
          if (parts.size() != 2) {
              error "--kmerseek_combos entries are alphabet:ksize, not '${spec}'"
          }
          def known = KNOWN.find { it[0] == parts[0] }
          if (!known) {
              error "Unknown encoding '${parts[0]}' in --kmerseek_combos. Known: ${KNOWN*.get(0).join(', ')}"
          }
          [parts[0], known[1], parts[1].toInteger()]
      }
    : expandEncodings(selectedEncodings())
).collectMany { cli, label, k -> LC_TOGGLE.collect { lc -> [cli, label, k, lc] } }

// A repeated combo puts two tasks on one storeDir entry, and the loser dies at unstage
// with "Directory not empty". Error, not dedup: a repeat is usually a wrong list.
def dupCombos = COMBOS.countBy { _cli, label, k, lc -> "${label}.k${k}.lc${lc}" }
    .findAll { _key, n -> n > 1 }*.key
if (dupCombos) {
    error "Duplicate kmerseek combos: ${dupCombos.join(', ')}. Check --kmerseek_combos " +
          "and --kmerseek_encodings for repeats."
}

def DB_CACHE = params.db_cache ?: params.outdir

// The tool name every downstream table carries. Parsed back into alphabet, ksize and
// low-complexity by bin/disprot_tool_names.py, so the two must agree.
def toolName = { label, k, lc -> "kmerseek_${label}_k${k}_lc${lc}" }

// ---------------------------------------------------------------------------
// Error handling and memory
// ---------------------------------------------------------------------------

// A task the cluster killed reports one of these; nothing else is a kill. Integer.MAX_VALUE
// is what Nextflow records when no .exitcode was written (cgroup OOM, walltime, node
// failure, preemption) and it is NOT in 128..143.
def killedByCluster = { task ->
    def status = task.exitStatus
    status == null || status == Integer.MAX_VALUE || status in 128..143
}

// Retry a kill, and hand everything else -- an exhausted retry included -- to `finish`,
// which stops submitting but lets running tasks complete. Never `terminate`, whose first
// failure cancels every task in flight.
def retryOnKill = { task, int retries = 2 ->
    (killedByCluster(task) && task.attempt <= retries) ? 'retry' : 'finish'
}

// kmerseekSearch's variant: a kill that has used its retries is ignored. An infeasible
// low-k reduced-alphabet combo is an expected outcome of the sweep, not a broken run, and
// an ignored search stores nothing, so -resume tries it again rather than serving a zero.
def retryOnKillElseIgnore = { task, int retries = 2 ->
    if (!killedByCluster(task)) return 'finish'
    if (task.attempt <= retries) return 'retry'
    log.warn "${task.process} (${task.tag}) was killed on attempt ${task.attempt} of " +
             "${retries + 1} at ${task.memory}; ignoring it. Nothing is stored for this " +
             "combo, so -resume will try it again."
    'ignore'
}

def kmerseekUnmodelledFloorMb = { label, ksize ->
    def brackets = (params.kmerseek_memory_unmodelled as String).tokenize(',')
        .collect { it.trim() }.findAll { it }
        .collect { it.tokenize(':') }
        .findAll { it[0] == label }
        .findAll { it.size() < 3 || (ksize as int) <= (it[2] as int) }
    if (!brackets) return 0L
    def best = brackets.min { it.size() > 2 ? (it[2] as int) : Integer.MAX_VALUE }
    (long) ((best[1] as double) * 1024L)
}

// Index memory tracks the target proteome alone: 2 GB + 0.9 GB per MB of FASTA.
def kmerseekIndexMemory = { targetBytes, attempt ->
    long mb      = Math.max(1L, (targetBytes as long).intdiv(1024L * 1024L))
    long estMb   = (long) ((2.0d + 0.9d * mb) * 1024L)
    long capMb   = Math.min(MemoryUnit.of(params.kmerseek_index_memory_max).toMega(),
                            MemoryUnit.of(params.kmerseek_memory_max).toMega())
    long floorMb = Math.min(capMb, MemoryUnit.of(params.kmerseek_memory_floor).toMega())
    MemoryUnit.of("${Math.max(floorMb, Math.min(capMb, estMb))} MB") * attempt
}

def kmerseekSearchMemory = { label, ksize, targetBytes, attempt ->
    long mb      = Math.max(1L, (targetBytes as long).intdiv(1024L * 1024L))
    double frac  = params.kmerseek_memory_size_floor_frac as double
    double sizeF = frac + (1.0d - frac) *
                   Math.min(1.0d, mb / (params.kmerseek_reference_proteome_mb as double))
    double gb    = (params.kmerseek_memory_headroom as double)
                   * (params.kmerseek_memory_bits_base as double)
                   * Math.exp(-(params.kmerseek_memory_bits_decay as double)
                              * keyspaceBits(label, ksize as int))
                   * sizeF
    long estMb   = (long) (gb * 1024L)
    long capMb   = MemoryUnit.of(params.kmerseek_memory_max).toMega()
    long floorMb = Math.min(capMb,
                            Math.max(MemoryUnit.of(params.kmerseek_memory_floor).toMega(),
                                     kmerseekUnmodelledFloorMb(label, ksize)))
    MemoryUnit.of("${Math.max(floorMb, Math.min(capMb, estMb))} MB") * attempt
}

// The one storeDir failure this repository keeps meeting, diagnosed where it happens.
workflow.onError {
    if (workflow.errorReport?.contains('Directory not empty')) {
        log.error """
        |
        |Two writers built the same storeDir entry under ${DB_CACHE}/kmerseek_index.
        |Nextflow reads the store when it CREATES a task and never re-checks, so both did
        |the full build and the loser's `mv` failed. The winner's entry is complete.
        |Likely second writer: tasks left in the queue by a previous run (`squeue -u \$USER`),
        |or the region benchmark running against the same --db_cache. Wait for the queue
        |to drain, then re-run with -resume; the entry is now a store hit.
        """.stripMargin()
    }
}

// ---------------------------------------------------------------------------
// PROCESS 1 -- The DisProt (or MobiDB) human query set
// ---------------------------------------------------------------------------

process downloadDisprot {
    label 'python'
    publishDir "${params.outdir}/disprot", mode: 'copy'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    val disprot_json_path   // null -> download; path string -> use file

    output:
    path "disprot_human.tsv"

    script:
    def use_local = (disprot_json_path != null && disprot_json_path != "null")
    def local_arg = use_local ? "--local ${disprot_json_path}" : ""
    """
    parse_disprot.py ${local_arg} disprot_human.tsv
    """

    stub:
    """
    printf 'uniprot_acc\\tdisprot_id\\n' > disprot_human.tsv
    """
}

process downloadMobidb {
    label 'python'
    publishDir "${params.outdir}/mobidb", mode: 'copy'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    val mobidb_json_path

    output:
    path "mobidb_human.tsv"

    script:
    def use_local = (mobidb_json_path != null && mobidb_json_path != "null")
    def local_arg  = use_local ? "--local ${mobidb_json_path}" : ""
    """
    parse_mobidb.py ${local_arg} --source ${params.mobidb_source} mobidb_human.tsv
    """
}

// ---------------------------------------------------------------------------
// PROCESS 2 -- Map DisProt proteins to the Pfam pair truth
// ---------------------------------------------------------------------------

process mapDisprotToPfam {
    label 'python'
    publishDir "${params.outdir}/disprot", mode: 'copy'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path disprot_tsv
    path pfam_pairs_dir

    output:
    path "disprot_pfam_mapping.tsv"

    script:
    """
    map_disprot_pfam.py ${disprot_tsv} ${pfam_pairs_dir} disprot_pfam_mapping.tsv
    """

    stub:
    """
    touch disprot_pfam_mapping.tsv
    """
}

// ---------------------------------------------------------------------------
// PROCESS 3 -- Ground truth per species + the query FASTA
// ---------------------------------------------------------------------------

process buildDisprotGroundTruth {
    label 'python'
    publishDir "${params.outdir}/disprot", mode: 'copy'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path disprot_pfam_mapping
    path pfam_pairs_dir
    path human_fasta

    output:
    path "gt/human_vs_*_ground_truth.parquet", emit: gt_parquets
    path "disprot_benchmark_queries.fasta",    emit: query_fasta
    path "benchmark_stats.txt",                emit: stats

    script:
    """
    build_disprot_ground_truth.py \\
        ${disprot_pfam_mapping} \\
        ${pfam_pairs_dir} \\
        ${human_fasta} \\
        gt/ \\
        disprot_benchmark_queries.fasta \\
        benchmark_stats.txt
    """

    stub:
    """
    mkdir -p gt
    for sp in ${SPECIES*.label.join(' ')}; do touch gt/human_vs_\${sp}_ground_truth.parquet; done
    printf '>P00001\\nMKTAYIAKQRQISFVKSHFSRQ\\n' > disprot_benchmark_queries.fasta
    echo 'stub' > benchmark_stats.txt
    """
}

// ---------------------------------------------------------------------------
// PROCESS 4 -- Per-residue disorder with metapredict
// ---------------------------------------------------------------------------

process predictDisorder {
    // Container, cpus and memory come from `withName: predictDisorder` in nextflow.config;
    // a body directive would lose to any config selector.
    publishDir "${params.outdir}/disprot", mode: 'copy'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path query_fasta

    output:
    path "query_disorder_scores.tsv"

    script:
    """
    predict_disorder.py ${query_fasta} query_disorder_scores.tsv
    """

    stub:
    """
    printf 'uniprot_acc\\tmean_disorder\\tdisorder_category\\nP00001\\t0.5\\tpartial\\n' > query_disorder_scores.tsv
    """
}

// ---------------------------------------------------------------------------
// PROCESS 5 -- kmerseek: one stored index per (target, combo), then search
// ---------------------------------------------------------------------------

process kmerseekIndex {
    /*
     * Build the RocksDB index for one target proteome under one alphabet/ksize/mask
     * setting and KEEP it under storeDir, named exactly as qfo-pfam-region-benchmark
     * names its own. Same targets, same matrix, same name: with --db_cache pointed at
     * that pipeline's results directory every one of these is a store hit.
     *
     * ONE output, the directory, with the spectrum and the log nested INSIDE it. A
     * sibling file next to a directory output under storeDir is the shape behind this
     * repository's recurring "Directory not empty" failure.
     */
    tag "${species}_${label}_k${ksize}_lc${lowcomp}"
    container params.kmerseek_image
    storeDir "${DB_CACHE}/kmerseek_index"
    memory { kmerseekIndexMemory(species_fasta.size(), task.attempt) }
    // Kills only. Retrying exit 1 does not recover the unstage collision (measured in the
    // region benchmark, 2026-08-27): the store decision is cached on the task.
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    tuple val(species), path(species_fasta), val(cli_flag), val(label), val(ksize), val(lowcomp)

    output:
    path "${species}.${label}.k${ksize}.lc${lowcomp}.kmerseek.rocksdb"

    script:
    def slug      = "${label}.k${ksize}.lc${lowcomp}"
    def index_dir = "${species}.${slug}.kmerseek.rocksdb"
    def spectrum  = "spectrum.${species}.${slug}.csv.gz"
    def lc_flag   = lowcomp ? "--remove-low-complexity" : ""
    """
    set -euo pipefail
    echo "=== Index: ${species} ${cli_flag} k=${ksize} lc=${lowcomp} ===" | tee index.log
    kmerseek index \\
        --alphabet ${cli_flag} \\
        --ksize    ${ksize} \\
        --input    ${species_fasta} \\
        --output   ${index_dir} \\
        ${lc_flag} \\
        --kmer-stats-out ${spectrum} \\
        2>&1 | tee -a index.log
    echo "index size: \$(du -sh ${index_dir} | cut -f1)" | tee -a index.log
    touch ${spectrum}
    mv ${spectrum} index.log ${index_dir}/
    """

    stub:
    """
    d=${species}.${label}.k${ksize}.lc${lowcomp}.kmerseek.rocksdb
    mkdir -p \$d && touch \$d/CURRENT
    """
}

process kmerseekSearch {
    /*
     * Search the DisProt queries against one stored target index. The regions table is
     * kept under storeDir so a relaunched run, from a fresh clone or a purged work/,
     * serves it back rather than searching again.
     *
     * target_bytes is the target FASTA size passed as a value: .size() on the staged
     * index directory returns the dirent size, not the tree, and sizing memory off that
     * gave every search the floor allocation.
     */
    tag "${species}_${label}_k${ksize}_lc${lowcomp}"
    container params.kmerseek_image
    storeDir "${params.outdir}/kmerseek"
    memory { kmerseekSearchMemory(label, ksize, target_bytes, task.attempt) }
    errorStrategy { retryOnKillElseIgnore(task) }
    maxRetries 2

    input:
    tuple val(species), val(cli_flag), val(label), val(ksize), val(lowcomp),
          val(target_bytes), path(index_dir), path(query_fasta)

    output:
    path "human_vs_${species}.${label}.k${ksize}.lc${lowcomp}.regions.csv.zst"

    script:
    def slug    = "${label}.k${ksize}.lc${lowcomp}"
    def out_zst = "human_vs_${species}.${slug}.regions.csv.zst"
    def lc_flag = lowcomp ? "--remove-low-complexity" : ""
    """
    set -euo pipefail
    set +e
    kmerseek search \\
        --alphabet ${cli_flag} \\
        --ksize    ${ksize} \\
        --query    ${query_fasta} \\
        --target   ${index_dir} \\
        ${lc_flag} \\
        --threshold         ${params.threshold} \\
        --min-shared-kmers  ${params.min_shared_kmers} \\
        --max-query-pvalue  ${params.max_query_pvalue} \\
        --min-region-score  ${params.min_region_score} \\
        2> search.log \\
        | zstd -T2 -o ${out_zst}
    rc=(\${PIPESTATUS[@]})
    set -e
    # zstd failing means a truncated stream, not a short result; re-raise it as a kill so
    # the retry ladder doubles the allocation. kmerseek's own non-zero exit is a no-hit
    # result and stays tolerated.
    if [ "\${rc[1]}" -ne 0 ]; then
        echo "zstd exited \${rc[1]}: the region stream is truncated" >&2
        exit 137
    fi
    if [ "\${rc[0]}" -ne 0 ]; then
        echo "note: kmerseek search exited \${rc[0]}; treating as a no-hit result" >&2
        tail -20 search.log >&2 || true
    fi
    touch ${out_zst}
    """

    stub:
    """
    printf 'query_name,target_name,max_containment,region_enrichment,region_tail_probability,region_search_space,db_n_targets\\nP00001,${species}_X,0.5,3.0,1e-9,100,1000\\n' \\
        | zstd -o human_vs_${species}.${label}.k${ksize}.lc${lowcomp}.regions.csv.zst
    """
}

// The pair table the evaluator reads: query, target, score, corrected p. Its own process,
// in the Python image, so the ranking column can change without re-searching.
process formatKmerseekResults {
    tag "${species}_${label}_k${ksize}_lc${lowcomp}"
    label 'python'
    // One directory for all 1_647+ pair tables; the tool name is in the filename.
    publishDir "${params.outdir}/kmerseek_pairs", mode: 'copy', pattern: '*.tsv.gz'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    tuple val(species), val(label), val(ksize), val(lowcomp), path(regions_zst)

    output:
    tuple val(species), val("${toolName(label, ksize, lowcomp)}"),
          path("human_vs_${species}.${toolName(label, ksize, lowcomp)}.tsv.gz")

    script:
    def tool = toolName(label, ksize, lowcomp)
    """
    set -euo pipefail
    # Decompressed in the shell: polars inflates a zstd CSV whole in RAM, and reading the
    # plain stream keeps the same code path for every size. From stdin, because zstd
    # refuses a symlink by name and Nextflow stages inputs as symlinks.
    zstd -dc < ${regions_zst} > regions.csv
    format_kmerseek_results.py regions.csv human_vs_${species}.${tool}.tsv.gz \\
        --rank-by ${params.kmerseek_rank_by} \\
        --max-bonferroni-p ${params.kmerseek_max_bonferroni_p}
    """

    stub:
    """
    printf 'P00001\\t${species}_X\\t0.5\\t1e-5\\n' | gzip -c > human_vs_${species}.${toolName(label, ksize, lowcomp)}.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESS 6 -- Foldseek (laptop only; off by default)
// ---------------------------------------------------------------------------

process downloadAlphaFoldStructures {
    tag "${label}"
    label 'python'
    publishDir "${params.alphafold_cache}", mode: 'copy', saveAs: { f -> f }

    input:
    tuple val(label), path(fasta)

    output:
    tuple val(label), path("structures/${label}/")

    script:
    """
    mkdir -p structures/${label}
    download_alphafold.py \\
        --fasta        ${fasta} \\
        --outdir       structures/${label} \\
        --cache        ${params.alphafold_cache} \\
        --max-workers  5
    """
}

process foldseekSearch {
    tag "human_vs_${species}"
    label 'medium_cpu'
    publishDir "${params.outdir}/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(query_structs), path(target_structs)

    output:
    tuple val(species), val("foldseek"), path("human_vs_${species}.foldseek.tsv.gz")

    script:
    """
    mkdir -p foldseek_tmp
    n_query=\$(find ${query_structs} -name '*.cif' | wc -l)
    n_target=\$(find ${target_structs} -name '*.cif' | wc -l)
    if [ "\$n_query" -eq 0 ] || [ "\$n_target" -eq 0 ]; then
        echo "WARNING: no CIF files in query (\$n_query) or target (\$n_target) -- writing empty results" >&2
        touch human_vs_${species}.foldseek.tsv
    else
        /Users/olga/anaconda3/envs/foldseek-10.941cd33/bin/foldseek easy-search \\
            ${query_structs} \\
            ${target_structs} \\
            human_vs_${species}.foldseek.tsv \\
            foldseek_tmp \\
            --format-output "query,target,bits,evalue" \\
            --threads ${task.cpus} \\
            -e ${params.evalue_report}
    fi
    gzip -c human_vs_${species}.foldseek.tsv > human_vs_${species}.foldseek.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESS 7 -- MMseqs2 baseline
// ---------------------------------------------------------------------------

process mmseqs2EasySearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/mmseqs2:18.8cc5c--hd6d6fdc_0'
    label 'medium_cpu'
    publishDir "${params.outdir}/mmseqs2", mode: 'copy', pattern: '*.tsv.gz'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    tuple val(species), path(species_fasta), path(query_fasta)

    output:
    tuple val(species), val("mmseqs2"), path("human_vs_${species}.mmseqs2.tsv.gz")

    script:
    """
    mkdir -p mmseqs_tmp
    mmseqs easy-search \\
        ${query_fasta} \\
        ${species_fasta} \\
        human_vs_${species}.mmseqs2.tsv \\
        mmseqs_tmp \\
        --threads ${task.cpus} \\
        -s ${params.mmseqs2_sensitivity} \\
        --max-seqs 1000 \\
        --format-output "query,target,bits,evalue"
    gzip -c human_vs_${species}.mmseqs2.tsv > human_vs_${species}.mmseqs2.tsv.gz
    """

    stub:
    """
    printf 'P00001\\t${species}_X\\t50\\t1e-9\\n' | gzip -c > human_vs_${species}.mmseqs2.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESS 8 -- Evaluate one (tool, species): metrics per disorder stratum
// ---------------------------------------------------------------------------

process evaluateDisprotBenchmark {
    tag "${tool} vs ${species}"
    label 'python'
    publishDir "${params.outdir}/metrics", mode: 'copy', pattern: '*.parquet'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    tuple val(species), val(tool), path(results_tsv_gz), path(ground_truth), path(disorder_scores)

    output:
    path "${tool}.${species}.disprot_metrics.parquet"
    path "${tool}.${species}.disprot_pr_curve.parquet"

    script:
    """
    evaluate_disprot_benchmark.py \\
        ${results_tsv_gz} \\
        ${species} \\
        ${ground_truth} \\
        ${tool} \\
        ${disorder_scores} \\
        ${tool}.${species}.disprot_metrics.parquet \\
        ${tool}.${species}.disprot_pr_curve.parquet
    """

    stub:
    """
    touch ${tool}.${species}.disprot_metrics.parquet ${tool}.${species}.disprot_pr_curve.parquet
    """
}

// ---------------------------------------------------------------------------
// PROCESS 9 -- Aggregate: every tool x species x stratum, plus the sweep summary
// ---------------------------------------------------------------------------

process aggregateDisprotMetrics {
    label 'python'
    publishDir params.outdir, mode: 'copy'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path 'metrics/*'

    output:
    path "all_disprot_metrics.parquet",   emit: metrics
    path "all_disprot_pr_curves.parquet", emit: pr_curves
    path "kmerseek_sweep_summary.parquet", emit: sweep
    path "figures/",                       emit: figures

    script:
    """
    aggregate_disprot_metrics.py \\
        metrics \\
        all_disprot_metrics.parquet \\
        all_disprot_pr_curves.parquet \\
        figures/ \\
        --sweep-out kmerseek_sweep_summary.parquet
    """

    stub:
    """
    mkdir -p figures
    touch all_disprot_metrics.parquet all_disprot_pr_curves.parquet kmerseek_sweep_summary.parquet
    """
}

// ---------------------------------------------------------------------------
// PROCESS 10 -- Markdown report
// ---------------------------------------------------------------------------

process generateReport {
    label 'python'
    publishDir params.outdir, mode: 'copy'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path metrics_parquet
    path figures_dir

    output:
    path "disprot_benchmark_report.md"

    script:
    """
    generate_report.py ${metrics_parquet} ${figures_dir} disprot_benchmark_report.md
    """

    stub:
    """
    touch disprot_benchmark_report.md
    """
}

// ---------------------------------------------------------------------------
// PROCESS 11 -- MultiQC: inputs in the Python image, the report in MultiQC's own
// ---------------------------------------------------------------------------

process buildMultiqcInputs {
    label 'python'
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path metrics_parquet

    output:
    path "mqc_input/"

    script:
    """
    make_multiqc_input.py ${metrics_parquet} mqc_input/
    """

    stub:
    """
    mkdir -p mqc_input && touch mqc_input/multiqc_config.yaml
    """
}

process multiQC {
    // multiqc 1.35, pinned by digest like the region benchmark's.
    container 'quay.io/biocontainers/multiqc@sha256:b65e3fe879df27b92334dda0fd987a6e21bdee09a2848551d4f287099a93b7ac'
    publishDir params.outdir, mode: 'copy'
    // The report is the last thing a run does. Failing a finished sweep over a plot would
    // throw the sweep away, so this arm may fail and the run still ends green.
    errorStrategy 'ignore'

    input:
    path mqc_input

    output:
    path "multiqc_report.html"
    path "multiqc_data/"

    script:
    // No outbound internet on Sherlock compute nodes, so the version check only costs a
    // timeout. MPLCONFIGDIR keeps matplotlib's cache inside the task dir; under Apptainer
    // $HOME can be read-only.
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    multiqc ${mqc_input}/ \\
        --config ${mqc_input}/multiqc_config.yaml \\
        --outdir . \\
        --filename multiqc_report.html \\
        --no-version-check \\
        --no-ansi \\
        --force
    """

    stub:
    """
    touch multiqc_report.html && mkdir -p multiqc_data
    """
}

// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")

    species_ch = Channel.fromList(
        SPECIES.collect { s ->
            tuple(s.label, file("${params.qfo_dir}/${s.subdir}/${s.proteome}_${s.taxon}.fasta"))
        }
    )

    log.info """
    |  query   : human DisProt proteins (${params.database})
    |  targets : ${SPECIES*.label.join(', ')}
    |  alphabet: ${COMBOS.collect { it[1] }.unique().join(', ')}
    |  combos  : ${COMBOS.size()} (alphabet x ksize x low-complexity ${LC_TOGGLE.join('/')})
    |  searches: ${COMBOS.size()} x ${SPECIES.size()} targets = ${COMBOS.size() * SPECIES.size()}
    |  indexes : ${DB_CACHE}/kmerseek_index (storeDir; shared with the region benchmark)
    |  score   : best ${params.kmerseek_rank_by} per pair, Bonferroni p < ${params.kmerseek_max_bonferroni_p}
    """.stripMargin()

    // -----------------------------------------------------------------------
    // Steps 1-3: the query set and its truth
    // -----------------------------------------------------------------------
    if (params.database == "mobidb") {
        disprot_raw = downloadMobidb(params.mobidb_json ?: "null")
    } else if (params.disprot_tsv) {
        def tsv = file(params.disprot_tsv)
        if (!tsv.exists()) error "--disprot_tsv ${params.disprot_tsv} does not exist"
        disprot_raw = Channel.value(tsv)
    } else {
        disprot_raw = downloadDisprot(params.disprot_json ?: "null")
    }

    disprot_mapping = mapDisprotToPfam(disprot_raw, file(params.pfam_pairs_dir))

    gt_out = buildDisprotGroundTruth(disprot_mapping, file(params.pfam_pairs_dir), human_fasta)
    gt_out.stats.subscribe { f -> log.info "Benchmark stats:\n" + f.text }

    disprot_gt_ch = gt_out.gt_parquets
        .flatten()
        .map { f ->
            def m = (f.name =~ /human_vs_(.+)_ground_truth\.parquet/)
            tuple(m ? m[0][1] : f.baseName, f)
        }

    query_fasta = gt_out.query_fasta

    // -----------------------------------------------------------------------
    // Step 4: disorder scores
    // -----------------------------------------------------------------------
    disorder_scores = predictDisorder(query_fasta)

    // -----------------------------------------------------------------------
    // Step 5: kmerseek sweep -- index per (target, combo), search, pair table
    // -----------------------------------------------------------------------
    def comboKey = { sp, lab, k, lc -> "${sp}|${lab}|${k}|${lc}".toString() }

    kmerseek_in = species_ch.combine(Channel.fromList(COMBOS))
        .map { species, fasta, cli_flag, label, ksize, lowcomp ->
            tuple(species, fasta, cli_flag, label, ksize, lowcomp)
        }
    idx_out = kmerseekIndex(kmerseek_in)

    // storeDir drops the tuple, so the key is rebuilt from the directory name and joined
    // back to recover cli_flag and the target FASTA size.
    combo_meta = kmerseek_in.map { species, fasta, cli_flag, label, ksize, lowcomp ->
        tuple(comboKey(species, label, ksize, lowcomp),
              species, cli_flag, label, ksize, lowcomp, fasta.size())
    }

    search_in = idx_out
        .map { d ->
            def m = (d.name =~ /^(.+?)\.(.+)\.k(\d+)\.lc(true|false)\.kmerseek\.rocksdb$/)
            if (!m) error "unexpected index directory name: ${d.name}"
            tuple(comboKey(m[0][1], m[0][2], m[0][3].toInteger(), m[0][4] == 'true'), d)
        }
        .join(combo_meta)
        .combine(query_fasta)
        .map { _key, d, species, cli_flag, label, ksize, lowcomp, target_bytes, qf ->
            tuple(species, cli_flag, label, ksize, lowcomp, target_bytes, d, qf)
        }
    regions_out = kmerseekSearch(search_in)

    format_in = regions_out.map { f ->
        def m = (f.name =~ /^human_vs_(.+?)\.(.+)\.k(\d+)\.lc(true|false)\.regions\.csv\.zst$/)
        if (!m) error "unexpected regions file name: ${f.name}"
        tuple(m[0][1], m[0][2], m[0][3].toInteger(), m[0][4] == 'true', f)
    }
    kmerseek_results = formatKmerseekResults(format_in)

    // -----------------------------------------------------------------------
    // Step 6: Foldseek (laptop only)
    // -----------------------------------------------------------------------
    foldseek_results = Channel.empty()
    if (!params.skip_foldseek) {
        all_for_af = query_fasta.map { f -> tuple("human", f) }.mix(species_ch)
        all_structs_ch = downloadAlphaFoldStructures(all_for_af)
        human_structs_ch   = all_structs_ch.filter { label, _structs -> label == "human" }
        species_structs_ch = all_structs_ch.filter { label, _structs -> label != "human" }
        foldseek_input = species_structs_ch
            .combine(human_structs_ch.map { _label, structs -> structs })
            .map { sp_label, sp_structs, hu_structs -> tuple(sp_label, hu_structs, sp_structs) }
        foldseek_results = foldseekSearch(foldseek_input)
    }

    // -----------------------------------------------------------------------
    // Step 7: MMseqs2
    // -----------------------------------------------------------------------
    mmseqs2_results = mmseqs2EasySearch(
        species_ch.combine(query_fasta).map { species, fasta, qf -> tuple(species, fasta, qf) }
    )

    // -----------------------------------------------------------------------
    // Steps 8-11: evaluate, aggregate, report
    // -----------------------------------------------------------------------
    all_results = kmerseek_results.mix(foldseek_results).mix(mmseqs2_results)

    eval_input = all_results
        .combine(disprot_gt_ch, by: 0)
        .combine(disorder_scores)
        .map { species, tool, tsv, gt, disorder -> tuple(species, tool, tsv, gt, disorder) }
    eval_out = evaluateDisprotBenchmark(eval_input)

    agg_out = aggregateDisprotMetrics(eval_out[0].mix(eval_out[1]).collect())

    generateReport(agg_out.metrics, agg_out.figures)

    multiQC(buildMultiqcInputs(agg_out.metrics))
}
