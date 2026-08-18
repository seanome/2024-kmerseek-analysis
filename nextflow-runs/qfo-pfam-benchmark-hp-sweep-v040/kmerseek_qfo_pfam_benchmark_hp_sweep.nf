#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * kmerseek_qfo_pfam_benchmark_hp_sweep.nf
 *
 * HP-alphabet x ksize sweep: human vs 9 QfO target species, kmerseek 0.4.0.
 *   encodings: hp-lehninger, hp-thomas-dill, hp-kyte-doolittle,
 *              hp-thomas-dill-no-c, hp-lehninger-plus-c, hp-pbotc-1st-ed
 *   ksizes:    18, 20, 22, 24, 26, 28, 30  (params.ksizes)
 *   -> 9 x 6 x 7 = 378 (species, encoding, ksize) combinations.
 *
 * Sibling of ../qfo-pfam-benchmark (single hp encoding, k=10-18) and
 * ../qfo-pfam-benchmark-pbotc-k26 (single encoding/ksize) -- kept fully
 * separate (own outdir, own workDir) so it never invalidates their resume
 * caches.
 *
 * Disk-saving design (disk is at a premium -- the earlier
 * qfo/2020-thomas-dill-k26 all-vs-all run actually died mid-run with
 * "No space left on device" after its RocksDB indices piled up):
 *
 *   1. INDEX_AND_SEARCH fuses indexing + search into ONE process. The
 *      RocksDB index is built in the task's own work dir, used once to
 *      search, then `rm -rf`'d before the task exits -- it is never
 *      declared as a process output, so Nextflow never copies or stores it
 *      anywhere. Steady-state disk use is bounded by (maxForks x one index
 *      size), not by the full sweep (378 indices, ~190GB+ if persisted).
 *      Tradeoff: `-resume` after a crash re-indexes any incomplete combos
 *      rather than reusing a saved index (accepted -- see conversation).
 *   2. Results are written to zstd-compressed CSV, then converted to
 *      parquet (dropping unused md5 columns) and the raw CSV deleted -- in
 *      the SAME task, same pattern as
 *      ../human-mouse-gencode-orthologs-hp-v040/kmerseek_human_mouse_orthologs_hp_v040.nf.
 *      storeDir persists whatever is declared as output forever, so
 *      keeping both formats around would double the final results
 *      footprint across 378 result files.
 *   3. Uses kmerseek 0.4.0's native --min-shared-kmers / --max-pvalue
 *      filters (applied during search) instead of the old container +
 *      awk-prefilter step used by the sibling pipelines above -- smaller
 *      intermediate, one less pass.
 *
 * Usage (from this directory):
 *   /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run kmerseek_qfo_pfam_benchmark_hp_sweep.nf
 *   /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run kmerseek_qfo_pfam_benchmark_hp_sweep.nf -resume
 *
 * NOTE: macOS without a container gives Nextflow no reliable peak_rss/%mem
 * in the trace file -- if a task looks like it died to OOM (exit 137),
 * check Activity Monitor live rather than trusting the trace column.
 */

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.qfo_dir  = "${System.getProperty('user.home')}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.outdir   = "${System.getProperty('user.home')}/data/qfo-pfam-benchmark/kmerseek-results-hp-sweep-v040"
params.ksizes   = "18,20,22,24,26,28,30"
params.scaled   = 1

// kmerseek 0.4.0 native search filters (see olgabot/bump-version-0.4.0 branch).
// Defaults match the kmerseek CLI's own defaults.
params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_pvalue       = 0.05

// Absolute path to the locally-built kmerseek 0.4.0 release binary -- no Docker
// container for this pipeline (matches human-mouse-gencode-orthologs-hp-v040).
params.kmerseek_bin = "/Users/olga/code/kmerseek/target/release/kmerseek"

// ---------------------------------------------------------------------------
process INDEX_AND_SEARCH {
    tag "${species}_${encoding}_k${ksize}"
    storeDir params.outdir
    publishDir "${params.outdir}/logs", mode: 'copy', pattern: '*.log'

    input:
    tuple val(species), path(species_fasta), val(encoding), val(ksize), path(human_fasta)

    output:
    tuple val(species), val(encoding), val(ksize),
        path("human_vs_${species}.${encoding.replace('-', '_')}.k${ksize}.results.parquet")
    path "human_vs_${species}.${encoding.replace('-', '_')}.k${ksize}.log"

    script:
    def enc_slug    = encoding.replace('-', '_')
    def index_dir   = "${species_fasta}.${enc_slug}.k${ksize}.scaled${params.scaled}.kmerseek.rocksdb"
    def out_zst     = "human_vs_${species}.${enc_slug}.k${ksize}.results.csv.zst"
    def out_parquet = "human_vs_${species}.${enc_slug}.k${ksize}.results.parquet"
    def log_file    = "human_vs_${species}.${enc_slug}.k${ksize}.log"
    """
    echo "=== Index: ${species} ${encoding} k=${ksize} scaled=${params.scaled} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}

    ${params.kmerseek_bin} index \\
        --encoding ${encoding} \\
        --ksize    ${ksize}    \\
        --input    ${species_fasta}   \\
        2>&1 | tee -a ${log_file}

    echo "" | tee -a ${log_file}
    echo "=== Search: human vs ${species} ${encoding} k=${ksize} ===" | tee -a ${log_file}

    ${params.kmerseek_bin} search \\
        --encoding ${encoding} \\
        --ksize    ${ksize}    \\
        --query    ${human_fasta}     \\
        --target   ${index_dir}       \\
        --threshold ${params.threshold} \\
        --min-shared-kmers ${params.min_shared_kmers} \\
        --max-pvalue ${params.max_pvalue} \\
        2>> ${log_file} \\
        | zstd -T2 -o ${out_zst} \\
        || true

    # Ensure output exists even if search produced nothing
    touch ${out_zst}

    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Compressed size (pre-parquet): \$(du -sh ${out_zst} | cut -f1)" | tee -a ${log_file}

    # Convert straight to parquet (dropping the two unused md5 columns) and delete
    # the raw CSV in the SAME task -- storeDir persists whatever this task declares
    # as output forever, so keeping both formats around for 378 result files is
    # exactly the kind of disk blow-up this pipeline is designed to avoid.
    if [ -s ${out_zst} ]; then
        /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 << 'PYEOF'
import polars as pl
DROP_COLUMNS = ["query_md5", "target_md5"]
lf = pl.scan_csv("${out_zst}", ignore_errors=True)
cols = [c for c in lf.collect_schema().names() if c not in DROP_COLUMNS]
lf.select(cols).sink_parquet("${out_parquet}", compression="zstd", compression_level=9)
PYEOF
    else
        # 0-byte input (search produced nothing) -- keep the empty-file signal,
        # downstream notebooks should check via os.path.getsize == 0.
        touch ${out_parquet}
    fi
    rm -f ${out_zst}

    # Ephemeral index: never declared as a process output above, so deleting it
    # here (rather than waiting for a future `nextflow clean`) frees disk the
    # moment this task finishes instead of leaving it to pile up in work/.
    rm -rf ${index_dir}

    echo "Compressed size (parquet): \$(du -sh ${out_parquet} | cut -f1)" | tee -a ${log_file}
    """
}

// ---------------------------------------------------------------------------
workflow {
    // Species table: label -> [taxon_id, proteome_id, subdir]. Human is
    // query-only; all others are indexed as targets.
    def species_map = [
        mouse:       ["10090",  "UP000000589", "Eukaryota"],
        chicken:     ["9031",   "UP000000539", "Eukaryota"],
        zebrafish:   ["7955",   "UP000000437", "Eukaryota"],
        ciona:       ["7719",   "UP000008144", "Eukaryota"],
        fly:         ["7227",   "UP000000803", "Eukaryota"],
        worm:        ["6239",   "UP000001940", "Eukaryota"],
        yeast:       ["559292", "UP000002311", "Eukaryota"],
        arabidopsis: ["3702",   "UP000006548", "Eukaryota"],
        ecoli:       ["83333",  "UP000000625", "Bacteria"],
    ]

    // Same 6 HP alphabet variants as ../human-mouse-gencode-orthologs-hp-v040
    def encodings = [
        'hp-lehninger',
        'hp-thomas-dill',
        'hp-kyte-doolittle',
        'hp-thomas-dill-no-c',
        'hp-lehninger-plus-c',
        'hp-pbotc-1st-ed',
    ]

    def ksize_list  = params.ksizes.tokenize(",").collect { it.trim().toInteger() }
    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")

    // Build (species, species_fasta, encoding, ksize, human_fasta) tuples
    def combos = species_map.collectMany { label, info ->
        def (taxon, proteome, subdir) = info
        def species_fasta = file("${params.qfo_dir}/${subdir}/${proteome}_${taxon}.fasta")
        encodings.collectMany { enc ->
            ksize_list.collect { ksize ->
                tuple(label, species_fasta, enc, ksize, human_fasta)
            }
        }
    }

    combos_ch = Channel.from(combos)

    results = INDEX_AND_SEARCH(combos_ch)

    results[0].subscribe { species, encoding, ksize, parquet ->
        println "Completed: human vs ${species} ${encoding} k=${ksize} -> ${parquet}"
    }
}
