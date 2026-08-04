#!/usr/bin/env nextflow

/*
 * kmerseek_qfo_pfam_benchmark.nf
 *
 * Index each QfO target species with kmerseek (HP encoding, k=10,12,14,16,18 by default;
 * k=20,24,28,32,36,40 already completed and converted separately),
 * then search human proteins against every species index.
 *
 * Output Parquet files land in params.outdir, consumed by
 * notebooks/123_pfam_subdomain_divergence_benchmark.ipynb.
 *
 * Rows are pre-filtered to poisson_pvalue < params.pvalue_filter (default 0.05,
 * uncorrected). Bonferroni correction (n_human * n_species pairs) is applied
 * in the notebook.
 *
 * Resource usage is logged by Nextflow's trace file (see nextflow.config).
 *
 * NOTE: the kmerseek container must have zstd installed. If not, replace
 *       `zstd -19` with `gzip -9` in searchHumanVsSpecies.
 *
 * Disk: searchHumanVsSpecies's raw *.csv.zst is intentionally NOT published --
 * convertToParquet (no container, runs on the host so it can use the host conda
 * env's polars) converts it to *.parquet and only the parquet is stored. This
 * keeps params.outdir 100% parquet, matching every other pipeline in this repo.
 *
 * Usage (from this directory):
 *   nextflow run kmerseek_qfo_pfam_benchmark.nf
 *   nextflow run kmerseek_qfo_pfam_benchmark.nf --resume
 *   nextflow run kmerseek_qfo_pfam_benchmark.nf --ksizes "10,12"   # subset
 */

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.kmerseek = "kmerseek"
// binary is on PATH inside the kmerseek:0.2.1 container
params.qfo_dir = "${System.getProperty('user.home')}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.outdir = "${System.getProperty('user.home')}/data/qfo-pfam-benchmark/kmerseek-results"
params.encoding = "hp"
params.scaled = 1
params.ksizes = "16,18,20,22,24,26,28,30"
// comma-separated list of k-mer sizes to index/search
params.pvalue_filter = 0.05
// raw (uncorrected) poisson_pvalue threshold

// ---------------------------------------------------------------------------
// Processes
// ---------------------------------------------------------------------------

process indexSpecies {
    tag "${species} k=${ksize}"
    container 'kmerseek:0.2.1'
    publishDir "${params.outdir}/indices", mode: 'copy', pattern: '*.rocksdb', type: 'dir'
    publishDir "${params.outdir}/logs", mode: 'copy', pattern: '*.index.log'

    input:
    tuple val(species), path(fasta), val(ksize)

    output:
    tuple val(species), val(ksize), path("${fasta}.${params.encoding}.k${ksize}.scaled${params.scaled}.kmerseek.rocksdb", type: 'dir')
    path "${fasta}.${params.encoding}.k${ksize}.scaled${params.scaled}.kmerseek.index.log"

    script:
    def log_file = "${fasta}.${params.encoding}.k${ksize}.scaled${params.scaled}.kmerseek.index.log"
    """
    echo "=== Index: ${species}  encoding=${params.encoding}  k=${ksize}  scaled=${params.scaled} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}

    ${params.kmerseek} index \\
        --encoding ${params.encoding} \\
        --ksize    ${ksize}           \\
        --scaled   ${params.scaled}   \\
        --input    ${fasta}           \\
        2>&1 | tee -a ${log_file}

    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    """
}

process searchHumanVsSpecies {
    tag "human_vs_${species} k=${ksize}"
    container 'kmerseek:0.2.1'
    publishDir "${params.outdir}/logs", mode: 'copy', pattern: '*.search.log'

    input:
    tuple val(species), val(ksize), path(species_index), path(human_fasta)

    output:
    tuple val(species), val(ksize), path("human_vs_${species}.${params.encoding}.k${ksize}.results.csv.zst")
    path "human_vs_${species}.${params.encoding}.k${ksize}.search.log"

    script:
    def out_csv = "human_vs_${species}.${params.encoding}.k${ksize}.results.csv.zst"
    def log_file = "human_vs_${species}.${params.encoding}.k${ksize}.search.log"
    // poisson_pvalue is the 23rd comma-separated field (1-indexed).
    // awk keeps the header (NR==1) plus rows where the raw p-value < pvalue_filter.
    """
    echo "=== Search: human vs ${species}  encoding=${params.encoding}  k=${ksize} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "P-value filter (raw, uncorrected): ${params.pvalue_filter}" | tee -a ${log_file}

    ${params.kmerseek} search \\
        --encoding ${params.encoding} \\
        --ksize    ${ksize}           \\
        --query    ${human_fasta}     \\
        --target   ${species_index}   \\
        2>> ${log_file} \\
        | awk -F',' 'NR==1 || \$23 < ${params.pvalue_filter}' \\
        | zstd -19 -o ${out_csv}

    echo "End:  \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Rows (incl. header): \$(zstdcat ${out_csv} | wc -l)" | tee -a ${log_file}
    """
}

// No container: needs the host conda env's polars. kmerseek:0.2.1 is a minimal
// binary-only image with no Python, so this has to run outside it.
process convertToParquet {
    tag "human_vs_${species} k=${ksize}"
    publishDir params.outdir, mode: 'copy', pattern: '*.parquet'

    input:
    tuple val(species), val(ksize), path(csv_zst)

    output:
    tuple val(species), val(ksize), path("human_vs_${species}.${params.encoding}.k${ksize}.results.parquet")

    script:
    def out_parquet = "human_vs_${species}.${params.encoding}.k${ksize}.results.parquet"
    """
    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 << 'PYEOF'
import os
import polars as pl

DROP_COLUMNS = ["query_md5", "target_md5"]
csv_path = "${csv_zst}"
out_path = "${out_parquet}"

if os.path.getsize(csv_path) == 0:
    open(out_path, "wb").close()
else:
    try:
        lf = pl.scan_csv(csv_path, ignore_errors=True)
        cols = [c for c in lf.collect_schema().names() if c not in DROP_COLUMNS]
        lf.select(cols).sink_parquet(out_path, compression="zstd", compression_level=9)
    except Exception as e:
        print(f"WARNING: {csv_path} has no readable rows ({e}); writing empty parquet.")
        open(out_path, "wb").close()
PYEOF
    """
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

workflow {
    // Species table: label -> [taxon_id, proteome_id, subdir]
    // Human is query-only; all others are indexed as targets.
    def species_map = [
        mouse: ["10090", "UP000000589", "Eukaryota"],
        chicken: ["9031", "UP000000539", "Eukaryota"],
        zebrafish: ["7955", "UP000000437", "Eukaryota"],
        ciona: ["7719", "UP000008144", "Eukaryota"],
        fly: ["7227", "UP000000803", "Eukaryota"],
        worm: ["6239", "UP000001940", "Eukaryota"],
        yeast: ["559292", "UP000002311", "Eukaryota"],
        arabidopsis: ["3702", "UP000006548", "Eukaryota"],
        ecoli: ["83333", "UP000000625", "Bacteria"],
    ]

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")
    def ksize_list = params.ksizes.tokenize(",").collect { k -> k.trim().toInteger() }

    // Build channel: (species_label, species_fasta, ksize)
    def species_ksize_tuples = species_map.collectMany { label, info ->
        def (taxon, proteome, subdir) = info
        ksize_list.collect { ksize ->
            tuple(label, file("${params.qfo_dir}/${subdir}/${proteome}_${taxon}.fasta"), ksize)
        }
    }
    species_ksize_ch = channel.from(species_ksize_tuples)

    // Index every (species, ksize) combination
    indexed = indexSpecies(species_ksize_ch)
    index_ch = indexed[0]
    // (species, ksize, index_path)

    // Attach human FASTA, then search
    search_input = index_ch.map { species, ksize, idx ->
        tuple(species, ksize, idx, human_fasta)
    }
    search_out = searchHumanVsSpecies(search_input)

    parquet_out = convertToParquet(search_out[0])

    parquet_out.subscribe { species, ksize, parquet ->
        println("Completed: human vs ${species} k=${ksize}  →  ${parquet}")
    }
}
