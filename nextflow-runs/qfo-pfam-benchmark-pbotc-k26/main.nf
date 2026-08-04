#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.qfo_dir        = "${System.getProperty('user.home')}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.outdir         = "${System.getProperty('user.home')}/data/qfo-pfam-benchmark/kmerseek-results-pbotc-k26"
params.encoding       = 'hp-pbotc-1st-ed'
params.encoding_slug  = 'hp_pbotc_1st_ed'   // as written in output paths by kmerseek
params.ksize          = 26
params.scaled         = 5
params.pvalue_filter  = 0.05   // raw uncorrected threshold; Bonferroni applied in notebook

// ---------------------------------------------------------------------------
process INDEX_SPECIES {
    container 'kmerseek:0.3.1'
    tag "${species}_k${params.ksize}"

    input:
    tuple val(species), path(fasta)

    output:
    tuple val(species),
          path("${fasta}.${params.encoding_slug}.k${params.ksize}.scaled${params.scaled}.kmerseek.rocksdb", type: 'dir')

    script:
    """
    kmerseek index \\
        --encoding ${params.encoding} \\
        --ksize    ${params.ksize}    \\
        --scaled   ${params.scaled}   \\
        --input    ${fasta}
    """
}

// ---------------------------------------------------------------------------
process SEARCH_HUMAN_VS_SPECIES {
    container 'kmerseek:0.3.1'
    tag "human_vs_${species}"

    // No storeDir here -- the raw csv.zst is intentionally NOT persisted. It only
    // lives in this task's own work dir, gets consumed by CONVERT_TO_PARQUET below,
    // and only the resulting parquet is stored (see that process's comment for why).
    publishDir "${params.outdir}/logs", mode: 'copy', pattern: '*.search.log'

    input:
    tuple val(species), path(species_index), path(human_fasta)

    output:
    tuple val(species), path("human_vs_${species}.${params.encoding_slug}.k${params.ksize}.results.csv.zst")
    path "human_vs_${species}.${params.encoding_slug}.k${params.ksize}.search.log"

    script:
    def out_csv  = "human_vs_${species}.${params.encoding_slug}.k${params.ksize}.results.csv.zst"
    def log_file = "human_vs_${species}.${params.encoding_slug}.k${params.ksize}.search.log"
    // poisson_pvalue is field 23 (1-indexed); awk keeps header + rows below threshold
    """
    kmerseek search \\
        --encoding ${params.encoding} \\
        --ksize    ${params.ksize}    \\
        --query    ${human_fasta}     \\
        --target   ${species_index}   \\
        2>> ${log_file} \\
        | awk -F',' 'NR==1 || \$23 < ${params.pvalue_filter}' \\
        | zstd -19 -o ${out_csv}

    echo "Rows (incl. header): \$(zstdcat ${out_csv} | wc -l)" >> ${log_file}
    """
}

// No container: needs the host conda env's polars. kmerseek:0.3.1 is a minimal
// binary-only image with no Python, so this has to run outside it.
process CONVERT_TO_PARQUET {
    tag "human_vs_${species}"
    storeDir params.outdir

    input:
    tuple val(species), path(csv_zst)

    output:
    tuple val(species), path("human_vs_${species}.${params.encoding_slug}.k${params.ksize}.results.parquet")

    script:
    def out_parquet = "human_vs_${species}.${params.encoding_slug}.k${params.ksize}.results.parquet"
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
workflow {
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

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")

    species_ch = Channel.from(
        species_map.collect { label, info ->
            def (taxon, proteome, subdir) = info
            tuple(label, file("${params.qfo_dir}/${subdir}/${proteome}_${taxon}.fasta"))
        }
    )

    indexed_ch = INDEX_SPECIES(species_ch)

    search_inputs = indexed_ch.map { species, idx -> tuple(species, idx, human_fasta) }

    (search_out, _log_out) = SEARCH_HUMAN_VS_SPECIES(search_inputs)

    CONVERT_TO_PARQUET(search_out)
}
