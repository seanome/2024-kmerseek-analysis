#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.data_dir  = '/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143'
// k=24 is DONE: kmerseek_k24_hp_qfo2020.orthoxml already exists in results/.
// Remaining k-sizes to run: 20, 28, 32, 36.
params.ksizes    = [20, 28, 32, 36]
params.scaled    = 1
params.moltype   = 'hp'
params.pvalues   = [0.05, 1e-5]   // raw and intermediate; Bonferroni applied in formatter
params.outdir    = "${launchDir}/results"

// ---------------------------------------------------------------------------
// INDEX each proteome FASTA with kmerseek
// One index per (fasta, ksize) combination.
// ---------------------------------------------------------------------------
process INDEX_PROTEOME {
    container 'kmerseek:0.2.0'
    tag "${fasta.simpleName}_k${ksize}"

    input:
    tuple path(fasta), val(ksize)

    output:
    tuple val(fasta.simpleName),
          val(ksize),
          path(fasta),
          path("${fasta}.hp.k${ksize}.scaled${params.scaled}.kmerseek.rocksdb", type: 'dir')

    script:
    """
    kmerseek index \\
        --encoding ${params.moltype} \\
        --ksize    ${ksize}          \\
        --scaled   ${params.scaled}  \\
        --input    ${fasta}
    """
}

// ---------------------------------------------------------------------------
// SEARCH one proteome against another — both directions per pair.
// Containment is asymmetric, so A→B and B→A can differ.
// ---------------------------------------------------------------------------
process KMERSEEK_SEARCH {
    container 'kmerseek:0.2.0'
    tag "${query_id}_vs_${target_id}_k${ksize}"

    // No storeDir here -- the raw csv.gz is intentionally NOT persisted. It only
    // lives in this task's own work dir, gets consumed by CONVERT_TO_PARQUET below,
    // and only the resulting parquet is stored.
    publishDir "${params.outdir}/search_logs", mode: 'copy', pattern: '*.stderr.log'

    input:
    tuple val(query_id), val(ksize), path(query_fasta),
          val(target_id),            path(target_rocksdb)

    output:
    tuple val(ksize), path("${query_id}_vs_${target_id}.k${ksize}.csv.gz")
    path "${query_id}_vs_${target_id}.k${ksize}.stderr.log"

    script:
    def out_csv  = "${query_id}_vs_${target_id}.k${ksize}.csv.gz"
    def log_file = "${query_id}_vs_${target_id}.k${ksize}.stderr.log"
    """
    kmerseek search \\
        --encoding ${params.moltype} \\
        --ksize    ${ksize}          \\
        --query    ${query_fasta}    \\
        --target   ${target_rocksdb} \\
        2> ${log_file} | gzip > ${out_csv}
    """
}

// No container: needs the host conda env's polars. kmerseek:0.2.0 is a minimal
// binary-only image with no Python, so this has to run outside it. storeDir keeps
// parquets in a named location instead of the ephemeral work dir -- on rerun, if
// the parquet already exists in search_results/, this process is skipped. This
// prevents the work/ directory from accumulating hundreds of GB of search results.
process CONVERT_TO_PARQUET {
    tag "${csv_gz.simpleName}"
    storeDir "${params.outdir}/search_results"

    input:
    tuple val(ksize), path(csv_gz)

    output:
    tuple val(ksize), path("${csv_gz.name.replaceAll(/\.csv\.gz$/, '')}.parquet")

    script:
    def out_stem = csv_gz.name.replaceAll(/\.csv\.gz$/, '')
    """
    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 << 'PYEOF'
import os
import polars as pl

csv_path = "${csv_gz}"
out_path = "${out_stem}.parquet"

if os.path.getsize(csv_path) == 0:
    open(out_path, "wb").close()
else:
    try:
        pl.scan_csv(csv_path, ignore_errors=True).sink_parquet(
            out_path, compression="zstd", compression_level=9
        )
    except Exception as e:
        print(f"WARNING: {csv_path} has no readable rows ({e}); writing empty parquet.")
        open(out_path, "wb").close()
PYEOF
    """
}

// ---------------------------------------------------------------------------
// FORMAT all pairwise results as OrthoXML for one (ksize, pvalue) combination.
// ---------------------------------------------------------------------------
process FORMAT_ORTHOXML {
    container 'kmerseek-analysis:latest'
    tag "k${ksize}_pval${pvalue}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(ksize), val(pvalue), path('search_results/*')

    output:
    path "kmerseek_k${ksize}_${params.moltype}_pval${pvalue}_qfo2020.orthoxml"

    script:
    // kmerseek-analysis:latest's shebang python (host conda path baked into
    // format_orthoxml.py) doesn't exist inside the container -- invoke it via the
    // container's own PATH python3 (which has polars) instead of relying on the
    // shebang.
    """
    python3 "\$(command -v format_orthoxml.py)" \\
        --pvalue  ${pvalue}          \\
        --ksize   ${ksize}           \\
        --moltype ${params.moltype}  \\
        --scaled  ${params.scaled}   \\
        --results search_results/    \\
        --output  kmerseek_k${ksize}_${params.moltype}_pval${pvalue}_qfo2020.orthoxml
    """
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------
workflow {
    // All canonical protein FASTA files (skip isoforms and DNA sequences)
    proteomes_ch = Channel
        .fromPath("${params.data_dir}/{Archaea,Bacteria,Eukaryota}/*.fasta")
        .filter { !it.name.contains('_additional') && !it.name.contains('_DNA') }

    ksizes_ch = Channel.fromList(params.ksizes)

    // Index every (fasta, ksize) combination
    indexed_ch = INDEX_PROTEOME(proteomes_ch.combine(ksizes_ch))

    // All-vs-all pairs, each unordered pair run once (query_id < target_id).
    // Grouped by ksize so each search uses the right index.
    pairs_ch = indexed_ch
        .combine(indexed_ch)
        .filter { qid, qk, qfasta, qdb, tid, tk, tfasta, tdb -> qk == tk && qid < tid }
        .map    { qid, qk, qfasta, qdb, tid, tk, tfasta, tdb ->
                  tuple(qid, qk, qfasta, tid, tdb) }

    (results_ch, _stderr_ch) = KMERSEEK_SEARCH(pairs_ch)

    parquet_ch = CONVERT_TO_PARQUET(results_ch)

    // Collect results per ksize, then fan out across pvalues for OrthoXML formatting
    results_by_ksize = parquet_ch.groupTuple(by: 0)   // (ksize, [parquet_files...])

    pvalues_ch = Channel.fromList(params.pvalues)

    format_inputs = results_by_ksize
        .combine(pvalues_ch)
        .map { ksize, csvs, pvalue -> tuple(ksize, pvalue, csvs) }

    FORMAT_ORTHOXML(format_inputs)
}
