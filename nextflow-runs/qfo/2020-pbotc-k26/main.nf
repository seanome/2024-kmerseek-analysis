#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.data_dir  = '/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143'
params.ksizes    = [26]
params.scaled    = 1
params.moltype        = 'hp-pbotc-1st-ed'   // CLI flag form (hyphens)
params.moltype_slug   = 'hp_pbotc_1st_ed'   // output path form (underscores, as kmerseek writes it)
params.pvalues   = [0.05, 1e-5]        // raw and intermediate; Bonferroni applied in formatter
params.outdir    = "${launchDir}/results"

// ---------------------------------------------------------------------------
// INDEX each proteome FASTA with kmerseek
// One index per (fasta, ksize) combination.
// ---------------------------------------------------------------------------
process INDEX_PROTEOME {
    container 'kmerseek:0.3.0'
    tag "${fasta.simpleName}_k${ksize}"

    input:
    tuple path(fasta), val(ksize)

    output:
    tuple val(fasta.simpleName),
          val(ksize),
          path(fasta),
          path("${fasta}.${params.moltype_slug}.k${ksize}.scaled${params.scaled}.kmerseek.rocksdb", type: 'dir')

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
    container 'kmerseek:0.3.0'
    tag "${query_id}_vs_${target_id}_k${ksize}"

    // storeDir keeps CSVs in a named location instead of the ephemeral work dir.
    // On rerun, if the CSV already exists in search_results/, this process is skipped.
    // This prevents the work/ directory from accumulating hundreds of GB of search CSVs.
    storeDir "${params.outdir}/search_results"
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
    path "kmerseek_k${ksize}_${params.moltype_slug}_pval${pvalue}_qfo2020.orthoxml"

    script:
    """
    format_orthoxml.py \\
        --pvalue  ${pvalue}          \\
        --ksize   ${ksize}           \\
        --moltype ${params.moltype_slug}  \\
        --scaled  ${params.scaled}   \\
        --results search_results/    \\
        --output  kmerseek_k${ksize}_${params.moltype_slug}_pval${pvalue}_qfo2020.orthoxml
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

    // Collect results per ksize, then fan out across pvalues for OrthoXML formatting
    results_by_ksize = results_ch.groupTuple(by: 0)   // (ksize, [csv_files...])

    pvalues_ch = Channel.fromList(params.pvalues)

    format_inputs = results_by_ksize
        .combine(pvalues_ch)
        .map { ksize, csvs, pvalue -> tuple(ksize, pvalue, csvs) }

    FORMAT_ORTHOXML(format_inputs)
}
