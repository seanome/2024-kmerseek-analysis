#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.qfo_dir        = "${System.getProperty('user.home')}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.outdir         = "${System.getProperty('user.home')}/data/qfo-pfam-benchmark/kmerseek-results-thomas-dill-k26"
params.encoding       = 'hp_thomas_dill'
params.encoding_slug  = 'hp_thomas_dill'   // as written in output paths by kmerseek
params.ksize          = 26
params.scaled         = 1
params.pvalues        = [0.05, 1e-5]       // raw uncorrected thresholds; Bonferroni applied in notebook

// ---------------------------------------------------------------------------
process INDEX_PROTEOME {
    container 'kmerseek:0.3.1'
    tag "${fasta.simpleName}_k${params.ksize}"

    input:
    path(fasta)

    output:
    tuple val(fasta.simpleName),
          path(fasta),
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
process KMERSEEK_SEARCH {
    container 'kmerseek:0.3.1'
    tag "${query_id}_vs_${target_id}"

    storeDir "${params.outdir}/search_results"
    publishDir "${params.outdir}/search_logs", mode: 'copy', pattern: '*.stderr.log'

    input:
    tuple val(query_id), path(query_fasta),
          val(target_id), path(target_rocksdb)

    output:
    path("${query_id}_vs_${target_id}.${params.encoding_slug}.k${params.ksize}.csv.gz")
    path "${query_id}_vs_${target_id}.${params.encoding_slug}.k${params.ksize}.stderr.log"

    script:
    def out_csv  = "${query_id}_vs_${target_id}.${params.encoding_slug}.k${params.ksize}.csv.gz"
    def log_file = "${query_id}_vs_${target_id}.${params.encoding_slug}.k${params.ksize}.stderr.log"
    """
    kmerseek search \\
        --encoding ${params.encoding} \\
        --ksize    ${params.ksize}    \\
        --query    ${query_fasta}     \\
        --target   ${target_rocksdb}  \\
        2> ${log_file} | gzip > ${out_csv}
    """
}

// ---------------------------------------------------------------------------
process FORMAT_ORTHOXML {
    container 'kmerseek-analysis:latest'
    tag "pval${pvalue}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(pvalue), path('search_results/*')

    output:
    path "kmerseek_${params.encoding_slug}_k${params.ksize}_pval${pvalue}_qfo2020.orthoxml"

    script:
    """
    format_orthoxml.py \\
        --pvalue  ${pvalue}               \\
        --ksize   ${params.ksize}         \\
        --moltype ${params.encoding_slug} \\
        --scaled  ${params.scaled}        \\
        --results search_results/         \\
        --output  kmerseek_${params.encoding_slug}_k${params.ksize}_pval${pvalue}_qfo2020.orthoxml
    """
}

// ---------------------------------------------------------------------------
workflow {
    // All canonical protein FASTA files (skip isoforms and DNA sequences)
    proteomes_ch = Channel
        .fromPath("${params.qfo_dir}/{Archaea,Bacteria,Eukaryota}/*.fasta")
        .filter { !it.name.contains('_additional') && !it.name.contains('_DNA') }

    indexed_ch = INDEX_PROTEOME(proteomes_ch)

    // All-vs-all pairs; each unordered pair run once (query_id < target_id)
    pairs_ch = indexed_ch
        .combine(indexed_ch)
        .filter { qid, qfasta, qdb, tid, tfasta, tdb -> qid < tid }
        .map    { qid, qfasta, qdb, tid, tfasta, tdb ->
                  tuple(qid, qfasta, tid, tdb) }

    (results_ch, _stderr_ch) = KMERSEEK_SEARCH(pairs_ch)

    pvalues_ch = Channel.fromList(params.pvalues)

    format_inputs = results_ch
        .collect()
        .combine(pvalues_ch)
        .map { csvs, pvalue -> tuple(pvalue, csvs) }

    FORMAT_ORTHOXML(format_inputs)
}
