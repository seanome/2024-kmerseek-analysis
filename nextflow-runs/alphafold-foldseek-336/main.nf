#!/usr/bin/env nextflow

/*
 * alphafold_foldseek_336.nf
 *
 * Validate KmerSeek's 336 FoldSeek-missed detections using AlphaFold2
 * structures and FoldSeek structure-based search.
 *
 * The 336 are SCOPe domains detected by KmerSeek at family level that FoldSeek
 * returns no hit for. FoldSeek only finds hits for ~1/3 of SCOPe-40 proteins;
 * these 336 are among the ~2/3 FoldSeek misses entirely.
 *
 * Pipeline:
 *   1. fetchQueryUniprotIds  — map 336 SCOPe domain PDB IDs → UniProt accessions
 *   2. fetchRefUniprotIds    — map SCOPe-40 reference domain PDB IDs → UniProt
 *   3. downloadAF2Structures — download AF-{ACC}-F1-model_v4.cif from EBI (×2)
 *   4. foldseekSearch        — foldseek easy-search query_structs/ ref_structs/
 *   5. compareToKmerseek     — compare FoldSeek hits to KmerSeek family assignments
 *
 * Prerequisites:
 *   - data/077_outside_benchmark_336_domains.txt  (from notebook 077)
 *   - data/foldseek-analysis/scop_lookup.fix.tsv
 *   - data/tea_scope40_rocx_files/foldseek.rocx
 *   - ~/data/scope/results-hp-alphabet-sweep/scope_eval.hp_pbotc_1st_ed.k26.tsv.gz
 *
 * Usage:
 *   nextflow run main.nf
 *   nextflow run main.nf --resume
 *   nextflow run main.nf --skip_ref_download  # use existing ref structures
 */

nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

def home       = System.getProperty('user.home')
def projectRoot = "${projectDir}/../.."

params.query_domains   = "${projectRoot}/data/077_outside_benchmark_336_domains.txt"
params.scope_lookup    = "${projectRoot}/data/foldseek-analysis/scop_lookup.fix.tsv"
params.foldseek_rocx   = "${projectRoot}/data/tea_scope40_rocx_files/foldseek.rocx"
params.kmerseek_eval   = "${home}/data/scope/results-hp-alphabet-sweep/scope_eval.hp_pbotc_1st_ed.k26.tsv.gz"

params.outdir          = "${home}/data/alphafold-foldseek-336/results"
params.af2_cache       = "${home}/data/alphafold_structures"

params.evalue_report   = 10.0

// ---------------------------------------------------------------------------
// PROCESS 1 — Map 336 query PDB IDs → UniProt accessions
// ---------------------------------------------------------------------------

process fetchQueryUniprotIds {
    label 'python'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path query_domains

    output:
    path "query_uniprot_map.tsv"

    script:
    """
    ${projectDir}/bin/fetch_uniprot_ids.py \\
        ${query_domains} \\
        query_uniprot_map.tsv \\
        --max-workers 8
    """
}

// ---------------------------------------------------------------------------
// PROCESS 2 — Map SCOPe-40 reference PDB IDs → UniProt accessions
// ---------------------------------------------------------------------------

process fetchRefUniprotIds {
    label 'python'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path scope_lookup

    output:
    path "ref_uniprot_map.tsv"

    script:
    """
    ${projectDir}/bin/fetch_uniprot_ids.py \\
        ${scope_lookup} \\
        ref_uniprot_map.tsv \\
        --max-workers 8
    """
}

// ---------------------------------------------------------------------------
// PROCESS 3 — Download AlphaFold2 CIF structures
//   Called twice: once for the 336 query set, once for the SCOPe-40 ref set
// ---------------------------------------------------------------------------

process downloadAF2Structures {
    tag "${label}"
    label 'python'
    publishDir "${params.af2_cache}", mode: 'copy', saveAs: { f -> f }

    input:
    tuple val(label), path(uniprot_map)

    output:
    tuple val(label), path("structures_${label}/")

    script:
    """
    mkdir -p structures_${label}
    ${projectDir}/bin/download_af2_structures.py \\
        ${uniprot_map} \\
        --outdir       structures_${label} \\
        --cache        ${params.af2_cache} \\
        --max-workers  5
    """
}

// ---------------------------------------------------------------------------
// PROCESS 4 — FoldSeek easy-search: 336 query structures vs ref structures
// ---------------------------------------------------------------------------

process foldseekSearch {
    container 'quay.io/biocontainers/foldseek:9.427df8a--pl5321h6a68c12_3'
    label 'medium_cpu'
    publishDir "${params.outdir}/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple path(query_structs), path(ref_structs)

    output:
    path "foldseek_336.tsv.gz"

    script:
    """
    mkdir -p foldseek_tmp

    foldseek easy-search \\
        ${query_structs} \\
        ${ref_structs} \\
        foldseek_336.tsv \\
        foldseek_tmp \\
        --format-output "query,target,bits,evalue" \\
        --threads ${task.cpus} \\
        -e ${params.evalue_report}

    gzip -c foldseek_336.tsv > foldseek_336.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESS 5 — Compare FoldSeek hits to KmerSeek assignments
// ---------------------------------------------------------------------------

process compareToKmerseek {
    label 'python'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path foldseek_hits
    path query_uniprot_map
    path ref_uniprot_map
    path scope_lookup
    path kmerseek_eval
    path foldseek_rocx

    output:
    path "comparison.parquet"
    path "summary.txt"

    script:
    """
    ${projectDir}/bin/compare_foldseek_kmerseek.py \\
        --foldseek-hits  ${foldseek_hits} \\
        --query-uniprot  ${query_uniprot_map} \\
        --ref-uniprot    ${ref_uniprot_map} \\
        --scope-lookup   ${scope_lookup} \\
        --kmerseek-eval  ${kmerseek_eval} \\
        --foldseek-rocx  ${foldseek_rocx} \\
        --outdir         .
    """
}

// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {

    query_domains_ch = Channel.fromPath(params.query_domains, checkIfExists: true)
    scope_lookup_ch  = Channel.fromPath(params.scope_lookup,  checkIfExists: true)
    foldseek_rocx_ch = Channel.fromPath(params.foldseek_rocx, checkIfExists: true)
    kmerseek_eval_ch = Channel.fromPath(params.kmerseek_eval, checkIfExists: true)

    // Step 1 & 2: Fetch UniProt IDs
    query_uniprot_ch = fetchQueryUniprotIds(query_domains_ch)
    ref_uniprot_ch   = fetchRefUniprotIds(scope_lookup_ch)

    // Step 3: Download AF2 structures for both sets
    structs_input_ch = Channel.of(
        ["query", query_uniprot_ch],
        ["ref",   ref_uniprot_ch],
    ).flatMap { label, map_ch ->
        map_ch.map { map_file -> tuple(label, map_file) }
    }

    structs_ch = downloadAF2Structures(structs_input_ch)

    // Step 4: FoldSeek search (join query and ref structure directories)
    query_structs_ch = structs_ch.filter { label, _dir -> label == "query" }
                                  .map    { _label, dir -> dir }
    ref_structs_ch   = structs_ch.filter { label, _dir -> label == "ref" }
                                  .map    { _label, dir -> dir }

    foldseek_input_ch = query_structs_ch.combine(ref_structs_ch)
    foldseek_hits_ch  = foldseekSearch(foldseek_input_ch)

    // Step 5: Compare FoldSeek to KmerSeek
    compareToKmerseek(
        foldseek_hits_ch,
        query_uniprot_ch,
        ref_uniprot_ch,
        scope_lookup_ch,
        kmerseek_eval_ch,
        foldseek_rocx_ch,
    )
}
