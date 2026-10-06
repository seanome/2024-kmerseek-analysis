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
 *
 * This file holds the build (-entry build, the default). The searches are added after the
 * build has been reviewed.
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
// feature's first residue and end on its last. 19 is hp_pbotc_1st_ed2's k (see README).
params.k_max          = 19
params.min_cluster_aa = 30
params.cluster_min_seq_id = 0.5
params.cluster_coverage   = 0.8
params.decoy_window = 10
params.decoy_seed   = 20261006

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
    tag "${name}"
    publishDir "${params.outdir}/${name}", mode: 'copy'

    input:
    tuple val(name), path(fasta)

    output:
    tuple val(name), path("${name}.target_decoy.fasta"), emit: combined
    path "${name}.decoy.fasta.md5"
    path "${name}.decoys.log"

    script:
    """
    ${py('make_window_decoys.py')} --fasta ${fasta} --decoy-out ${name}.decoy.fasta \\
        --combined-out ${name}.target_decoy.fasta \\
        --window ${params.decoy_window} --seed ${params.decoy_seed} 2> ${name}.decoys.log
    cat ${name}.decoys.log >&2
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

workflow build {
    if (!params.outdir) error "--outdir is required"
    dat = file(params.swissprot_dat, checkIfExists: true)

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
    makeDecoys(indexes)
    pair = makeDecoys.out.combined.toSortedList { a, b -> b[0] <=> a[0] }
        .map { l -> l.flatten() }
    indexSummary(buildQueryTruth.out.queries, pair)
}

workflow {
    build()
}
