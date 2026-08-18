#!/usr/bin/env nextflow

/*
 * Nextflow pipeline: SCOPe40 low-complexity null
 *
 * Builds the null distribution needed to calibrate a low-complexity k-mer mask
 * for kmerseek's HP alphabets, expressed as a minority *fraction* alpha rather
 * than a fixed minority count.
 *
 * The null is 10 dipeptide-preserving (Altschul-Erikson) shuffles of SCOPe40,
 * searched with the real HP alphabets. Every hit on a shuffled set is a false
 * positive by construction, while each domain keeps its own residue and
 * dipeptide composition — so low-complexity k-mers are still generated at the
 * rate the real database generates them, and the complexity distribution being
 * measured is in the alphabet the mask actually ships in.
 *
 * The unshuffled set is searched identically so the same measurements can be
 * split into true positives (same superfamily) and false positives, which is
 * what turns the null into a cost curve.
 *
 * Note this is a different null from nextflow-runs/hp-alphabet-sweep, which
 * shuffles the *alphabet* (random H/P partitions). There, sequences are still
 * real proteins, so same-family hits remain genuine and the k-mers come from
 * random partitions rather than from hp_thomas_dill et al. Those runs are a
 * useful independent cross-check but cannot define "a hit that shouldn't exist".
 *
 * Usage:
 *   nextflow run main.nf -profile local
 *   nextflow run main.nf -profile local --n_test 300      // fast smoke test
 */

nextflow.enable.dsl = 2

params.fasta       = "${System.getProperty('user.home')}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir      = "${System.getProperty('user.home')}/data/scope/results-lowcomplexity-null"
params.n_shuffles  = 10
params.shuffle_k   = 2      // 2 = dipeptide-preserving; 1 = composition only
params.threshold   = 0.0    // keep every hit; the null's tail is the point
params.n_test      = 0      // >0 truncates the FASTA to this many sequences
params.kmerseek    = "${System.getProperty('user.home')}/code/kmerseek/target/release/kmerseek"
params.python      = "${System.getProperty('user.home')}/anaconda3/envs/2025-kmerseek-analysis/bin/python3"

// All six real HP alphabets. `hp` is omitted as an exact alias of hp-lehninger.
// Their H:P residue splits range from 7/13 (kyte-doolittle) to 11/9
// (lehninger-plus-c), so the calibrated cutoff is not expected to be identical
// across them — that spread is part of what the sweep measures.
params.alphabets = [
    'hp-lehninger',
    'hp-thomas-dill',
    'hp-kyte-doolittle',
    'hp-thomas-dill-no-c',
    'hp-lehninger-plus-c',
    'hp-pbotc-1st-ed',
]

params.ksizes = [22, 24, 26, 28]


process subsetFasta {
    tag "n=${params.n_test}"
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path 'subset.fa'

    script:
    """
    awk '/^>/{n++} n>${params.n_test}{exit} {print}' ${fasta} > subset.fa
    """
}


process parseScopeHeaders {
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path 'scope_domains.tsv'

    script:
    """
    ${params.python} ${projectDir}/bin/parse_scope_headers.py \\
        --fasta ${fasta} --output scope_domains.tsv
    """
}


process shuffleFasta {
    tag "shuf${String.format('%02d', seed)}"
    publishDir "${params.outdir}/shuffled", mode: 'copy', pattern: '*.fa'
    publishDir params.outdir, mode: 'copy', pattern: '*.shuffle_report.tsv'

    input:
    tuple path(fasta), val(seed)

    output:
    tuple val("shuf${String.format('%02d', seed)}"), path("shuf${String.format('%02d', seed)}.fa"), emit: fasta
    path "*.shuffle_report.tsv", emit: report

    script:
    def label = "shuf${String.format('%02d', seed)}"
    """
    ${params.python} ${projectDir}/bin/shuffle_fasta.py \\
        --input ${fasta} \\
        --output ${label}.fa \\
        --seed ${seed} \\
        --shuffle-k ${params.shuffle_k} \\
        --report ${label}.shuffle_report.tsv
    """
}


process indexDatabase {
    tag "${seqset}_${alphabet}_k${ksize}"

    input:
    tuple val(seqset), path(fasta), val(alphabet), val(ksize)

    output:
    tuple val(seqset), val(alphabet), val(ksize), path("*.rocksdb", type: 'dir')

    script:
    def label = alphabet.replace('-', '_')
    """
    ${params.kmerseek} index \\
        --encoding ${alphabet} \\
        --ksize ${ksize} --scaled 1 \\
        --input ${fasta} \\
        --output ${seqset}.${label}.k${ksize}.rocksdb \\
        2> ${seqset}.${label}.k${ksize}.index.log
    """
}


process searchAllVsAll {
    tag "${seqset}_${alphabet}_k${ksize}"

    input:
    tuple val(seqset), val(alphabet), val(ksize), path(index)

    output:
    tuple val(seqset), val(alphabet), val(ksize), path("*.results.csv")

    script:
    def label = alphabet.replace('-', '_')
    """
    ${params.kmerseek} search \\
        --encoding ${alphabet} \\
        --ksize ${ksize} \\
        --threshold ${params.threshold} \\
        --query-is-index --query ${index} --target ${index} \\
        > ${seqset}.${label}.k${ksize}.results.csv \\
        2> ${seqset}.${label}.k${ksize}.search.log
    """
}


process hitComplexity {
    tag "${seqset}_${alphabet}_k${ksize}"
    publishDir "${params.outdir}/per_task", mode: 'copy'

    input:
    tuple val(seqset), val(alphabet), val(ksize), path(results), path(scop_domains)

    output:
    path "*.hits.parquet",  emit: hits
    path "*.kmers.parquet", emit: kmers
    path "*.stats.tsv",     emit: stats

    script:
    def label = alphabet.replace('-', '_')
    """
    ${params.python} ${projectDir}/bin/hit_complexity.py \\
        --results ${results} \\
        --seqset ${seqset} \\
        --alphabet ${alphabet} \\
        --ksize ${ksize} \\
        --scop-domains ${scop_domains} \\
        --out-prefix ${seqset}.${label}.k${ksize}
    """
}


process kmerFrequency {
    tag "${alphabet}_k${ksize}"
    publishDir "${params.outdir}/per_task", mode: 'copy'

    input:
    tuple path(fasta), val(alphabet), val(ksize)

    output:
    path "*.freq_by_minority.parquet", emit: freq
    path "*.top_kmers.parquet",        emit: top
    path "*.freq_stats.tsv",           emit: stats

    script:
    def label = alphabet.replace('-', '_')
    """
    ${params.python} ${projectDir}/bin/kmer_frequency.py \\
        --fasta ${fasta} \\
        --alphabet ${alphabet} \\
        --ksize ${ksize} \\
        --out-prefix real.${label}.k${ksize}
    """
}


process aggregate {
    publishDir params.outdir, mode: 'copy'

    input:
    path per_task_files

    output:
    path 'hits.parquet'
    path 'kmers_by_minority.parquet'
    path 'freq_by_minority.parquet'
    path 'top_kmers.parquet'
    path 'run_stats.tsv'
    path 'shuffle_reports.tsv', optional: true

    script:
    """
    ${params.python} ${projectDir}/bin/aggregate_null.py --outdir .
    """
}


workflow {
    raw_fasta = channel.fromPath(params.fasta, checkIfExists: true)

    // A truncated FASTA makes the whole DAG runnable in a couple of minutes,
    // which is how the pipeline gets tested without a multi-hour full run.
    fasta_ch = params.n_test > 0 ? subsetFasta(raw_fasta) : raw_fasta

    scop_domains = parseScopeHeaders(fasta_ch)

    shuffles = shuffleFasta(fasta_ch.combine(channel.of(1..params.n_shuffles)))

    // The unshuffled set runs through the identical path so true-positive cost
    // is measured with the same code that measures the null's false positives.
    seqsets = channel.of('real').combine(fasta_ch).mix(shuffles.fasta)

    search_inputs = seqsets
        .combine(channel.fromList(params.alphabets))
        .combine(channel.fromList(params.ksizes))

    indexed  = indexDatabase(search_inputs)
    searched = searchAllVsAll(indexed)

    complexity = hitComplexity(searched.combine(scop_domains))

    // Reference k-mer frequency is a property of the real database only.
    frequency = kmerFrequency(
        fasta_ch
            .combine(channel.fromList(params.alphabets))
            .combine(channel.fromList(params.ksizes))
    )

    aggregate(
        complexity.hits
            .mix(complexity.kmers)
            .mix(complexity.stats)
            .mix(frequency.freq)
            .mix(frequency.top)
            .mix(frequency.stats)
            .mix(shuffles.report)
            .collect()
    )
}
