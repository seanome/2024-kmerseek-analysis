#!/usr/bin/env nextflow

/*
 * QfO benchmark: full HP-alphabet x ksize sweep.
 *
 * All 6 named HP alphabets x 7 ksizes (18,20,22,24,26,28,30) x 9 reference
 * species vs human = 378 (encoding, ksize, species) combos. Batched one job
 * per species per stage (index / search), looping over all 42 encoding x
 * ksize combos inside a single task — 18 AWS Batch jobs total instead of
 * 756, since per-job container-pull/scheduling overhead dominates cost at
 * this task size (QfO proteomes are all <20 MB; see conversation notes).
 *
 * Uses kmerseek 0.4.0 (olgabot/bump-version-0.4.0 branch, which merges in
 * both olgabot/min-shared-kmers-filter and olgabot/remove-scaled-option):
 *   - `search` natively drops matches below --min-shared-kmers / above
 *     --max-pvalue, so no bash awk prefilter step is needed (unlike the
 *     older qfo-pfam-benchmark-pbotc-k26 pipeline this supersedes for sweep
 *     purposes).
 *   - `--scaled` no longer exists as a CLI flag on this branch — full k-mer
 *     capture (what scaled=1 meant) is now the only mode, so there's
 *     nothing to pass.
 *
 * Usage:
 *   nextflow run main.nf -profile local,test -resume   # 1 species, quick check
 *   nextflow run main.nf -profile local -resume         # full sweep, local binary
 *   nextflow run main.nf -profile aws -resume            # full sweep, AWS Batch (Fusion+Wave)
 */

params.qfo_dir  = "${System.getProperty('user.home')}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.outdir   = "${System.getProperty('user.home')}/data/qfo-hp-alphabet-sweep"

// kmerseek 0.4.0 native search filters — defaults match the CLI's own defaults.
params.min_shared_kmers = 2
params.max_pvalue       = 0.05

// All 6 named biological HP alphabets (excludes hp-shuffled-control, which is
// a null model, not a biological alphabet under test here).
params.encodings = [
    'hp-lehninger',
    'hp-thomas-dill',
    'hp-kyte-doolittle',
    'hp-thomas-dill-no-c',
    'hp-lehninger-plus-c',
    'hp-pbotc-1st-ed',
]

params.ksizes = [18, 20, 22, 24, 26, 28, 30]

// Optional species subset (used by -profile test); null = all 9.
params.species = null

// Absolute path to the locally-built kmerseek 0.4.0 release binary
// (olgabot/bump-version-0.4.0 branch — Cargo.toml version already bumped,
// includes the min-shared-kmers-filter merge). Overridden to 'kmerseek' (on
// $PATH inside the container) under -profile aws.
params.kmerseek_bin = "/Users/olga/code/kmerseek/target/release/kmerseek"

// Container image, only consumed under -profile aws (see nextflow.config).
params.kmerseek_container = 'kmerseek:0.4.0'

// ---------------------------------------------------------------------------
process INDEX_SPECIES {
    tag "${species}"
    storeDir "${params.outdir}/indices/${species}"

    input:
    tuple val(species), path(fasta)

    output:
    tuple val(species), path("*.kmerseek.rocksdb", type: 'dir'), path("${species}.index.log")

    script:
    def log_file = "${species}.index.log"
    """
    echo "=== Indexing ${species}: \$(date) ===" > ${log_file}
    for encoding in ${params.encodings.join(' ')}; do
        enc_fname=\$(echo \${encoding} | tr '-' '_')
        for ksize in ${params.ksizes.join(' ')}; do
            db="${species}.\${enc_fname}.k\${ksize}.kmerseek.rocksdb"
            echo "--- \${encoding} k=\${ksize} -> \${db} ---" >> ${log_file}
            ${params.kmerseek_bin} index \\
                --input    ${fasta} \\
                --output   \${db} \\
                --encoding \${encoding} \\
                --ksize    \${ksize} \\
                2>> ${log_file}
        done
    done
    echo "=== Done: \$(date) ===" >> ${log_file}
    """
}

// ---------------------------------------------------------------------------
process SEARCH_HUMAN_VS_SPECIES {
    tag "${species}"
    storeDir "${params.outdir}/results/${species}"

    input:
    tuple val(species), path(indices), path(human_fasta)

    output:
    tuple val(species), path("human_vs_${species}.*.results.csv.zst")
    path "${species}.search.log"

    script:
    def log_file = "${species}.search.log"
    """
    echo "=== Searching human vs ${species}: \$(date) ===" > ${log_file}
    for encoding in ${params.encodings.join(' ')}; do
        enc_fname=\$(echo \${encoding} | tr '-' '_')
        for ksize in ${params.ksizes.join(' ')}; do
            db="${species}.\${enc_fname}.k\${ksize}.kmerseek.rocksdb"
            out="human_vs_${species}.\${enc_fname}.k\${ksize}.results.csv.zst"
            echo "--- \${encoding} k=\${ksize} ---" >> ${log_file}
            ${params.kmerseek_bin} search \\
                --query           ${human_fasta} \\
                --target          \${db} \\
                --encoding        \${encoding} \\
                --ksize           \${ksize} \\
                --min-shared-kmers ${params.min_shared_kmers} \\
                --max-pvalue      ${params.max_pvalue} \\
                2>> ${log_file} \\
                | zstd -19 -o \${out}
        done
    done
    echo "=== Done: \$(date) ===" >> ${log_file}
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

    if (params.species) {
        species_map = species_map.subMap(params.species)
    }

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")

    species_ch = Channel.from(
        species_map.collect { label, info ->
            def (taxon, proteome, subdir) = info
            tuple(label, file("${params.qfo_dir}/${subdir}/${proteome}_${taxon}.fasta"))
        }
    )

    indexed_ch = INDEX_SPECIES(species_ch)

    search_inputs = indexed_ch.map { species, dbs, log -> tuple(species, dbs, human_fasta) }

    SEARCH_HUMAN_VS_SPECIES(search_inputs)
}
