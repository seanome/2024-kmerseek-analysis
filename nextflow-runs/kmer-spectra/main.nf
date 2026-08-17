#!/usr/bin/env nextflow

/*
 * Nextflow pipeline: k-mer frequency spectra across alphabets and k-sizes
 *
 * Reproduces ~/data/kmerseek-kmer-spectra/ by running `kmerseek index
 * --kmer-stats-out` over Swiss-Prot for every (alphabet, ksize) combo.
 *
 * kmerseek branch: olgabot/kmer-frequency-histogram
 * Docker image: kmerseek-spectra:latest (build with the Dockerfile in this directory)
 *
 * Usage:
 *   nextflow run main.nf
 *   nextflow run main.nf --fasta /path/to/uniprot_sprot.fasta.gz --outdir /path/to/results
 *   nextflow run main.nf --fasta /path/to/shuffled.fasta.gz --outdir /path/to/results --hp_only true
 */

params.fasta   = "${System.getProperty('user.home')}/data/uniprot/uniprot_sprot.fasta.gz"
params.outdir  = "${launchDir}/results"
// When true, restrict the sweep to the 7 HP-family alphabets (skip protein/dayhoff).
// Meant for null-comparison fastas (e.g. ushuffle order-2/order-3 controls) where only
// the HP-family degeneracy question is being re-tested, not the full alphabet/ksize sweep.
params.hp_only = false

// [cli_flag, label, kmin, kmax] per alphabet. cli_flag uses clap's kebab-case;
// label uses snake_case, matching the moltype kmerseek writes into the CSV.
def ALL_ENCODINGS = [
    ['protein',              'protein',             5,  15],
    ['dayhoff',               'dayhoff',             10, 20],
    ['hp',                    'hp',                  15, 30],
    ['hp-kyte-doolittle',     'hp_kyte_doolittle',   15, 30],
    ['hp-lehninger-plus-c',   'hp_lehninger_plus_c', 15, 30],
    ['hp-thomas-dill',        'hp_thomas_dill',       15, 30],
    ['hp-thomas-dill-no-c',   'hp_thomas_dill_no_c', 15, 30],
    ['hp-pbotc-1st-ed',       'hp_pbotc_1st_ed',     15, 30],
    ['hp-shuffled-control',   'hp_shuffled_control', 15, 30],
]

def ENCODINGS = params.hp_only
    ? ALL_ENCODINGS.findAll { cli_flag, label, kmin, kmax -> label.startsWith('hp') }
    : ALL_ENCODINGS


process indexAndSpectrum {
    tag "${label}_k${ksize}"
    publishDir params.outdir, mode: 'copy', pattern: '*.spectrum.csv.gz'
    publishDir params.outdir, mode: 'copy', pattern: '*.index.log'

    input:
    tuple path(fasta), val(cli_flag), val(label), val(ksize)

    output:
    path "sprot.${label}.k${ksize}.spectrum.csv.gz"
    path "sprot.${label}.k${ksize}.index.log"

    script:
    def db_name   = "sprot.${label}.k${ksize}.kmerseek.rocksdb"
    def spectrum  = "sprot.${label}.k${ksize}.spectrum.csv.gz"
    def log_file  = "sprot.${label}.k${ksize}.index.log"
    """
    set -euo pipefail

    # 2>&1 | tee (not `2> \$log_file`) so a failure's message lands in Nextflow's
    # own .command.err too -- a plain stderr redirect hid it there, and a failed
    # task never publishes \$log_file (publishDir only copies declared outputs
    # on success), so there was no way to see what went wrong without grepping
    # the work dir by hash. `pipefail` keeps kmerseek's exit code (not tee's)
    # as the pipeline's exit code.
    kmerseek index \\
        --input ${fasta} \\
        --encoding ${cli_flag} \\
        --ksize ${ksize} \\
        --output ${db_name} \\
        --kmer-stats-out ${spectrum} \\
        2>&1 | tee ${log_file}

    # The rocksdb index itself isn't needed downstream -- only the spectrum
    # CSV is. Drop it so per-task work dirs don't accumulate ~100 indices.
    rm -rf ${db_name}
    """
}


workflow {
    fasta_ch = channel.fromPath(params.fasta)

    combos = channel.of(*ENCODINGS)
        .flatMap { cli_flag, label, kmin, kmax ->
            (kmin..kmax).collect { k -> tuple(cli_flag, label, k) }
        }

    inputs = fasta_ch.combine(combos)

    indexAndSpectrum(inputs)
}
