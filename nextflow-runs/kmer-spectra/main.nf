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
// HP-family kmin. Default 15 (Swiss-Prot). UniRef50/90/100 are ~100-700x bigger
// than Swiss-Prot by compressed fasta size, so raise this (e.g. 18) to skip the
// worst combinatorial-degeneracy zone rather than trying to memory-size for it.
params.hp_kmin = 15
// Multiplier on indexAndSpectrum's base memory tiers (see taskMemory below), for
// input fastas much bigger than Swiss-Prot. This is a starting estimate, not a
// measurement -- kmerseek hasn't been profiled at UniRef scale yet. Check the
// trace file's peak_rss after the first real run and correct this up or down.
params.mem_scale = 1
// Wall-clock limit per task. Swiss-Prot combos finish well under the 4h default;
// UniRef-scale inputs will need more -- same "starting estimate" caveat as above.
params.task_time = '4h'

// params.hp_kmin arrives as a String when set via --hp_kmin on the CLI, and a
// Groovy range needs comparable endpoints -- cast before using it as a range bound.
def hpKmin = params.hp_kmin as Integer

// [cli_flag, label, kmin, kmax] per alphabet. cli_flag uses clap's kebab-case;
// label uses snake_case, matching the moltype kmerseek writes into the CSV.
def ALL_ENCODINGS = [
    ['protein',              'protein',             5,  15],
    ['dayhoff',               'dayhoff',             10, 20],
    ['hp',                    'hp',                  hpKmin, 30],
    ['hp-kyte-doolittle',     'hp_kyte_doolittle',   hpKmin, 30],
    ['hp-lehninger-plus-c',   'hp_lehninger_plus_c', hpKmin, 30],
    ['hp-thomas-dill',        'hp_thomas_dill',       hpKmin, 30],
    ['hp-thomas-dill-no-c',   'hp_thomas_dill_no_c', hpKmin, 30],
    ['hp-pbotc-1st-ed',       'hp_pbotc_1st_ed',     hpKmin, 30],
    ['hp-shuffled-control',   'hp_shuffled_control', hpKmin, 30],
]

def ENCODINGS = params.hp_only
    ? ALL_ENCODINGS.findAll { cli_flag, label, kmin, kmax -> label.startsWith('hp') }
    : ALL_ENCODINGS

// HP-family alphabets collapse 20 amino acids onto 2 symbols, so a handful of
// k-mers absorb a huge share of the proteome (see notebook 24's degeneracy-ratio
// figures). kmerseek's --kmer-stats-out apparently scales with the single
// most-degenerate k-mer's occurrence count, not the number of unique k-mers.
// This was first assumed to be a low-ksize-only problem (ksize <= 20), but
// hp_k23 and hp_k24 OOM'd too (16 GB, exit 137) -- there's no ksize past which
// HP-family alphabets are safe at the default tier, so every HP-family combo
// gets the high tier across its whole ksize range instead of trying to find
// another cutoff and getting it wrong again.
//
// `sinfo -p hns -o "%N %m %c"` puts every hns node at 191000 MB (~186.5 GiB).
// Cap requests at 176 GB -- comfortably under that on every node, with margin
// for the OS -- so a retry can never ask for more than any node will ever
// have, which would leave the job stuck in the queue forever instead of
// failing loudly.
def isHpFamily = { label -> label.startsWith('hp') }
def NODE_MEM_CAP = 176.GB

def taskMemory = { label, ksize, attempt ->
    def base = isHpFamily(label) ? 96.GB : 16.GB
    // params.mem_scale arrives as a String when set via --mem_scale on the CLI
    // (Nextflow doesn't coerce CLI params), and MemoryUnit * String isn't defined.
    def scale = (params.mem_scale as Number)
    def requested = base * scale * attempt
    requested <= NODE_MEM_CAP ? requested : NODE_MEM_CAP
}

process indexAndSpectrum {
    tag "${label}_k${ksize}"
    publishDir params.outdir, mode: 'copy', pattern: '*.spectrum.csv.gz'
    publishDir params.outdir, mode: 'copy', pattern: '*.index.log'

    memory { taskMemory(label, ksize, task.attempt) }
    time   { params.task_time }
    errorStrategy { task.exitStatus in 128..143 ? 'retry' : 'finish' }
    maxRetries 1

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
