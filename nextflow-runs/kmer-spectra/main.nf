#!/usr/bin/env nextflow

/*
 * Nextflow pipeline: k-mer frequency spectra across datasets, alphabets, and k-sizes
 *
 * One invocation sweeps kmerseek's --kmer-stats-out over every selected dataset
 * (Swiss-Prot order-2/order-3 shuffled nulls, UniRef50/90/100) x alphabet x ksize
 * combo -- instead of a separate `nextflow run` per dataset. Nextflow's own
 * maxForks (see the sherlock/standard profiles) caps concurrency across ALL
 * selected datasets at once, so this is also safer than running several
 * uncoordinated invocations that don't know about each other's resource use.
 *
 * kmerseek branch: olgabot/kmer-stats-review-fixes (requires --stats-only support)
 * Docker image: kmerseek-spectra:latest (build with the Dockerfile in this directory)
 *
 * Usage:
 *   nextflow run main.nf -profile sherlock                       # datasets: k2,k3 (default)
 *   nextflow run main.nf -profile sherlock --datasets k2,k3,uniref50
 *   nextflow run main.nf -profile standard --datasets k2
 */

params.outdir_root = "${launchDir}"
// Comma-separated subset of ALL_DATASETS' names to actually run. Default is the
// two with a validated memory profile (k2, k3). uniref50/90/100 are opt-in until
// their peak_rss has been checked in the trace file -- add them explicitly once
// ready, e.g. --datasets k2,k3,uniref50. See README-sherlock.md.
params.datasets = 'k2,k3'
// HP-family kmin for the UniRef datasets specifically (k2/k3 always use 15,
// Swiss-Prot's floor). UniRef50/90/100 are ~100-700x bigger than Swiss-Prot by
// compressed fasta size, so this is raised by default to skip the worst
// combinatorial-degeneracy zone rather than trying to memory-size for it.
params.hp_kmin_uniref = 18
// Multiplier on indexAndSpectrum's base memory tiers (see taskMemory below), for
// the UniRef datasets. Starting estimate, not a measurement -- kmerseek hasn't
// been profiled at UniRef scale yet. Check the trace file's peak_rss after the
// first real run and correct this up or down before enabling uniref90/100.
params.mem_scale_uniref = 1
// Wall-clock limit per task. Swiss-Prot-scale combos finish well under the 4h
// default; UniRef-scale inputs will need more -- same "starting estimate" caveat.
params.task_time = '4h'
// SLURM partition (sherlock profile only). Default 'hns' -- the school-condo
// partition, ~186 GB/node. Switch to 'bigmem' (up to 4096 GB/node) for datasets
// whose in-memory proteome doesn't fit in hns at all: uniref50 hp k=26 was
// SIGKILL'd at 130K/~65M sequences processed, on an alphabet that already
// needed the 96 GB tier just for Swiss-Prot's 573K sequences -- 176 GB (hns's
// practical ceiling, see NODE_MEM_CAP below) was reached on retry and still
// wasn't enough. Raise node_mem_cap_gb to match whichever queue you pick --
// it exists specifically so a retry can't ask a node for more than it will
// ever have, which would otherwise leave the job stuck in the queue forever.
params.queue = 'hns'
params.node_mem_cap_gb = 176

// name, fasta (relative to launchDir), hp_only, hp_kmin, mem_scale.
// fasta/hp_kmin/mem_scale are resolved lazily (params.* read here, at parse time,
// so CLI overrides of hp_kmin_uniref/mem_scale_uniref apply to every UniRef entry).
def ALL_DATASETS = [
    [name: 'k2',        fasta: 'data/uniprot_sprot.ushuffle_k2.fasta.gz', hp_only: true,  hp_kmin: 15,                        mem_scale: 1],
    [name: 'k3',        fasta: 'data/uniprot_sprot.ushuffle_k3.fasta.gz', hp_only: true,  hp_kmin: 15,                        mem_scale: 1],
    [name: 'uniref50',  fasta: 'data/uniref50.fasta.gz',                  hp_only: false, hp_kmin: params.hp_kmin_uniref as Integer, mem_scale: params.mem_scale_uniref],
    [name: 'uniref90',  fasta: 'data/uniref90.fasta.gz',                  hp_only: false, hp_kmin: params.hp_kmin_uniref as Integer, mem_scale: params.mem_scale_uniref],
    [name: 'uniref100', fasta: 'data/uniref100.fasta.gz',                 hp_only: false, hp_kmin: params.hp_kmin_uniref as Integer, mem_scale: params.mem_scale_uniref],
]

def selectedNames = (params.datasets as String).split(',')*.trim() as Set
def DATASETS = ALL_DATASETS.findAll { it.name in selectedNames }
if (DATASETS.isEmpty()) {
    error "params.datasets (${params.datasets}) matched none of: ${ALL_DATASETS*.name.join(', ')}"
}
def unmatched = selectedNames - DATASETS*.name
if (unmatched) {
    error "params.datasets named unknown dataset(s) ${unmatched.join(', ')} -- valid names: ${ALL_DATASETS*.name.join(', ')}"
}

// [cli_flag, label, kmin, kmax] per alphabet. cli_flag uses clap's kebab-case;
// label uses snake_case, matching the moltype kmerseek writes into the CSV.
// hpOnly restricts to the 7 HP-family alphabets (skip protein/dayhoff) -- used
// for null-comparison datasets (ushuffle order-2/order-3 controls) where only
// the HP-family degeneracy question is being re-tested, not the full sweep.
def encodingsFor(hpOnly, hpKmin) {
    def all = [
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
    hpOnly ? all.findAll { cli_flag, label, kmin, kmax -> label.startsWith('hp') } : all
}

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
// `sinfo -p hns -o "%N %m %c"` puts every hns node at 191000 MB (~186.5 GiB) --
// hence the 176 GB default for params.node_mem_cap_gb (comfortably under that,
// with margin for the OS). If you've switched params.queue to 'bigmem' (up to
// 4096 GB/node), raise node_mem_cap_gb to match, or this cap silently keeps
// capping retries at hns-sized memory on a queue that could give you far more.
def isHpFamily = { label -> label.startsWith('hp') }
def NODE_MEM_CAP = (params.node_mem_cap_gb as Integer).GB

def taskMemory = { label, memScale, attempt ->
    def base = isHpFamily(label) ? 96.GB : 16.GB
    // memScale arrives as a String when the underlying param was set via the CLI
    // (Nextflow doesn't coerce CLI params), and MemoryUnit * String isn't defined.
    def scale = (memScale as Number)
    def requested = base * scale * attempt
    requested <= NODE_MEM_CAP ? requested : NODE_MEM_CAP
}

process indexAndSpectrum {
    tag "${dataset}_${label}_k${ksize}"
    publishDir { "${params.outdir_root}/results-${dataset}" }, mode: 'copy', pattern: '*.spectrum.csv.gz'
    publishDir { "${params.outdir_root}/results-${dataset}" }, mode: 'copy', pattern: '*.index.log'

    queue  { params.queue }
    memory { taskMemory(label, mem_scale, task.attempt) }
    time   { params.task_time }
    errorStrategy { task.exitStatus in 128..143 ? 'retry' : 'finish' }
    maxRetries 1

    input:
    tuple path(fasta), val(cli_flag), val(label), val(ksize), val(dataset), val(mem_scale)

    output:
    path "${dataset}.${label}.k${ksize}.spectrum.csv.gz"
    path "${dataset}.${label}.k${ksize}.index.log"

    script:
    def db_name   = "${dataset}.${label}.k${ksize}.kmerseek.rocksdb"
    def spectrum  = "${dataset}.${label}.k${ksize}.spectrum.csv.gz"
    def log_file  = "${dataset}.${label}.k${ksize}.index.log"
    """
    set -uo pipefail

    # --stats-only (kmerseek branch olgabot/kmer-stats-review-fixes): compute the
    # frequency spectrum straight from the in-memory signatures and skip persisting
    # a searchable index entirely -- no chunked signature storage, no SearchCache.
    # This pipeline never uses the persisted index (rm -rf'd below regardless), and
    # building it unconditionally is what broke real runs at proteome scale, in two
    # independent ways: the SearchCache is one RocksDB value holding the whole
    # inverted index, which can exceed RocksDB's ~4 GiB single-value limit at high
    # ksize + a small alphabet + a large proteome ("Invalid argument: value is too
    # large", seen on hp k=30); and chunked signature storage writes thousands of
    # small files (500K+ signatures / 100 per chunk), which hit transient
    # filesystem I/O errors under concurrent load on Sherlock's shared \$SCRATCH
    # ("IO error: ... Input/output error", seen on hp_kyte_doolittle k30). Both are
    # sidestepped entirely with --stats-only rather than working around either one.
    #
    # 2>&1 | tee (not `2> \$log_file`) so a failure's message lands in Nextflow's
    # own .command.err too -- a plain stderr redirect hid it there, and a failed
    # task never publishes \$log_file (publishDir only copies declared outputs
    # on success), so there was no way to see what went wrong without grepping
    # the work dir by hash. `pipefail` keeps kmerseek's exit code (not tee's)
    # as the pipeline's exit code.
    #
    # No `set -e` here: kept as defense in depth from before --stats-only existed
    # (when kmerseek could crash in the SearchCache/chunk-write steps *after* the
    # spectrum CSV was already fully written) -- with --stats-only there's nothing
    # left to run after the CSV write, so this realistically can't trigger anymore,
    # but checking for the completed CSV ourselves instead of blindly trusting
    # kmerseek's exit code costs nothing and catches any future surprise the same way.
    kmerseek index \\
        --input ${fasta} \\
        --encoding ${cli_flag} \\
        --ksize ${ksize} \\
        --output ${db_name} \\
        --kmer-stats-out ${spectrum} \\
        --stats-only \\
        2>&1 | tee ${log_file}
    kmerseek_exit=\$?

    if [ \$kmerseek_exit -ne 0 ]; then
        if [ -s ${spectrum} ] && grep -q 'Wrote k-mer frequency spectrum' ${log_file}; then
            echo "[wrapper] kmerseek exited \$kmerseek_exit after the spectrum CSV was already written -- treating as success" | tee -a ${log_file}
        else
            echo "[wrapper] kmerseek exited \$kmerseek_exit and the spectrum CSV was not confirmed written -- real failure" | tee -a ${log_file}
            exit \$kmerseek_exit
        fi
    fi

    # --stats-only leaves this dir tiny (a handful of near-empty RocksDB files,
    # not a full index), but drop it anyway so per-task work dirs stay clean.
    rm -rf ${db_name}
    """
}


workflow {
    println "Datasets: ${DATASETS*.name.join(', ')}"

    inputs = channel.of(*DATASETS).flatMap { ds ->
        def fastaFile = file(ds.fasta, checkIfExists: true)
        encodingsFor(ds.hp_only, ds.hp_kmin).collectMany { cli_flag, label, kmin, kmax ->
            (kmin..kmax).collect { k -> tuple(fastaFile, cli_flag, label, k, ds.name, ds.mem_scale) }
        }
    }

    indexAndSpectrum(inputs)
}
