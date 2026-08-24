#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * Containers are pinned by DIGEST, not tag. The foldseek pin previously used a bioconda
 * tag that was retagged upstream, and the run died on Sherlock with "manifest unknown"
 * -- the standard symptom. Digests are immutable, so a reviewer re-running this in six
 * months gets the same image. Human-readable versions, for Methods:
 *   foldseek 9.427df8a   hmmer 3.4   mmseqs2 18.8cc5c   hhsuite 3.3.0
 * Resolve a new digest with:
 *   curl -s 'https://quay.io/api/v1/repository/biocontainers/<tool>/tag/?specificTag=<tag>'
 */

/*
 * qfo-pfam-region-benchmark
 *
 * Domain finding, not orthology.
 *
 * Every tool here searches human query proteins against a QfO target proteome and
 * reports aligned regions. Each reported region is turned into a *domain call* by
 * transferring the Pfam domains annotated on the overlapped target interval onto the
 * query interval. The call is then scored against the query protein's own Pfam domain
 * instances: right family, right place. A tool that recovers the correct protein but
 * puts the region in the wrong place scores no better than a miss.
 *
 * This is deliberately not the pair-level ROC used by ../pfam-benchmark-tools and
 * ../qfo-pfam-benchmark. Those ask "is this human protein homologous to that species
 * protein". This asks "which stretch of this human protein is a PF00001, and did the
 * tool find it".
 *
 * Tools compared:
 *   kmerseek       region-scoped Poisson scoring, full alphabet x ksize matrix
 *   hmmer3-phmmer  single-sequence HMM search
 *   hmmer3-jackhmmer   3 iterations (params.jackhmmer_iterations)
 *   mmseqs2-seqseq / mmseqs2-iterative
 *   hhblits        profile-profile
 *   foldseek       structure-structure (AlphaFold models)
 *   hmmscan        query vs Pfam-A directly -- not a competitor, the reference ceiling
 *                  for what direct annotation achieves without any target proteome
 *
 * Runs on Stanford Sherlock (SLURM + Apptainer). See README-sherlock.md and Makefile.
 */

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
def home = System.getProperty('user.home')

params.qfo_dir       = "${home}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.annotations   = "${projectDir}/../../results/pfam_benchmark/annotations"
params.outdir        = "${home}/data/qfo-pfam-region-benchmark/results"
params.structures    = "${home}/data/alphafold_structures"

// Pfam-A.hmm for the hmmscan annotation ceiling. Skipped when absent.
params.pfam_hmm      = "${home}/data/pfam/Pfam-A.hmm"

// --- query-protein covariates ---------------------------------------------
// Results are cut by HGNC gene group, dN/dS, pLDDT and disorder, the same axes the
// 200-series notebooks stratify on. Each is optional; a missing file drops that axis
// and leaves the rest working.
params.hgnc_file  = "${home}/data/gencode/results-human-mouse-orthologs/hgnc_complete_set.txt"
params.omega_file = "${projectDir}/../human-mouse-dnds-omega/results/omega_results.tsv"
params.mobidb_cache = null   // optional curated disorder; pLDDT<50 proxy is always computed

// Second, function-anchored truth set. Pfam-A domains are DEFINED by profile HMMs and
// phmmer/jackhmmer/hhblits are profile methods, so Pfam truth is circular with those
// baselines -- and worse, a region Pfam never annotated is labelled absent, so every
// correct cryptic-domain rescue scores as a false positive. Swiss-Prot FT features are
// literature-curated and defined by function, circular with neither the profile baselines
// nor the structure baselines. Set to null to run the Pfam arm alone.
params.swissprot_dat = "${home}/data/uniprot/uniprot_sprot.dat.gz"

// Pfam-N: the explicit "Pfam-A HMMs missed these" label set, and the only truth set here
// that is not circular with the profile baselines -- it exists where those HMMs
// failed. The gray-zone convention stops the benchmark charging for calls in Pfam-silent
// territory; this is what turns a slice of them back into scoreable true positives.
//
// Built ahead of the run rather than inside it: the source is ~17.4 GB streamed from EBI
// and compute nodes here have no outbound internet. `make pfamn-truth` writes this dir.
params.pfamn_dir = null

// M-CSA catalytic residues: function defined by mechanism, curated from the literature,
// circular with neither the profile HMMs nor any fold classification. Coverage is small
// (95 human proteins, 0.5% of the query set), so it is a VIGNETTE like the MHC block --
// do not let it carry a headline number. Built by `make mcsa-truth`.
params.mcsa_dir = null

// Held-out evaluation. Nothing here learns from the data, so there is no model to
// overfit -- but picking the best of 113 alphabet x ksize combos on the same instances
// you report IS selection, and reporting the winner on the data that chose it is
// optimistically biased. Grouped by Pfam family so a family cannot appear on both sides.
params.split_by         = "family"
params.holdout_fraction = 0.5
params.split_seed       = 20260818

// Strict "correctly parsed" criterion used by structure-based domain parsers, reported
// alongside the looser params.min_overlap.
params.strict_iou = 0.8

// HHblits background database (UniRef30). Without it hhblits runs single-sequence
// profiles and lands near phmmer -- see README-sherlock.md for the download.
params.hhblits_db    = null

// --- kmerseek alphabet x ksize matrix -------------------------------------
// Same alphabet set as the kmer-spectra pipeline. HP floor is 18, not 15: a 2-letter
// alphabet at k=15 has 32768 possible k-mers against ~20k x 20k proteins, and the
// measured output volume at k=18 was already 838 MB compressed for the *smallest*
// species. k=15-17 is a separate, deliberate experiment, not part of this sweep.
// Scope a run down without editing the matrix, for smoke tests on a mini set.
//   --target_species   comma-separated TARGET labels, e.g. "yeast,ecoli"
//   --kmerseek_combos  comma-separated encoding:ksize, e.g. "protein20:10,gbmr4:14"
// Both default to null, meaning the full sweep.
//
// Human is always the query and never appears in the species list. Every search is human
// against one target proteome, so --target_species yeast,ecoli means two searches:
// human_vs_yeast and human_vs_ecoli. The old --species spelling still works.
params.target_species  = null
params.species         = null
params.kmerseek_combos = null

// Low-complexity k-mer removal, swept as a toggle rather than fixed: every alphabet and
// ksize runs both with and without it, doubling the search count. Whether dropping
// homopolymer-like k-mers helps depends on the alphabet, since a 2-letter alphabet
// generates far more low-complexity k-mers than a 20-letter one.
params.low_complexity_toggle = [false, true]

// [cli_flag, label, kmin, kmax]. In kmerseek v0.4.0 the CLI name and the moltype written
// into the CSV are the same string, so one column serves both.
//
// Every alphabet was renamed in PR #43 to state its class count: protein is protein20,
// dayhoff is dayhoff6, hp_lehninger is hp_lehninger2, hp_lehninger_plus_c is
// hp_lehninger_c_nonpolar2, hp_shuffled_control is hp_random_control2. Results produced
// under the old names will not join these labels.
//
// Twelve ksizes for the HP family, ten for everything else, from a bit-matched floor. The
// HP alphabets get the wider range because they are what the paper is testing and the k
// optimum is least constrained there.
//
// The floor uses real entropy, not log2(classes). log2(n) assumes every class is equally
// likely, which overstates every coarse alphabet. The bits/symbol below come from
// amino-acid background frequencies grouped as kmerseek groups them
// (notebooks/ortholog_analysis_utils.entropy_per_symbol). HP carries 0.994 bits/symbol, so
// its k18 floor is 17.9 bits, and every kmin is round(17.9 / bits). Below that floor a
// coarse alphabet produces prohibitive output volume.
//
// Two entries contradict class count, which is why entropy is measured rather than
// assumed. hp_lehninger_hpc3 has three classes but 1.128 bits/symbol against HP's 0.994,
// because cysteine is ~1.4% of residues. gbmr7 carries less information than wwmj5, 1.976
// against 2.197, despite two more classes, because its classes are unbalanced.
def ALL_ENCODINGS = [
    ['protein20', 'protein20', 4, 13],                      // 4.176 bits/sym, 10 ksizes
    ['uniprot18', 'uniprot18', 5, 14],                      // 3.951 bits/sym, 10 ksizes
    ['hsdm17', 'hsdm17', 5, 14],                            // 3.742 bits/sym, 10 ksizes
    ['wass14', 'wass14', 5, 14],                            // 3.626 bits/sym, 10 ksizes
    ['mmseqs12', 'mmseqs12', 5, 14],                        // 3.293 bits/sym, 10 ksizes
    ['sdm12', 'sdm12', 6, 15],                              // 3.127 bits/sym, 10 ksizes
    ['dayhoff6', 'dayhoff6', 8, 17],                        // 2.278 bits/sym, 10 ksizes
    ['wwmj5', 'wwmj5', 8, 17],                              // 2.197 bits/sym, 10 ksizes
    ['gbmr7', 'gbmr7', 9, 18],                              // 1.976 bits/sym, 10 ksizes
    ['gbmr4', 'gbmr4', 12, 21],                             // 1.522 bits/sym, 10 ksizes
    ['hp_lehninger_hpc3', 'hp_lehninger_hpc3', 16, 27],     // 1.128 bits/sym, 12 ksizes
    ['hp_lehninger2', 'hp_lehninger2', 18, 29],             // 1.000 bits/sym, 12 ksizes
    ['hp_lehninger_c_nonpolar2', 'hp_lehninger_c_nonpolar2', 18, 29],// 0.999 bits/sym, 12 ksizes
    ['hp_pbotc_1st_ed2', 'hp_pbotc_1st_ed2', 18, 29],       // 0.994 bits/sym, 12 ksizes
    ['hp_thomas_dill2', 'hp_thomas_dill2', 19, 30],         // 0.966 bits/sym, 12 ksizes
    ['hp_thomas_dill_no_c2', 'hp_thomas_dill_no_c2', 19, 30],// 0.951 bits/sym, 12 ksizes
    ['hp_kyte_doolittle2', 'hp_kyte_doolittle2', 19, 30],   // 0.937 bits/sym, 12 ksizes
]

// kmerseek search filters. min_region_score is the region-scoped cutoff: -log10 of the
// region's Poisson tail, so 1.3 ~ p=0.05 and 3.16 ~ p=0.0007. It is OR'd with
// max_query_pvalue, so a strong sub-protein domain hit survives a weak whole-query
// p-value -- which is the entire point of region scoring for domain finding.
params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_query_pvalue = 0.05
params.min_region_score = 1.3

// --- baseline tool settings ------------------------------------------------
params.mmseqs2_sensitivity = 7
params.mmseqs2_iterations  = 3
params.jackhmmer_iterations = 3
params.evalue_report       = 10.0

// --- domain-call scoring ---------------------------------------------------
// A transferred call counts as hitting a true domain instance when the two intervals
// reciprocally overlap by at least this fraction. 0.5 is the usual domain-boundary
// convention; the eval also emits the full IoU distribution so this can be re-cut
// downstream without re-running the sweep.
params.min_overlap = 0.5

// Toggles
params.skip_kmerseek  = false
params.skip_baselines = false
params.skip_foldseek  = false
params.skip_folddisco = false

// Reseek (Edgar 2024, Bioinformatics btae687). Structure search over a ~8.6e10-state
// "mega-alphabet" -- it scales alphabet size UP where kmerseek scales it DOWN to two
// letters. Both working is itself the interesting result: it says alphabet size is not the
// binding constraint. Foldseek's ">20 letters gives only incremental gains" finding is
// disputed, so this is a live disagreement rather than settled ground.
params.skip_reseek = false
// -fast, -sensitive or -verysensitive; one is required. Strongest by default.
params.reseek_mode = "verysensitive"

// ProstT5 via Foldseek: predicts 3Di directly from amino acid sequence, so it needs NO
// structures on either side. That makes it the closest published thing to what kmerseek
// claims. structural signal without structure prediction. and therefore the baseline
// the paper most has to differentiate from. It still depends on Foldseek and on a target
// database, which is the differentiation the review points at.
params.skip_prostt5   = false
params.prostt5_weights = null   // set to a pre-downloaded weights dir to skip the fetch

// ProstT5 is a 3B-parameter T5 encoder and Foldseek runs it over each sequence whole, so
// peak memory scales with the SQUARE of sequence length (self-attention) times the number
// of threads holding a sequence at once. Titin (Q8WZ42) is 34_350 aa and sits in every
// human proteome file here; its attention matrix alone is far past any node's RAM, so it
// OOM-killed `createdb` before the search arm ever started. Sequences longer than this are
// dropped from BOTH sides of this arm -- target proteomes have their own giants -- and
// written to the published .prostt5_skipped.tsv.
//
// The cap is a real limit of ProstT5, not of the benchmark: no other arm here has it, and
// the eval's reachable-domain denominator is tool-independent, so these instances count
// against ProstT5's recall. That is the honest accounting. What must not happen is the
// gap going unrecorded, hence the skipped list. On the full human proteome a cap of 6000
// drops 11 proteins carrying 664 of 50_185 domain instances (1.3%); 8000 drops 4 proteins
// and 347 instances (0.7%). Raise it if the trace shows peak_rss with headroom to spare.
params.prostt5_max_len = 6000

// Deliberately NOT the `high_cpu` label. Threads are the multiplier on ProstT5's peak
// memory, so this arm buys RAM rather than cores; `high_cpu`'s 16 threads on 64 GB is
// exactly the shape that died. Set in the process body from params, because a bare
// `process { cpus = ... }` in a profile does not override a body directive while a
// `withLabel:`/`withName:` selector does.
params.prostt5_cpus   = 4
params.prostt5_memory = '64 GB'
params.prostt5_time   = '48h'

// folddisco, rebuilt with the ENTRYPOINT cleared. The upstream image sets
// ENTRYPOINT ["/usr/local/bin/folddisco"], which under Apptainer becomes the SIF runscript
// and stops Nextflow from running `/bin/bash .command.run` -- the task exits 1 with an
// entirely empty .command.out. Docker bypasses the entrypoint the way Nextflow invokes it,
// so this reproduces only on the cluster. See Dockerfile.folddisco.
params.folddisco_image = 'docker.io/olgabot/folddisco:2026-08-20-noentrypoint'

// Per-domain-pair percent identity, the twilight-zone axis. Skipping it removes the
// stratification the central claim is stated on, so only skip for a quick smoke test.
params.skip_identity  = false

// Structure download. `--fetch_structures true` runs it as SLURM jobs, one per species,
// instead of on the login node where a multi-GB transfer gets killed.
params.fetch_structures = false
params.afdb_base = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"

// Folddisco queries one structure per invocation, so ~19.4k human queries per species are
// spread over this many SLURM tasks rather than serialised into one.
params.folddisco_chunks = 20
params.folddisco_top    = 1000
params.run_hmmscan    = file(params.pfam_hmm).exists()

// ---------------------------------------------------------------------------
// Species: human is query-only; the other 9 are targets.
// ---------------------------------------------------------------------------
def ALL_SPECIES = [
    [label: "mouse",       taxon: "10090",  proteome: "UP000000589", subdir: "Eukaryota", mya: 100],
    [label: "chicken",     taxon: "9031",   proteome: "UP000000539", subdir: "Eukaryota", mya: 300],
    [label: "zebrafish",   taxon: "7955",   proteome: "UP000000437", subdir: "Eukaryota", mya: 430],
    [label: "ciona",       taxon: "7719",   proteome: "UP000008144", subdir: "Eukaryota", mya: 550],
    [label: "fly",         taxon: "7227",   proteome: "UP000000803", subdir: "Eukaryota", mya: 600],
    [label: "worm",        taxon: "6239",   proteome: "UP000001940", subdir: "Eukaryota", mya: 650],
    [label: "yeast",       taxon: "559292", proteome: "UP000002311", subdir: "Eukaryota", mya: 900],
    [label: "arabidopsis", taxon: "3702",   proteome: "UP000006548", subdir: "Eukaryota", mya: 1500],
    [label: "ecoli",       taxon: "83333",  proteome: "UP000000625", subdir: "Bacteria",  mya: 2000],
]

def requested = params.target_species ?: params.species
def SPECIES = requested
    ? ALL_SPECIES.findAll { it.label in requested.tokenize(',')*.trim() }
    : ALL_SPECIES

if (SPECIES.isEmpty()) {
    error "No target species matched '${requested}'. Known targets: " +
          "${ALL_SPECIES*.label.join(', ')} (human is the query and is not a target)"
}

// HP-family alphabets at low ksize need far more RAM than everything else: a handful of
// k-mers absorb a large share of the proteome, and the inverted index scales with the
// most-degenerate k-mer's occurrence count. Size for the known-risk zone up front rather
// than retrying into it -- each retry costs a full SLURM requeue.
// Class count is now in the name, so the memory rule reads it rather than guessing from a
// prefix: the fewer classes, the more a handful of k-mers absorb the proteome, and the
// more RAM the inverted index needs. hp_*2 and hp_*3 are the degenerate ones.
def isDegenerateHp = { label -> label ==~ /.*[23]$/ }

// Sized for full QfO proteomes. A mini/smoke run indexes a few hundred sequences and
// needs nothing like this, so both figures are params -- the `mini` profile lowers them
// rather than forcing a 128 GB request for a 300-protein test set.
params.kmerseek_memory         = '32 GB'
params.kmerseek_memory_hp_lowk = '128 GB'

def kmerseekMemory = { label, ksize, attempt ->
    def base = (isDegenerateHp(label) && ksize <= 20)
        ? MemoryUnit.of(params.kmerseek_memory_hp_lowk)
        : MemoryUnit.of(params.kmerseek_memory)
    base * attempt
}

// ===========================================================================
// GROUND TRUTH — query-side Pfam domain instances
// ===========================================================================

process buildDomainTruth {
    /*
     * Two products, both derived from results/pfam_benchmark/annotations:
     *   human_domain_truth.parquet  — the answer key: every Pfam domain instance on a
     *                                 human protein, with its interval.
     *   <species>_domain_map.parquet — the transfer table: every Pfam domain instance on
     *                                 a target protein, used to label an aligned region.
     * Nothing here depends on any tool, so it runs once and is reused by every arm.
     */
    label 'python'
    publishDir "${params.outdir}/truth", mode: 'copy'

    input:
    path annotations_dir

    output:
    path "human_domain_truth.parquet", emit: truth
    path "*_domain_map.parquet",       emit: maps
    path "truth_summary.json",         emit: summary

    script:
    """
    build_domain_truth.py \\
        --annotations ${annotations_dir} \\
        --truth-out   human_domain_truth.parquet \\
        --map-outdir  . \\
        --summary-out truth_summary.json \\
        --split-by         ${params.split_by} \\
        --holdout-fraction ${params.holdout_fraction} \\
        --split-seed       ${params.split_seed}
    """
}

process buildSwissprotTruth {
    /*
     * Same output schema as buildDomainTruth, so the entire scoring path runs against it
     * unchanged. The pfam_id column carries a Swiss-Prot feature type (ACT_SITE,
     * TRANSMEM, ...) rather than a Pfam accession; the name is kept for compatibility.
     */
    label 'python'
    publishDir "${params.outdir}/truth_swissprot", mode: 'copy'

    input:
    tuple path(sprot_dat), path(annotations_dir)

    output:
    path "human_swissprot_truth.parquet", emit: truth
    path "*_domain_map.parquet",          emit: maps
    path "swissprot_summary.json",        emit: summary

    script:
    """
    build_swissprot_truth.py \\
        --sprot-dat   ${sprot_dat} \\
        --annotations ${annotations_dir} \\
        --truth-out   human_swissprot_truth.parquet \\
        --map-outdir  . \\
        --summary-out swissprot_summary.json
    """
}

process domainIdentity {
    /*
     * Percent identity between each human domain instance and the closest same-family
     * domain in the target proteome. This is the twilight-zone axis, and the claim under
     * test lives on it: HP patterning is supposed to survive BELOW the ~30-40% identity
     * where profile methods lose the signal.
     *
     * Deliberately measured with MMseqs2 rather than by any tool being scored, and over
     * the TRUTH pairs rather than over anyone's predictions, so identity is a property of
     * the benchmark rather than of the method. Using a tool's own alignment would let each
     * method define the difficulty of its own test.
     *
     * Emits raw TSV only. Parsing lives in parseIdentity because this container has
     * MMseqs2 and no python -- running polars here exited 127, command not found.
     */
    tag "${species}"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    label 'high_cpu'

    input:
    tuple val(species), path(human_domains_fa), path(target_domains_fa)

    output:
    tuple val(species), path("${species}.identity.tsv")

    script:
    """
    set -euo pipefail
    mkdir -p tmp
    # Permissive on purpose: the job is to MEASURE identity for remote pairs, not to decide
    # whether they are homologous -- the shared Pfam family already decided that. A strict
    # search would silently drop the twilight-zone pairs this axis exists for.
    mmseqs easy-search \\
        ${human_domains_fa} ${target_domains_fa} \\
        ${species}.identity.tsv tmp \\
        --threads ${task.cpus} \\
        -s 7.5 -e 10000 --max-seqs 300 \\
        --format-output "query,target,pident,alnlen,evalue"
    """
}

process parseIdentity {
    tag "${species}"
    label 'python'
    publishDir "${params.outdir}/identity", mode: 'copy', pattern: '*.parquet'

    input:
    tuple val(species), path(tsv)

    output:
    tuple val(species), path("${species}.domain_identity.parquet")

    script:
    """
    parse_domain_identity.py --hits ${tsv} --out ${species}.domain_identity.parquet
    """
}

process fetchStructures {
    /*
     * Download one species' AlphaFold structures as a SLURM job.
     *
     * This ran on the login node and was SIGKILLed mid-download (exit 137): Sherlock
     * enforces limits there and a multi-GB sustained transfer trips them. One task per
     * species also means a kill costs one proteome rather than the whole set, and the
     * script is resumable so a retry continues instead of restarting.
     *
     * No container. The script needs curl, tar and gzip, which the cluster has natively
     * and the pipeline image does not carry -- adding curl to that image would mean a
     * rebuild and push for a step that needs nothing from it.
     *
     * Writes straight into params.structures and emits only a small manifest. Declaring a
     * 20k-file directory as an output would make Nextflow stage every file, and a
     * directory output under storeDir is the shape that produced recurring "Directory not
     * empty" failures in a sibling pipeline here.
     */
    tag "${species}"
    cpus 2
    memory '8 GB'
    time '24h'
    // A transfer that dies partway is normal on a shared filesystem; resume rather than
    // fail the run, since the script picks up where it stopped.
    errorStrategy 'retry'
    maxRetries 2

    input:
    tuple val(species), path(acc_dir)

    output:
    tuple val(species), path("${species}.fetch.log")

    script:
    """
    set -euo pipefail

    # Sherlock compute nodes have NO outbound internet -- established 2026-08-20 when the
    # ProstT5 weights job hit this same check. So this process cannot work there and the
    # login-node fallback is the real path; the check stays so the failure is instant and
    # named rather than a stall.
    if ! curl --fail --silent --head --max-time 30 "${params.afdb_base}/" >/dev/null 2>&1; then
        echo "no outbound internet from this compute node -- cannot reach ${params.afdb_base}" >&2
        echo "Run the fetch on a login node instead:" >&2
        echo "  PARALLEL=2 bin/fetch_alphafold_structures.sh data/structures data/structures/_accessions ${species}" >&2
        exit 1
    fi

    mkdir -p ${params.structures}
    PARALLEL=${task.cpus} FLAT_CACHE=${params.structures} \\
      ${projectDir}/bin/fetch_alphafold_structures.sh \\
        ${params.structures} ${acc_dir} ${species} 2>&1 | tee ${species}.fetch.log
    """
}

process extractDomainSequences {
    tag "${label}"
    label 'python'

    input:
    tuple val(label), path(truth), path(fasta)

    output:
    tuple val(label), path("${label}.domains.fasta")

    script:
    """
    extract_domain_sequences.py \\
        --truth ${truth} --fasta ${fasta} --out ${label}.domains.fasta
    """
}

process buildQueryCovariates {
    /*
     * Per-query-protein biology: HGNC gene group, dN/dS, mean pLDDT, disorder fraction.
     * pLDDT and its disorder proxy are parsed from the same AlphaFold .cif files the
     * Foldseek arm already stages, so this costs no extra download.
     */
    label 'python'
    publishDir "${params.outdir}/truth", mode: 'copy'

    input:
    tuple path(truth), path(hgnc), path(omega), path(structures)

    output:
    path "human_query_covariates.parquet", emit: covariates
    path "covariates_summary.json",        emit: summary

    script:
    def hgnc_arg   = hgnc.name   == 'NO_HGNC'   ? "" : "--hgnc ${hgnc}"
    def omega_arg  = omega.name  == 'NO_OMEGA'  ? "" : "--omega ${omega}"
    def struct_arg = structures.name == 'NO_STRUCTURES' ? "" : "--structures ${structures}"
    def mobidb_arg = params.mobidb_cache ? "--mobidb ${params.mobidb_cache}" : ""
    """
    build_query_covariates.py \\
        --truth       ${truth} \\
        ${hgnc_arg} ${omega_arg} ${struct_arg} ${mobidb_arg} \\
        --out         human_query_covariates.parquet \\
        --summary-out covariates_summary.json
    """
}

// ===========================================================================
// KMERSEEK — fused index + search, region-level output
// ===========================================================================

process kmerseekIndexAndSearch {
    /*
     * Index and search in one task so the RocksDB index never becomes a declared
     * output: it is built in the task work dir, used once, and deleted before the task
     * exits. Steady-state disk is bounded by (maxForks x one index), not by the whole
     * 113-combo matrix. An earlier all-vs-all run died with "No space left on device"
     * doing this the other way.
     *
     * Tradeoff: -resume after a crash re-indexes an incomplete combo instead of reusing
     * a saved index. Accepted -- the index is cheap relative to the search.
     */
    tag "${species}_${label}_k${ksize}_lc${lowcomp}"
    storeDir "${params.outdir}/kmerseek"

    memory { kmerseekMemory(label, ksize, task.attempt) }
    // Retry the OOM signals (128..143), stop the run on anything else. Deliberately not
    // 'ignore': a combo that dies and gets skipped leaves an empty result that reads
    // downstream as "this alphabet found nothing", which is indistinguishable from a real
    // negative. That has already happened once on this project -- 17 combos silently
    // searched ~1000 of 19,696 queries and looked like genuine misses. Failing loudly and
    // resuming costs queue time; a silent partial costs a wrong conclusion.
    errorStrategy { task.exitStatus in 128..143 ? 'retry' : 'finish' }
    maxRetries 2

    input:
    tuple val(species), path(species_fasta), val(cli_flag), val(label), val(ksize),
          val(lowcomp), path(human_fasta)

    // ONE path output, deliberately. storeDir supports only val/path outputs -- a tuple
    // output silently disabled it ("storeDir can only be used with `val` and `path`
    // outputs"), so nothing was being persisted and -resume would have recomputed all 1017
    // searches. A sibling pipeline in this repo also hit a recurring storeDir "Directory
    // not empty" failure from a two-output design, fixed the same way: collapse to one.
    //
    // The (species, tool, variant) metadata is not lost, it is recovered in the workflow
    // from the filename, which already encodes all three.
    output:
    path "human_vs_${species}.${label}.k${ksize}.lc${lowcomp}.regions.parquet",  emit: regions
    path "spectrum.${species}.${label}.k${ksize}.lc${lowcomp}.csv.gz",           emit: spectrum

    script:
    def slug      = "${label}.k${ksize}.lc${lowcomp}"
    def index_dir = "${species}.${slug}.kmerseek.rocksdb"
    def out_zst   = "human_vs_${species}.${slug}.regions.csv.zst"
    def out_pq    = "human_vs_${species}.${slug}.regions.parquet"
    def log_file  = "human_vs_${species}.${slug}.log"
    def spectrum  = "spectrum.${species}.${slug}.csv.gz"
    // The new CLI treats --remove-low-complexity as a presence-only index flag. Search
    // inherits the index setting when the option is omitted, so false emits nothing and
    // true emits the flag without a value.
    def lc_flag   = lowcomp ? "--remove-low-complexity" : ""
    """
    set -euo pipefail

    echo "=== Index: ${species} ${cli_flag} k=${ksize} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}

    # --kmer-stats-out writes the k-mer frequency spectrum for this proteome under this
    # alphabet/ksize/low-complexity setting. Kept as a first-class output: the spectra are
    # what show WHY an alphabet behaves as it does, and the with/without low-complexity
    # pair is only interpretable if both spectra exist.
    kmerseek index \\
        --alphabet ${cli_flag} \\
        --ksize    ${ksize} \\
        --input    ${species_fasta} \\
        --output   ${index_dir} \\
        ${lc_flag} \\
        --kmer-stats-out ${spectrum} \\
        2>&1 | tee -a ${log_file}

    echo "=== Search: human vs ${species} ===" | tee -a ${log_file}

    # --min-region-score is OR'd with --max-query-pvalue inside kmerseek, so this keeps
    # sub-protein domain hits whose whole-query p-value is unimpressive. Do not "tighten"
    # this by also lowering --max-query-pvalue expecting fewer rows; the OR means the
    # looser of the two governs.
    kmerseek search \\
        --alphabet ${cli_flag} \\
        --ksize    ${ksize} \\
        --query    ${human_fasta} \\
        --target   ${index_dir} \\
        ${lc_flag} \\
        --threshold         ${params.threshold} \\
        --min-shared-kmers  ${params.min_shared_kmers} \\
        --max-query-pvalue  ${params.max_query_pvalue} \\
        --min-region-score  ${params.min_region_score} \\
        2>> ${log_file} \\
        | zstd -T2 -o ${out_zst} \\
        || true

    touch ${out_zst}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "regions csv.zst: \$(du -sh ${out_zst} | cut -f1)" | tee -a ${log_file}

    # Straight to parquet, dropping columns no downstream step reads. Both formats kept
    # around for 1017 result files is the disk blow-up this design avoids.
    # Test the DECOMPRESSED stream, not the file size. zstd emits a 13-byte frame header
    # for empty input, so [ -s file ] is true for a search that found nothing and polars
    # then dies with NoDataError. A combo finding zero matches is a real result -- at
    # protein k=10 across 2000 MYA it is the expected one -- and must not fail the run.
    if [ -n "\$(zstd -dc ${out_zst} | head -c 1)" ]; then
        python3 - << 'PYEOF'
import polars as pl
DROP = ["query_md5", "target_md5", "region_subseq", "target_subseq", "moltype_seq"]
try:
    lf = pl.scan_csv("${out_zst}", ignore_errors=True)
    cols = [c for c in lf.collect_schema().names() if c not in DROP]
    lf.select(cols).sink_parquet("${out_pq}", compression="zstd", compression_level=9)
except pl.exceptions.NoDataError:
    # Header-only or otherwise contentless; same meaning as the empty branch below.
    open("${out_pq}", "wb").close()
PYEOF
    else
        touch ${out_pq}
    fi
    touch ${spectrum}
    rm -f ${out_zst}
    rm -rf ${index_dir}

    echo "regions parquet: \$(du -sh ${out_pq} | cut -f1)" | tee -a ${log_file}
    """
}

// ===========================================================================
// BASELINES — every one emits the same normalized 8-column TSV:
//   query, target, qstart, qend, tstart, tend, score, evalue
// Query/target intervals are both required: the target interval selects which Pfam
// domain gets transferred, the query interval is what gets scored.
// ===========================================================================

process phmmerSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/hmmer3_phmmer", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_fasta), path(human_fasta)

    output:
    tuple val(species), val("hmmer3_phmmer"), val("default"),
          path("human_vs_${species}.hmmer3_phmmer.tsv.gz")

    script:
    // domtblout: \$4 query, \$1 target, \$16/\$17 hmm (= query profile) coords,
    // \$20/\$21 env (= target sequence) coords, \$14 domain bitscore, \$13 i-Evalue.
    """
    set -euo pipefail
    phmmer \\
        --domtblout /dev/stdout \\
        --tblout /dev/null \\
        -o /dev/stderr \\
        --noali \\
        -E ${params.evalue_report} \\
        --cpu ${task.cpus} \\
        ${human_fasta} ${species_fasta} \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$4 "\\t" \$1 "\\t" \$16 "\\t" \$17 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > human_vs_${species}.hmmer3_phmmer.tsv.gz
    """
}

process jackhmmerSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/hmmer3_jackhmmer", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_fasta), path(human_fasta)

    output:
    tuple val(species), val("hmmer3_jackhmmer"), val("n${params.jackhmmer_iterations}"),
          path("human_vs_${species}.hmmer3_jackhmmer.tsv.gz")

    script:
    // jackhmmer iterates the query profile against the target proteome. --domtblout holds
    // the FINAL iteration only, which is what should be scored -- the intermediate
    // iterations are the search getting there, not its answer.
    """
    set -euo pipefail
    jackhmmer \\
        -N ${params.jackhmmer_iterations} \\
        --domtblout /dev/stdout \\
        --tblout /dev/null \\
        -o /dev/stderr \\
        --noali \\
        -E ${params.evalue_report} \\
        --cpu ${task.cpus} \\
        ${human_fasta} ${species_fasta} \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$4 "\\t" \$1 "\\t" \$16 "\\t" \$17 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > human_vs_${species}.hmmer3_jackhmmer.tsv.gz
    """
}

process mmseqs2Search {
    tag "human_vs_${species} [${variant}]"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/${variant}", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), val(variant), val(num_iter), path(species_fasta), path(human_fasta)

    output:
    tuple val(species), val(variant), val("s${params.mmseqs2_sensitivity}"),
          path("human_vs_${species}.${variant}.tsv.gz")

    script:
    def iter_flag = num_iter > 1 ? "--num-iterations ${num_iter}" : ""
    """
    set -euo pipefail
    mkdir -p mmseqs_tmp
    mmseqs easy-search \\
        ${human_fasta} ${species_fasta} \\
        out.tsv mmseqs_tmp \\
        --threads ${task.cpus} \\
        -s ${params.mmseqs2_sensitivity} \\
        ${iter_flag} \\
        --max-seqs 1000 \\
        -e ${params.evalue_report} \\
        --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"
    gzip -c out.tsv > human_vs_${species}.${variant}.tsv.gz
    """
}

process hhblitsSearch {
    /*
     * Profile-profile. Without params.hhblits_db this builds single-sequence profiles and
     * performs close to phmmer -- the run is still valid, it just is not measuring what
     * HHblits is famous for. README-sherlock.md has the UniRef30 download.
     */
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/hhsuite@sha256:4bf9bb5229de18f522a94f4443c19fdcbb0f0cb0e6ea92f5390aa170bcb0a24f'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/hhblits", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_hhdb), path(human_hhdb)

    output:
    tuple val(species), val("hhblits"), val(params.hhblits_db ? "uniref30" : "single_seq"),
          path("human_vs_${species}.hhblits.tsv.gz")

    script:
    // -blasttab: 1 qseqid 2 sseqid 3 pident 4 length 5 mismatch 6 gapopen
    //            7 qstart 8 qend 9 sstart 10 send 11 evalue 12 bitscore
    """
    set -euo pipefail
    ffindex_apply \\
        ${human_hhdb}/a3m.ffdata ${human_hhdb}/a3m.ffindex \\
        -d results.ffdata -i results.ffindex \\
        -- hhsearch -i stdin -d ${species_hhdb}/hhm -blasttab /dev/stdout -cpu 1 -v 0

    ffindex_apply results.ffdata results.ffindex -- cat \\
    | awk 'NF >= 12 {print \$1 "\\t" \$2 "\\t" \$7 "\\t" \$8 "\\t" \$9 "\\t" \$10 "\\t" \$12 "\\t" \$11}' \\
    | gzip -c > human_vs_${species}.hhblits.tsv.gz
    """
}

process foldseekSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/foldseek@sha256:c46d6fb854099780597e3adfa48e93c991f4b4d542391c144b9cae4de1ed22f9'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_structs), path(human_structs)

    output:
    tuple val(species), val("foldseek"), val("3di_aa"),
          path("human_vs_${species}.foldseek.tsv.gz")

    script:
    """
    set -euo pipefail
    mkdir -p foldseek_tmp
    foldseek easy-search \\
        ${human_structs}/ ${species_structs}/ \\
        out.tsv foldseek_tmp \\
        --threads ${task.cpus} \\
        -e ${params.evalue_report} \\
        --max-seqs 1000 \\
        --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"

    # Foldseek names rows by structure filename (AF-<acc>-F1-model_v6.cif). Reduce to the
    # bare UniProt accession so it joins the Pfam annotation tables like every other tool.
    awk -F'\\t' 'BEGIN{OFS="\\t"} {
        for (i = 1; i <= 2; i++) {
            if (\$i ~ /^AF-/) { split(\$i, p, "-"); \$i = p[2] }
        }
        print
    }' out.tsv | gzip -c > human_vs_${species}.foldseek.tsv.gz
    """
}

process reseekConvert {
    /*
     * Build a Reseek .bcb database from a directory of structures. Structure-based, so it
     * sits under the same staged-structures guard as Foldseek and Folddisco.
     */
    tag "${species}"
    container 'quay.io/biocontainers/reseek@sha256:24f7c37150dd2c2f2f322b1387a08d2d1a4a279f46f98f1051f1745417675752'
    label 'high_cpu'
    storeDir "${params.outdir}/reseek_db"

    input:
    tuple val(species), path(structures)

    output:
    tuple val(species), path("${species}.bca")

    script:
    """
    set -euo pipefail
    # -bca, not -bcb: .bca is the binary C-alpha format Reseek recommends for databases.
    # -threads, not -t. Both taken from `reseek` with no arguments in the pinned image,
    # after -bcb failed with "Unknown option bcb" -- the flags used before came from a
    # README summary rather than from the binary.
    # STRUCTS accepts a directory and .cif/.mmcif, both confirmed in that same usage text.
    reseek -convert ${structures} -bca ${species}.bca \\
        -threads ${task.cpus} -log ${species}.convert.log
    """
}

process reseekSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/reseek@sha256:24f7c37150dd2c2f2f322b1387a08d2d1a4a279f46f98f1051f1745417675752'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/reseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(db), path(human_structures)

    output:
    tuple val(species), val("reseek"), val("sensitive"),
          path("human_vs_${species}.reseek.tsv.gz")

    script:
    // Bound as locals rather than interpolated inline. `${db}` followed by more flags
    // tripped the Groovy lexer at that column; naming them keeps the script block
    // plain text.
    def q_dir   = human_structures.toString()
    def db_file = db.toString()
    // aq, not pctid, in the score slot: it is Reseek's own homology measure (alignment
    // quality 0-1, >0.5 suggests homology) and leads its default output. pctid is percent
    // identity, which ranks similar sequences rather than probable homologs.
    // Reseek DOES report an E-value -- an earlier note here claimed it did not.
    def cols    = "query+target+qlo+qhi+tlo+thi+aq+evalue"
    def mode    = params.reseek_mode
    """
    set -euo pipefail
    # One of -fast/-sensitive/-verysensitive is REQUIRED. Default is -verysensitive: the
    # claim under test is remote-homolog detection, and benchmarking an incumbent below its
    # strongest documented setting is the tell reviewers look for.
    reseek -search ${q_dir} -db ${db_file} -${mode} -columns ${cols} \\
        -threads ${task.cpus} -log search.log -output raw.tsv

    # Accession normalisation lives in bin/normalize_reseek.awk -- see the note there.
    awk -f ${projectDir}/bin/normalize_reseek.awk raw.tsv \\
      | gzip -c > human_vs_${species}.reseek.tsv.gz
    """
}

process prostt5Weights {
    /*
     * ProstT5 model weights, fetched once. Needs outbound internet, same constraint as the
     * structure download, so it is checked rather than left to stall.
     */
    container 'quay.io/biocontainers/foldseek@sha256:c46d6fb854099780597e3adfa48e93c991f4b4d542391c144b9cae4de1ed22f9'
    label 'high_cpu'
    storeDir "${params.outdir}/prostt5"

    output:
    path "weights"

    script:
    """
    set -euo pipefail
    if ! curl --fail --silent --head --max-time 30 https://foldseek.steineggerlab.workers.dev >/dev/null 2>&1; then
        echo "no outbound internet from this node -- cannot fetch ProstT5 weights." >&2
        echo "Download them elsewhere and pass --prostt5_weights <dir>." >&2
        exit 1
    fi
    mkdir -p tmp
    foldseek databases ProstT5 weights tmp
    """
}

process prostt5Search {
    /*
     * Foldseek over 3Di predicted from SEQUENCE on both sides -- no structures anywhere.
     * That is what makes this the differentiating baseline: it runs on every species,
     * including those where AlphaFold coverage is too thin for Foldseek or Reseek.
     */
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/foldseek@sha256:c46d6fb854099780597e3adfa48e93c991f4b4d542391c144b9cae4de1ed22f9'
    cpus   { Math.max(1, (params.prostt5_cpus as int).intdiv(task.attempt)) }
    memory { MemoryUnit.of(params.prostt5_memory) * task.attempt }
    time   { params.prostt5_time }
    // Halve the threads and double the RAM on each retry. Both directions matter: a SLURM
    // OOM here is as often "too many sequences in flight" as "one sequence too big".
    errorStrategy { task.attempt <= 3 ? 'retry' : 'finish' }
    maxRetries 3
    publishDir "${params.outdir}/regions/prostt5", mode: 'copy', pattern: '*.tsv.gz'
    publishDir "${params.outdir}/regions/prostt5", mode: 'copy', pattern: '*_skipped.tsv'

    input:
    tuple val(species), path(species_fasta), path(human_fasta), path(weights)

    output:
    tuple val(species), val("prostt5"), val("3di_from_seq"),
          path("human_vs_${species}.prostt5.tsv.gz"),      emit: regions
    path "human_vs_${species}.prostt5_skipped.tsv",        emit: skipped

    script:
    """
    set -euo pipefail
    mkdir -p tmp

    # Length filter, applied to query and target alike -- see params.prostt5_max_len.
    # Every dropped sequence lands in the skipped table with its length and which side it
    # came from, so the coverage gap is a published file rather than an inference from
    # missing rows.
    printf 'accession\\tside\\tlength\\n' > human_vs_${species}.prostt5_skipped.tsv
    filter_fasta() {
        awk -v maxlen=${params.prostt5_max_len} -v side="\$2" -v skipped="\$3" '
            function flush(   parts, acc) {
                if (hdr == "") return
                if (length(seq) <= maxlen) { print hdr; print seq; return }
                split(substr(hdr, 2), parts, "|")
                acc = (parts[2] != "") ? parts[2] : parts[1]
                print acc "\t" side "\t" length(seq) >> skipped
            }
            /^>/ { flush(); hdr = \$0; seq = ""; next }
            { seq = seq \$0 }
            END { flush() }
        ' "\$1"
    }
    filter_fasta ${human_fasta}   query  human_vs_${species}.prostt5_skipped.tsv > query.fasta
    filter_fasta ${species_fasta} target human_vs_${species}.prostt5_skipped.tsv > target.fasta

    n_skipped=\$(( \$(wc -l < human_vs_${species}.prostt5_skipped.tsv) - 1 ))
    echo "prostt5: dropped \${n_skipped} sequences longer than ${params.prostt5_max_len} aa" >&2

    # A database built this way carries predicted 3Di only, with no Ca coordinates, so
    # TMalign-based alignment types and TM-score/LDDT outputs are unavailable here. The
    # columns below are all sequence-space and unaffected.
    foldseek easy-search \\
        query.fasta target.fasta \\
        out.tsv tmp \\
        --prostt5-model ${weights} \\
        --threads ${task.cpus} \\
        -e ${params.evalue_report} \\
        --max-seqs 1000 \\
        --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"

    awk -F'\t' 'BEGIN{OFS="\t"} {
        for (i = 1; i <= 2; i++) {
            n = split(\$i, p, "|"); if (n >= 2) \$i = p[2]
        }
        print
    }' out.tsv | gzip -c > human_vs_${species}.prostt5.tsv.gz
    """
}

process folddiscoIndex {
    /*
     * Folddisco indexes discontinuous geometric motifs rather than sequences or 3Di
     * strings, so it needs its own index per target proteome. storeDir keeps it: unlike
     * the kmerseek indices this one is reused by every query chunk, so rebuilding it per
     * task would be pure waste.
     */
    tag "${species}"
    container params.folddisco_image
    label 'high_cpu'
    storeDir "${params.outdir}/folddisco_index"

    input:
    tuple val(species), path(structures)

    output:
    tuple val(species), path("${species}_folddisco")

    script:
    """
    set -euo pipefail
    # folddisco decides its input mode by is_dir() on the path it is given
    # (src/cli/workflows/build_index.rs:110). A path that is NOT a directory -- missing, or
    # a symlink that dangles inside the container -- falls through to the Foldcomp branch
    # and panics with "Failed to read Foldcomp DB lookup", which names the wrong cause.
    # These checks report the real state first.
    #
    # Note what this canNOT diagnose: if .command.out is EMPTY, none of this ran, and the
    # problem is upstream of the script entirely -- the container's ENTRYPOINT stopping
    # Nextflow from invoking bash. That is what the original failures were, and the absence
    # of these messages was the evidence, misread at the time as the guard not firing.
    echo "=== staged input as seen inside the container ==="
    ls -ld ${structures} || echo "  ${structures} does not exist"
    if [ -L ${structures} ]; then
        echo "  symlink -> \$(readlink ${structures})"
        [ -e ${structures} ] || echo "  !! DANGLING: target not reachable inside the container"
    fi
    if [ ! -d ${structures} ]; then
        echo "" >&2
        echo "${structures} is not a directory inside the container." >&2
        echo "folddisco will read this as a Foldcomp database and panic with" >&2
        echo "'Failed to read Foldcomp DB lookup', which names the wrong cause." >&2
        echo "Check that ${params.structures}/${species} exists and holds AF-*.cif files:" >&2
        echo "  ls -l ${params.structures}/${species} | head" >&2
        exit 1
    fi

    n=\$(find -L ${structures}/ -name 'AF-*.cif*' 2>/dev/null | wc -l)
    echo "  structures visible: \$n"
    if [ "\$n" -eq 0 ]; then
        echo "${structures} is a directory but holds no AF-*.cif files." >&2
        echo "Run 'make fetch-structures'." >&2
        exit 1
    fi

    mkdir -p ${species}_folddisco
    # -v so a failure says something. Without it folddisco exits 1 silently and the only
    # thing in the log is Nextflow's unrelated "Command 'ps' ... cannot be found" warning.
    folddisco index -v \\
        -p ${structures} \\
        -i ${species}_folddisco/index \\
        -t ${task.cpus}
    """
}

process folddiscoQuery {
    /*
     * One chunk of human query structures against one species index.
     *
     * A failed or empty single query must not take the chunk down with it: Folddisco
     * returns nothing for structures with no matched motif, which is a real result, and
     * AlphaFold coverage is incomplete so some queries have no structure at all.
     */
    tag "human_vs_${species} chunk ${chunk}"
    container params.folddisco_image
    label 'high_cpu'

    input:
    tuple val(species), path(index), path(human_structures), val(chunk)

    output:
    tuple val(species), path("human_vs_${species}.folddisco.chunk${chunk}.tsv")

    script:
    """
    set -euo pipefail
    find -L ${human_structures}/ -name 'AF-*.cif' | sort \\
      | awk 'NR % ${params.folddisco_chunks} == ${chunk}' > chunk.list

    : > regions.tsv
    while read -r f; do
        acc=\$(basename "\$f" | cut -d- -f2)
        folddisco query \\
            -i ${index}/index \\
            -p "\$f" \\
            -t ${task.cpus} \\
            --top ${params.folddisco_top} \\
            > hits.tsv 2>/dev/null || true
        [ -s hits.tsv ] || continue
        folddisco_to_regions.py \\
            --hits hits.tsv \\
            --query-accession "\$acc" \\
            --out regions.tsv >/dev/null
    done < chunk.list

    mv regions.tsv human_vs_${species}.folddisco.chunk${chunk}.tsv
    """
}

process folddiscoMerge {
    tag "${species}"
    label 'python'
    publishDir "${params.outdir}/regions/folddisco", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(chunks)

    output:
    tuple val(species), val("folddisco"), val("motif"),
          path("human_vs_${species}.folddisco.tsv.gz")

    script:
    """
    cat ${chunks} | gzip -c > human_vs_${species}.folddisco.tsv.gz
    """
}

process hmmscanAnnotate {
    /*
     * The reference ceiling, not a competitor: human proteins straight against Pfam-A.
     * No target proteome and no domain transfer, so its coordinates are already
     * query-side domain calls. Everything else is trying to reach this without the
     * Pfam library in hand.
     */
    tag "hmmscan_human"
    container 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/hmmscan", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple path(human_fasta), path(pfam_hmm), path(pfam_hmm_aux)

    output:
    tuple val("hmmscan"), path("human.hmmscan.tsv.gz")

    script:
    // \$4 query protein, \$2 Pfam accession, \$20/\$21 env coords on the query,
    // \$14 domain bitscore, \$13 i-Evalue.
    """
    set -euo pipefail
    hmmscan \\
        --domtblout /dev/stdout \\
        --noali \\
        -E ${params.evalue_report} --domE ${params.evalue_report} \\
        --cpu ${task.cpus} \\
        ${pfam_hmm} ${human_fasta} \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$4 "\\t" \$2 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > human.hmmscan.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// HHblits profile databases
// ---------------------------------------------------------------------------

process hhblitsBuildDB {
    tag "${label}"
    container 'quay.io/biocontainers/hhsuite@sha256:4bf9bb5229de18f522a94f4443c19fdcbb0f0cb0e6ea92f5390aa170bcb0a24f'
    label 'high_cpu'

    input:
    tuple val(label), path(fasta)

    output:
    tuple val(label), path("${label}_hhdb")

    script:
    def n_iter = params.hhblits_db ? "2" : "0"
    """
    set -euo pipefail
    mkdir -p ${label}_hhdb
    ffindex_from_fasta -s ${label}_hhdb/seq.ffdata ${label}_hhdb/seq.ffindex ${fasta}

    if [ -n "${params.hhblits_db ?: ''}" ]; then
        ffindex_apply ${label}_hhdb/seq.ffdata ${label}_hhdb/seq.ffindex \\
            -d ${label}_hhdb/a3m.ffdata -i ${label}_hhdb/a3m.ffindex \\
            -- hhblits -i stdin -d ${params.hhblits_db} -oa3m stdout -n ${n_iter} -cpu 1 -v 0
    else
        ffindex_apply ${label}_hhdb/seq.ffdata ${label}_hhdb/seq.ffindex \\
            -d ${label}_hhdb/a3m.ffdata -i ${label}_hhdb/a3m.ffindex \\
            -- awk '/^>/{print} !/^>/{print toupper(\$0)}'
    fi

    # Target databases also need hhm + cs219; the query database only needs a3m.
    if [ "${label}" != "human" ]; then
        ffindex_apply ${label}_hhdb/a3m.ffdata ${label}_hhdb/a3m.ffindex \\
            -d ${label}_hhdb/hhm.ffdata -i ${label}_hhdb/hhm.ffindex \\
            -- hhmake -i stdin -o stdout -v 0
        cstranslate -f -x 0.3 -c 4 -I a3m -i ${label}_hhdb/a3m -o ${label}_hhdb/cs219
        sort -k3 -n ${label}_hhdb/cs219.ffindex | cut -f1 > ${label}_hhdb/db.sort
    fi
    """
}

// ===========================================================================
// EVALUATION — transfer target domains onto query regions, score against truth
// ===========================================================================

process scoreDomainCalls {
    /*
     * One task per (tool, variant, species). Reads the tool's regions, transfers Pfam
     * labels through the target interval, scores the resulting query-side calls against
     * human_domain_truth.parquet. Emits per-call detail (for downstream re-cutting) and
     * a metrics row.
     */
    tag "${truth_set}: ${tool}/${variant} vs ${species}"
    label 'python'
    publishDir "${params.outdir}/calls",   mode: 'copy', pattern: '*.calls.parquet'
    publishDir "${params.outdir}/metrics", mode: 'copy', pattern: '*.metrics.parquet'
    publishDir "${params.outdir}/curves",  mode: 'copy', pattern: '*.curve.parquet'

    input:
    tuple val(truth_set), val(species), val(tool), val(variant), val(mya), path(regions),
          path(truth), path(domain_map), path(covariates), path(identity)

    output:
    path "${truth_set}.${tool}.${variant}.${species}.calls.parquet",   emit: calls
    path "${truth_set}.${tool}.${variant}.${species}.metrics.parquet", emit: metrics
    path "${truth_set}.${tool}.${variant}.${species}.curve.parquet",   emit: curve

    script:
    // Folddisco reports the envelope of a discontinuous residue set, not an alignment.
    // Scoring that by interval IoU would measure the envelope reduction rather than the
    // prediction, so this arm is scored on coverage instead. See evaluate_domain_calls.py.
    def semantics = tool == 'folddisco' ? 'motif' : 'alignment'
    """
    evaluate_domain_calls.py \\
        --regions      ${regions} \\
        --tool         ${tool} \\
        --interval-semantics ${semantics} \\
        --variant      ${variant} \\
        --species      ${species} \\
        --species-mya  ${mya} \\
        --truth        ${truth} \\
        --domain-map   ${domain_map} \\
        --covariates   ${covariates} \\
        --identity     ${identity} \\
        --min-overlap  ${params.min_overlap} \\
        --strict-iou   ${params.strict_iou} \\
        --truth-set    ${truth_set} \\
        --calls-out    ${truth_set}.${tool}.${variant}.${species}.calls.parquet \\
        --metrics-out  ${truth_set}.${tool}.${variant}.${species}.metrics.parquet \\
        --curve-out    ${truth_set}.${tool}.${variant}.${species}.curve.parquet
    """
}

process scoreHmmscanCeiling {
    tag "hmmscan ceiling"
    label 'python'
    publishDir "${params.outdir}/calls",   mode: 'copy', pattern: '*.calls.parquet'
    publishDir "${params.outdir}/metrics", mode: 'copy', pattern: '*.metrics.parquet'
    publishDir "${params.outdir}/curves",  mode: 'copy', pattern: '*.curve.parquet'

    input:
    tuple val(tool), path(regions), path(truth), path(covariates)

    output:
    path "hmmscan.direct.all.calls.parquet",   emit: calls
    path "hmmscan.direct.all.metrics.parquet", emit: metrics
    path "hmmscan.direct.all.curve.parquet",   emit: curve

    script:
    """
    evaluate_domain_calls.py \\
        --regions      ${regions} \\
        --tool         hmmscan \\
        --variant      direct \\
        --species      all \\
        --truth        ${truth} \\
        --direct-annotation \\
        --covariates   ${covariates} \\
        --identity     ${identity} \\
        --min-overlap  ${params.min_overlap} \\
        --strict-iou   ${params.strict_iou} \\
        --calls-out    hmmscan.direct.all.calls.parquet \\
        --metrics-out  hmmscan.direct.all.metrics.parquet \\
        --curve-out    hmmscan.direct.all.curve.parquet
    """
}

process aggregateMetrics {
    label 'python'
    publishDir params.outdir, mode: 'copy'

    input:
    path 'metrics/*'
    path 'curves/*'

    output:
    path "all_domain_metrics.parquet"
    path "all_domain_metrics.csv"
    path "all_domain_curves.parquet"

    script:
    """
    aggregate_domain_metrics.py \\
        metrics curves \\
        all_domain_metrics.parquet all_domain_metrics.csv all_domain_curves.parquet
    """
}

// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")
    def annotations = file(params.annotations)

    // Structure download as SLURM jobs. A standalone entry: it does the download and
    // nothing else, because the download must finish before the structure arms can run
    // and there is no point holding a whole pipeline open while ~60 GB transfers.
    if (params.fetch_structures) {
        def acc_dir = file("${params.structures}/_accessions")
        if (!acc_dir.exists()) {
            error "no accession lists at ${acc_dir} -- run `make structure-lists` first"
        }
        fetch_in = Channel.fromList(
            (SPECIES*.label + ['human']).collect { tuple(it, acc_dir) }
        )
        fetchStructures(fetch_in).subscribe { sp, log_file ->
            log.info "structures fetched: ${sp}"
        }
        return
    }

    // (label, fasta) for the 9 target species
    species_ch = Channel.fromList(
        SPECIES.collect { s ->
            tuple(s.label, file("${params.qfo_dir}/${s.subdir}/${s.proteome}_${s.taxon}.fasta"))
        }
    )

    // ---- ground truth + query covariates ----
    truth_out = buildDomainTruth(annotations)

    // Optional inputs are passed as sentinel files rather than nulls: Nextflow cannot
    // stage a null path, and a sentinel keeps the process signature fixed whether or not
    // the file exists.
    def optional_or = { pathStr, sentinel ->
        pathStr && file(pathStr).exists() ? file(pathStr) : file("${projectDir}/assets/${sentinel}")
    }
    cov_in = truth_out.truth.map { t ->
        tuple(t,
              optional_or(params.hgnc_file,  'NO_HGNC'),
              optional_or(params.omega_file, 'NO_OMEGA'),
              file("${params.structures}/human").exists()
                  ? file("${params.structures}/human")
                  : file("${projectDir}/assets/NO_STRUCTURES"))
    }
    covariates = buildQueryCovariates(cov_in).covariates

    // One entry per truth set: (label, truth_parquet, species->map channel). Scoring runs
    // once per set, so every metric row says which truth it was measured against.
    def map_of = { maps -> maps.flatten()
        .map { m -> tuple(m.name.replaceAll(/_domain_map\.parquet$/, ''), m) } }

    truth_sets = Channel.of(tuple("pfam", 1))
        .combine(truth_out.truth)
        .map { label, _i, t -> tuple(label, t) }
    map_ch = map_of(truth_out.maps).map { sp, m -> tuple("pfam", sp, m) }

    // Pre-built truth sets, added when their directory is present. Each contributes a
    // human_*_truth.parquet plus per-species *_domain_map.parquet, the same shape as the
    // Pfam arm, so scoring runs against them unchanged.
    def add_prebuilt = { label, dir ->
        if (!dir) return
        def d = file(dir)
        if (!d.exists()) {
            log.warn "--${label}_dir points at ${dir}, which does not exist -- skipping " +
                     "that truth arm. Build it with `make ${label}-truth`."
            return
        }
        def t = file("${dir}/human_${label}_truth.parquet")
        if (!t.exists()) {
            log.warn "no human_${label}_truth.parquet under ${dir} -- skipping"
            return
        }
        truth_sets = truth_sets.mix(Channel.of(tuple(label, t)))
        map_ch = map_ch.mix(
            Channel.fromList(
                d.listFiles().findAll { it.name.endsWith('_domain_map.parquet') }
                 .collect { tuple(label, it.name.replaceAll(/_domain_map\.parquet$/, ''), it) }
            )
        )
    }

    if (params.swissprot_dat && file(params.swissprot_dat).exists()) {
        sprot = buildSwissprotTruth(
            Channel.of(tuple(file(params.swissprot_dat), annotations))
        )
        truth_sets = truth_sets.mix(
            Channel.of(tuple("swissprot", 1)).combine(sprot.truth)
                .map { label, _i, t -> tuple(label, t) }
        )
        map_ch = map_ch.mix(map_of(sprot.maps).map { sp, m -> tuple("swissprot", sp, m) })
    } else {
        log.warn "swissprot_dat not found (${params.swissprot_dat}) -- running without the " +
                 "Swiss-Prot truth arm. Pfam is circular with the profile baselines; see README."
    }

    add_prebuilt("pfamn", params.pfamn_dir)
    add_prebuilt("mcsa", params.mcsa_dir)

    // ---- twilight-zone axis: per-domain-pair percent identity ----
    // Off by default only if explicitly skipped; it is the axis claim 1 is stated on.
    identity_ch = Channel.empty()
    if (!params.skip_identity) {
        // One invocation, mixed inputs. A DSL2 process cannot be called twice in the same
        // workflow, so query and target extraction share a call and are split afterwards
        // on the label.
        dom_in = truth_out.truth
            .map { t -> tuple("human", t, human_fasta) }
            .mix(
                // Restrict to the species this run targets. map_ch carries a map
                // for every species in the annotations directory, a superset of SPECIES
                // whenever --target_species narrows the run -- looking one of those up
                // returned null and died on `.subdir`.
                map_ch.filter { ts, sp, _m -> ts == "pfam" && SPECIES*.label.contains(sp) }
                      .map { _ts, sp, m ->
                          def info = SPECIES.find { it.label == sp }
                          tuple(sp, m,
                                file("${params.qfo_dir}/${info.subdir}/${info.proteome}_${info.taxon}.fasta"))
                      }
            )
        dom_fa     = extractDomainSequences(dom_in)
        human_dom  = dom_fa.filter { label, _fa -> label == "human" }.map { _l, fa -> fa }
        target_dom = dom_fa.filter { label, _fa -> label != "human" }

        identity_ch = parseIdentity(
            domainIdentity(
                target_dom.combine(human_dom)
                          .map { sp, tfa, hfa -> tuple(sp, hfa, tfa) }
            )
        )
    }



    // ---- kmerseek: alphabet x ksize x species ----
    kmerseek_regions = Channel.empty()
    if (!params.skip_kmerseek) {
        // An explicit combo list overrides the matrix entirely. The label is derived from
        // the CLI flag the same way ALL_ENCODINGS does it, so output filenames and the
        // per-combo memory sizing keep working unchanged.
        def combos = params.kmerseek_combos
            ? params.kmerseek_combos.tokenize(',').collect { spec ->
                  def (enc, k) = spec.trim().split(':')
                  def known = ALL_ENCODINGS.find { it[0] == enc }
                  if (!known) {
                      error "Unknown encoding '${enc}' in --kmerseek_combos. Known: ${ALL_ENCODINGS*.get(0).join(', ')}"
                  }
                  tuple(enc, known[1], k.toInteger())
              }
            : ALL_ENCODINGS.collectMany { cli_flag, label, kmin, kmax ->
                  (kmin..kmax).collect { k -> tuple(cli_flag, label, k) }
              }

        // Cross every alphabet x ksize with the low-complexity toggle. This is what
        // doubles the sweep, and it is the point: whether dropping low-complexity k-mers
        // helps is alphabet-dependent, so it has to be measured rather than chosen.
        combos = combos.collectMany { cli_flag, label, k ->
            params.low_complexity_toggle.collect { lc -> tuple(cli_flag, label, k, lc) }
        }
        // Spell out the query/target asymmetry at startup. "2 species" reading as
        // "yeast and ecoli, so where does human_vs_ecoli come from" is a real confusion
        // this line exists to prevent.
        log.info """
        |  query   : human (UP000005640_9606) -- always, and never listed as a target
        |  targets : ${SPECIES*.label.join(', ')}
        |  combos  : ${combos.size()} (alphabet x ksize x low-complexity on/off)
        |  searches: ${combos.size()} x ${SPECIES.size()} targets = ${combos.size() * SPECIES.size()}
        |            each named human_vs_<target>, e.g. human_vs_${SPECIES[0].label}
        |  spectra : one k-mer frequency spectrum per combo, published for plotting
        """.stripMargin()

        kmerseek_in = species_ch.combine(Channel.fromList(combos))
            .map { species, fasta, cli_flag, label, ksize, lowcomp ->
                tuple(species, fasta, cli_flag, label, ksize, lowcomp, human_fasta)
            }
        // Rebuild (species, tool, variant) from the filename. The process emits a bare
        // path so storeDir works; the name carries everything the tuple used to.
        ks_out = kmerseekIndexAndSearch(kmerseek_in)
        // Rebuild (species, tool, variant) from the filename; the process emits bare paths
        // so storeDir works. The variant now carries the low-complexity setting, so the two
        // arms of the toggle are separate rows everywhere downstream rather than pooled.
        kmerseek_regions = ks_out.regions
            .map { pq ->
                def m = (pq.name =~ /^human_vs_(.+?)\.(.+)\.k(\d+)\.lc(true|false)\.regions\.parquet$/)
                if (!m) error "cannot parse kmerseek result filename: ${pq.name}"
                def lc = m[0][4] == 'true' ? 'lcTrue' : 'lcFalse'
                tuple(m[0][1], "kmerseek", "${m[0][2]}_k${m[0][3]}_${lc}", pq)
            }
        // Spectra are published for plotting and are not scored.
        ks_out.spectrum.collectFile(
            storeDir: "${params.outdir}/spectra", keepHeader: false
        )
    }

    // ---- baselines ----
    baseline_regions = Channel.empty()
    if (!params.skip_baselines) {
        pair_ch = species_ch.map { species, fasta -> tuple(species, fasta, human_fasta) }

        phmmer_out    = phmmerSearch(pair_ch)
        jackhmmer_out = jackhmmerSearch(pair_ch)

        mmseqs_in = species_ch.flatMap { species, fasta ->
            [
                tuple(species, "mmseqs2_seqseq",    1,                          fasta, human_fasta),
                tuple(species, "mmseqs2_iterative", params.mmseqs2_iterations,  fasta, human_fasta),
            ]
        }
        mmseqs_out = mmseqs2Search(mmseqs_in)

        // HHblits: build the human query profile DB once, every species target DB once.
        all_for_hhdb = species_ch.mix(Channel.of(tuple("human", human_fasta)))
        hhdb_ch      = hhblitsBuildDB(all_for_hhdb)
        human_hhdb   = hhdb_ch.filter { label, _db -> label == "human" }.map { _label, db -> db }
        species_hhdb = hhdb_ch.filter { label, _db -> label != "human" }

        hhblits_out = hhblitsSearch(species_hhdb.combine(human_hhdb))

        baseline_regions = phmmer_out.mix(jackhmmer_out).mix(mmseqs_out).mix(hhblits_out)

        // ---- ProstT5: 3Di predicted from sequence, no structures on either side ----
        // Deliberately OUTSIDE the structure guard below. This is the arm that can run
        // where AlphaFold coverage is too thin for Foldseek or Reseek, which is         // the regime the invertebrate claim lives in.
        if (!params.skip_prostt5) {
            // Sherlock compute nodes have no outbound internet, so the download process
            // cannot work there -- its preflight fails in 30 seconds by design. When a
            // weights path is given it must exist; falling back to the download
            // would just re-fail with a message about the wrong thing.
            def w = params.prostt5_weights ? file(params.prostt5_weights) : null
            if (params.prostt5_weights && !w.exists()) {
                error """
                |--prostt5_weights points at ${params.prostt5_weights}, which does not exist.
                |Compute nodes here have no outbound internet, so the pipeline cannot fetch it.
                |Download it once on a login node:
                |    make prostt5-weights
                |or pass --skip_prostt5 true to leave that arm out.
                """.stripMargin()
            }
            weights = w ? Channel.value(w) : prostt5Weights()
            prostt5_out = prostt5Search(
                species_ch.map { sp, fa -> tuple(sp, fa, human_fasta) }.combine(weights)
            )
            baseline_regions = baseline_regions.mix(prostt5_out.regions)
        }

        // ---- foldseek ----
        // Foldseek and Folddisco need actual structure files. An empty or missing
        // directory makes both fail deep inside a container with no usable message --
        // folddisco index exits 1 printing nothing at all -- so the emptiness is checked
        // here, where it can name the cause and the fix. `make sync-data` deliberately
        // does NOT ship structures (they are ~36 GB); `make sync-structures` does.
        def has_structs = { label ->
            def d = file("${params.structures}/${label}")
            d.exists() && d.list().any { it ==~ /(?i)^AF-.*\.cif(\.gz)?$/ }
        }
        def struct_species = SPECIES.findAll { has_structs(it.label) }
        def missing_structs = SPECIES*.label - struct_species*.label
        def human_ok = has_structs('human')

        if (!params.skip_foldseek && (!human_ok || struct_species.isEmpty())) {
            log.warn """
            |Skipping the Foldseek and Folddisco arms: no AlphaFold structures found under
            |  ${params.structures}
            |  human structures present: ${human_ok}
            |  targets with structures : ${struct_species*.label ?: 'none'}
            |Both arms need query AND target structures. Populate them with
            |  make fetch-structures   (on the Mac, ~36 GB)
            |  make sync-structures    (Mac -> cluster)
            |or pass --skip_foldseek true to silence this. Every sequence-based arm runs
            |regardless; only the two structure arms are affected.
            """.stripMargin()
        }

        if (!params.skip_foldseek && human_ok && !struct_species.isEmpty()) {
            if (missing_structs) {
                log.warn "Structure arms skip ${missing_structs.join(', ')}: no .cif files staged for them"
            }
            def human_structs = file("${params.structures}/human")
            struct_ch = Channel.fromList(
                struct_species.collect { s -> tuple(s.label, file("${params.structures}/${s.label}")) }
            ).map { label, dir -> tuple(label, dir, human_structs) }

            foldseek_out     = foldseekSearch(struct_ch)
            baseline_regions = baseline_regions.mix(foldseek_out)

            // ---- Reseek: same structures, opposite alphabet direction ----
            if (!params.skip_reseek) {
                reseek_db  = reseekConvert(
                    Channel.fromList(
                        struct_species.collect { sp -> tuple(sp.label, file("${params.structures}/${sp.label}")) }
                    )
                )
                reseek_out = reseekSearch(
                    reseek_db.combine(Channel.of(human_structs))
                )
                baseline_regions = baseline_regions.mix(reseek_out)
            }

            // ---- folddisco ----
            if (!params.skip_folddisco) {
                fd_index = folddiscoIndex(
                    Channel.fromList(
                        struct_species.collect { s -> tuple(s.label, file("${params.structures}/${s.label}")) }
                    )
                )
                fd_chunks = Channel.of(0..(params.folddisco_chunks - 1)).flatten()
                fd_out = folddiscoQuery(
                    fd_index.combine(Channel.of(human_structs)).combine(fd_chunks)
                )
                folddisco_regions = folddiscoMerge(fd_out.groupTuple(by: 0))
                baseline_regions  = baseline_regions.mix(folddisco_regions)
            }
        }
    }

    // ---- score every arm ----
    all_regions = kmerseek_regions.mix(baseline_regions)

    // join on species to attach that species' domain-transfer map, plus the shared truth
    // Divergence axis for this benchmark. The 200-series used human-mouse percent identity
    // because it had one target species; here the species IS the divergence axis, so its
    // age travels with every metric row and notebooks can plot against it directly.
    def MYA = SPECIES.collectEntries { [(it.label): it.mya] }

    // Cross every result with every truth set, joining on (truth_set, species) so a run
    // never scores Pfam regions against a Swiss-Prot map or vice versa.
    score_in = all_regions
        .map { species, tool, variant, regions ->
            tuple(species, tool, variant, MYA[species] ?: 0, regions)
        }
        .combine(map_ch.map { ts, sp, m -> tuple(sp, ts, m) }, by: 0)
        .map { species, tool, variant, mya, regions, ts, domain_map ->
            tuple(ts, species, tool, variant, mya, regions, domain_map)
        }
        .combine(truth_sets, by: 0)
        .combine(covariates)
        .map { ts, species, tool, variant, mya, regions, domain_map, truth, cov ->
            tuple(species, ts, tool, variant, mya, regions, truth, domain_map, cov)
        }
        // Identity is per (species, domain instance), so it joins on species. An empty
        // sentinel keeps the process signature fixed when --skip_identity is set.
        .combine(
            params.skip_identity
                ? Channel.fromList(SPECIES.collect { tuple(it.label, file("${projectDir}/assets/NO_IDENTITY")) })
                : identity_ch,
            by: 0
        )
        .map { species, ts, tool, variant, mya, regions, truth, domain_map, cov, ident ->
            tuple(ts, species, tool, variant, mya, regions, truth, domain_map, cov, ident)
        }

    scored = scoreDomainCalls(score_in)

    // ---- hmmscan annotation ceiling ----
    ceiling_metrics = Channel.empty()
    ceiling_curves  = Channel.empty()
    if (params.run_hmmscan) {
        def pfam_hmm = file(params.pfam_hmm)
        // hmmpress writes Pfam-A.hmm.{h3m,h3i,h3f,h3p} alongside the .hmm; hmmscan needs them.
        def pfam_aux = file("${params.pfam_hmm}.h3*")
        hmmscan_out = hmmscanAnnotate(Channel.of(tuple(human_fasta, pfam_hmm, pfam_aux)))
        // The ceiling is scored against Pfam only: hmmscan IS the Pfam annotation
        // procedure, so a Swiss-Prot row for it would compare two unrelated label spaces.
        ceiling     = scoreHmmscanCeiling(
            hmmscan_out.combine(truth_out.truth).combine(covariates)
        )
        ceiling_metrics = ceiling.metrics
        ceiling_curves  = ceiling.curve
    }

    aggregateMetrics(
        scored.metrics.mix(ceiling_metrics).collect(),
        scored.curve.mix(ceiling_curves).collect(),
    )
}
