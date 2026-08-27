#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * Containers are pinned by DIGEST, not tag. The foldseek pin previously used a bioconda
 * tag that was retagged upstream, and the run died on Sherlock with "manifest unknown"
 * -- the standard symptom. Digests are immutable, so a reviewer re-running this in six
 * months gets the same image. Human-readable versions, for Methods:
 *   foldseek 10.941cd33  hmmer 3.4   mmseqs2 18.8cc5c   hhsuite 3.3.0
 *
 * foldseek moved 9 -> 10 for one reason: 9.427df8a has no --gpu flag at all, so the
 * ProstT5 arm could only ever run on CPU. Both foldseek arms move together rather than
 * pinning two versions, so foldseekSearch results from before this change are not
 * comparable to results after it -- rerun that arm rather than mixing them.
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
// Sequence-based disorder, on by default: it costs one short CPU task over the query FASTA
// and gives an axis that does not share pLDDT's MSA-depth confound.
params.skip_metapredict = false
params.metapredict_threshold = null   // null means metapredict's own default

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
    // k=4 dropped: 20^4 = 160_000 keys against ~11.3M proteome k-mers means the entire
    // keyspace is occupied ~70 times over, so every query 4-mer matches a large share of
    // the proteome. It OOM-killed its task and the result would be noise either way.
    ['protein20', 'protein20', 5, 13],                      // 4.176 bits/sym, 9 ksizes
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

// The LABEL, not the CLI flag, is what goes into the kmerseekIndex storeDir entry name.
// Two rows sharing a label would therefore name one store entry from two different
// encodings: both tasks build, and the second one's unstage dies with
//   mv: cannot move '<entry>' to '<store>/./<entry>': Directory not empty
// because a directory cannot be moved onto a non-empty directory of the same name.
// Nextflow decides "already in the store" once, when it CREATES the task, and never looks
// again -- not before the move, and not on a retry -- so nothing downstream catches this.
// Checked here rather than trusted, since the two columns are identical today and the
// coupling between them is invisible at the point where a new alphabet gets added.
def dupEncodingLabels = ALL_ENCODINGS*.get(1).countBy { it }.findAll { _l, n -> n > 1 }*.key
if (dupEncodingLabels) {
    error "ALL_ENCODINGS has repeated labels: ${dupEncodingLabels.join(', ')}. " +
          "Labels name the kmerseek index store entries and have to be unique."
}

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

// --- MultiQC report --------------------------------------------------------
// One HTML for the whole run: accuracy, the alphabet x ksize sweep, and the trace's
// time/CPU/memory, in the sections built by bin/build_multiqc_inputs.py.
//
// The frontier and the curve sections need ONE truth set, since a number averaged across
// Pfam and Swiss-Prot has no interpretation. Default is whichever of them is present,
// preferring Swiss-Prot because Pfam is circular with the profile baselines.
params.skip_multiqc = false
params.multiqc_primary_truth = null
params.multiqc_config = "${projectDir}/assets/multiqc_config.yaml"
// Rows per grouped plot and curves per PR/ROC plot, ranked by Fmax. The sweep is 113
// alphabet x ksize combos x 2 low-complexity arms; plotting all of them against ten
// baselines would bury the baselines, so each baseline contributes its one variant and
// the sweep contributes its best few. Several rather than one on purpose: whether the
// winning combo is a lone spike or the top of a plateau of near-identical combos changes
// what the result means, and one point cannot show it.
params.multiqc_max_tools    = 20
params.multiqc_max_lines    = 12
params.multiqc_top_kmerseek = 5

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
// Where the cached, target-side databases live. Defaults to outdir, but a run whose QUERY
// set differs while its TARGETS are identical -- the chr6 midi run against full proteomes --
// wants its own results directory and the SAME databases. Those databases are built only
// from the target proteome, so they are genuinely shared: pointing db_cache at the full
// run's outdir means the midi run builds each ProstT5 / foldseek / mmseqs / reseek /
// folddisco database once and the full run reuses it, instead of paying for ProstT5
// inference over nine proteomes twice.
params.db_cache = null

params.prostt5_max_len = 6000

// ProstT5 on a GPU is one to two orders of magnitude faster than on CPU. Requires
// foldseek >= 10 (9.427df8a has no --gpu flag), a CUDA runtime visible in the
// container (--nv) and a GPU allocation from the scheduler; the sherlock profile sets
// the latter two. Set false to fall back to CPU without touching anything else.
params.prostt5_gpu = true

// GPU for the SEARCH stages is a separate question from ProstT5, and it exists so the
// paper can say every baseline was run at its best available setting. A reviewer reading
// "kmerseek is fast without structure" will ask whether the structure and profile
// baselines were given a GPU; without this flag the honest answer was no.
//
// Both pinned binaries do support it. foldseek 10.941cd33 and mmseqs2 18.8cc5c each carry
// `--gpu INT` on `search` and an undocumented-in-top-level-help `makepaddedseqdb`
// subcommand. The GPU prefilter refuses any target database that is not in the padded
// layout -- verified against both pinned images, which answer
//   Database <db> is not a valid GPU database / Please call: makepaddedseqdb <db> <db>_pad
// and exit before touching CUDA. mmseqsPaddedDb and foldseekPaddedDb build that layout;
// they run only when a GPU arm is actually requested, so a CPU run pays nothing.
//
// Only the TARGET side is padded. The query database is read unpadded in both tools,
// confirmed the same way. foldseek's own makepaddedseqdb handles the _ss (3Di) and _ca
// (coordinate) sub-databases as well as the amino-acid one, so one call covers a
// structure database.
//
// What this does NOT do is run the same algorithm faster. `--gpu 1` REPLACES the k-mer
// prefilter with an exhaustive ungapped-alignment prefilter in both tools: mmseqs drops
// from `prefilter -s 7` to `ungappedprefilter --prefilter-mode 1`, foldseek from
// `prefilter -s 9.5` to `ungappedprefilter`. So -s has no effect in GPU mode and the hit
// lists differ from the CPU arm's. That is why the GPU arms carry their own `variant`
// label and their own published region files rather than overwriting the CPU ones: they
// are a second measurement, not a cheaper copy of the first.
params.gpu_search = false

// Run both modes side by side in ONE invocation, restricted to the two arms that have a
// GPU path. Same databases, same query set, same scheduler session, one trace -- which is
// what makes the per-task comparison a comparison rather than two runs on two days.
// bin/gpu_search_benchmark.py turns the resulting trace into queries/s per tool per mode.
params.gpu_benchmark = false

// Deliberately NOT the `high_cpu` label. Threads are the multiplier on ProstT5's peak
// memory, so this arm buys RAM rather than cores; `high_cpu`'s 16 threads on 64 GB is
// exactly the shape that died. Set in the process body from params, because a bare
// `process { cpus = ... }` in a profile does not override a body directive while a
// `withLabel:`/`withName:` selector does.
// 8, not 16. ProstT5 inference scales with threads, but so does peak memory: at the
// 6000 aa cap a worst case of 8 concurrent long sequences is ~37 GB of activations plus
// the model, which fits 64 GB. 16 threads is the shape that OOM-killed the first run.
// Raising this further means lowering prostt5_max_len or raising prostt5_memory too.
params.prostt5_cpus   = 8
// Do NOT scale this down for a smoke test. ProstT5 is the same 3B-parameter model
// whatever the input size, so the floor here is the model plus one sequence's
// activations, not the number of sequences. A 16 GB `mini` override OOM-killed a
// 194-sequence run just as dead as the uncapped full one.
params.prostt5_memory = '64 GB'
params.prostt5_time   = '48h'

// folddisco, rebuilt with the ENTRYPOINT cleared. The upstream image sets
// ENTRYPOINT ["/usr/local/bin/folddisco"], which under Apptainer becomes the SIF runscript
// and stops Nextflow from running `/bin/bash .command.run` -- the task exits 1 with an
// entirely empty .command.out. Docker bypasses the entrypoint the way Nextflow invokes it,
// so this reproduces only on the cluster. See Dockerfile.folddisco.
params.folddisco_image = 'docker.io/olgabot/folddisco:2026-08-20-noentrypoint'
// Its own image: metapredict needs a compiler to build and drags in torch. See
// Dockerfile.metapredict for why that does not belong in the shared kmerseek image.
params.metapredict_image = 'docker.io/olgabot/metapredict:2026-08-27'

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
// What blows up memory is a SATURATED k-mer keyspace, not a particular alphabet family.
// When classes^ksize is small relative to the number of k-mers in the proteome, every key
// is occupied many times over, each query k-mer matches a large share of the targets, and
// both the inverted index and the region output explode.
//
// The old rule keyed on the alphabet NAME (hp_*2/hp_*3 at ksize<=20). It flagged 40 combos
// and missed protein20 k=4: 20^4 = 160_000 keys against ~11.3M proteome k-mers, the whole
// keyspace occupied ~70 times over. That combo OOM-killed a task -- and the model predicts
// 70.6 k-mers per key where the task's own index log reported mean 70.50. The rule below
// flags 47 of the 184 alphabet x ksize combos, including every low-ksize case in the
// non-HP alphabets that a name-based rule cannot see.
def DB_CACHE = params.db_cache ?: params.outdir

// Three runs have now died on the same unstage failure, on three different directory
// outputs under storeDir (foldseekDb, prostt5Db, kmerseekIndex), and each time it had to
// be diagnosed from scratch because the only thing Nextflow prints is a bare `mv` error.
// The message below is the diagnosis, attached to the failure that produces it.
//
// What it cannot do is prevent the failure. The move runs host-side in .command.run,
// outside the container, after the task's own script has already succeeded, so no guard in
// a `script:` block sits in front of it. See the note on kmerseekIndex's errorStrategy for
// why retrying it does not work either.
workflow.onError {
    if (workflow.errorReport?.contains('Directory not empty')) {
        log.error """
        |
        |Two writers built the same storeDir entry. Nextflow reads the store when it CREATES
        |a task and never re-checks -- not at dispatch, not on a retry -- so both tasks did
        |the full build, and the loser's `mv` failed because a directory will not move onto
        |a non-empty directory of the same name. Only DIRECTORY outputs can fail this way;
        |file outputs overwrite silently.
        |
        |The winner's entry in the store is COMPLETE. The move is a same-filesystem rename,
        |so it either happened whole or not at all, and the run that reported this error is
        |the one that lost. Nothing needs repairing.
        |
        |Where the second writer comes from, in order of likelihood:
        |  1. Tasks left over from a previous run. Killing the Nextflow driver does not
        |     cancel jobs it already submitted. Check with `squeue -u \$USER` and wait for
        |     it to drain -- the terminal returning is not the same as the queue emptying.
        |  2. Another pipeline running now against the same --db_cache (${DB_CACHE}).
        |  3. Two tasks inside THIS run naming one entry. The combo guard in the kmerseek
        |     block rejects the way that used to happen; if it fires anyway, that is a new
        |     path and worth tracking down rather than working around.
        |
        |To continue: confirm the queue is empty, then re-run with -resume. The entry now
        |exists, so the task becomes a store hit and costs nothing.
        """.stripMargin()
    }
}

// Shell helpers both kmerseek processes paste into their scripts to time themselves.
//
// The kmerseek arm is the only one on storeDir rather than publishDir, and a storeDir hit
// executes no task, so Nextflow writes no trace row for it. Every resource figure in the
// MultiQC report is joined off that trace, which is why kmerseek's throughput and
// CPU-hours came out blank while foldseek and prostt5 were populated. Splitting
// kmerseekIndexAndSearch into kmerseekIndex + kmerseekSearch made it structural: the 3294
// target indexes now live in a shared DB_CACHE, so the hit is the normal case. Timing
// inside the task and storing the record next to the result is what survives that.
//
// Single-quoted so Groovy leaves the shell's own $ alone.
def KMERSEEK_TIMER_SH = '''# GNU date's %N is nanoseconds. BSD date (macOS, when this runs without a container)
    # prints a literal N instead, so fall back to whole seconds rather than emit a number
    # that is silently wrong by six orders of magnitude.
    _now_ms() {
        local t
        t=$(date +%s%N)
        case "$t" in
            *N) echo $(( $(date +%s) * 1000 )) ;;
            *)  echo $(( t / 1000000 )) ;;
        esac
    }
    _elapsed_s() { awk -v a="$1" -v b="$(_now_ms)" 'BEGIN{ printf "%.3f", (b - a) / 1000 }'; }'''

// Class count is the trailing number in every encoding name: protein20, gbmr4,
// hp_lehninger_hpc3, hp_thomas_dill_no_c2.
def alphabetClasses = { label ->
    def m = label =~ /(\d+)$/
    m ? (m[0][1] as int) : 20
}

// Roughly the k-mer count of one full proteome; mouse reports 11_284_322 at protein20 k=4.
// A combo whose keyspace is at or below this has every key occupied on average, which is
// the regime that needs the large allocation.
params.kmerseek_saturation_kmers = 11_300_000

def isSaturated = { label, ksize ->
    ksize * Math.log10(alphabetClasses(label)) <= Math.log10(params.kmerseek_saturation_kmers as double)
}

// Sized for full QfO proteomes. A mini/smoke run indexes a few hundred sequences and
// needs nothing like this, so both figures are params -- the `mini` profile lowers them
// rather than forcing a 128 GB request for a 300-protein test set.
params.kmerseek_memory         = '32 GB'
params.kmerseek_memory_hp_lowk = '128 GB'

// scoreDomainCalls is ~10_179 tasks on a full run, and a flat 96 GB for all of them is a
// 5.8 TB standing ask at maxForks 60 -- it queues rather than runs. Size from the actual
// input instead: the region file IS the thing that gets loaded, joined against the domain
// map and expanded, so its compressed size predicts the peak far better than the combo
// name does. HP at low ksize produces the big files and still gets the big allocation;
// the median combo stops asking for 96 GB it never touches.
//
// The multiplier is compressed-bytes to peak RAM, and it is deliberately generous: the
// join in transfer_domains multiplies rows before it filters them. Retries double it, so
// an underestimate costs one requeue rather than a dead task.
params.score_memory_base   = '8 GB'
params.score_memory_per_mb = 120     // MB of RAM per compressed MB of regions
params.score_memory_max    = '96 GB' // the old flat value, now a ceiling rather than a floor

def scoreMemory = { regions, attempt ->
    long mb    = Math.max(1L, (regions.size() as long).intdiv(1024L * 1024L))
    long estMb = MemoryUnit.of(params.score_memory_base).toMega() + mb * (params.score_memory_per_mb as long)
    long capMb = MemoryUnit.of(params.score_memory_max).toMega()
    MemoryUnit.of("${Math.min(estMb, capMb)} MB") * attempt
}

// Target proteome size, in MB of FASTA, that kmerseek_memory_hp_lowk is sized for.
// Zebrafish is the largest QfO proteome at 16.7 MB. Ecoli is 1.8 MB.
params.kmerseek_reference_proteome_mb = 17
// Nothing drops below this however small the proteome: RocksDB write buffers and the
// zstd output stream cost the same regardless of how little goes through them.
params.kmerseek_memory_floor = '8 GB'

// The SATURATED allocation, and only that one, scales with the target proteome. The index
// holds one posting list per proteome k-mer, so a 1.8 MB proteome cannot build the index a
// 16.7 MB one does, and sizing on saturation alone asked 128 GB for every ecoli and yeast
// task in the sweep against measured peaks of 1.0-4.1 GB. ecoli_hp_k20 reserved 128 GB and
// touched 1.02 GB. Scaling those cuts the sweep's standing reservation by 26% with no task
// dropping below the headroom it already had.
//
// The UNSATURATED 32 GB branch stays flat, and that is not an oversight. isSaturated is a
// hard threshold, so the hungriest tasks in the whole sweep are the ones sitting just
// OUTSIDE it: fly_dayhoff_k10 is unsaturated by the rule and peaked at 28.1 GB, which is
// 1.14x of its 32 GB. There is no room to scale that branch down -- doing so by proteome
// size was measured to put fly_dayhoff_k10 and k11 into OOM. Scale it only with new peak
// data in hand, and raise the base first if you do.
// Wall clock for a batched scoring task, from the TOTAL bytes it will read. One task now
// scores every arm for a species sequentially, so time is additive where memory is not.
//
// The rate is deliberately pessimistic. scoreDomainCalls timings measured so far are all
// yeast on protein/dayhoff, reading 7.9-8.3 MB with no spread, so they cannot calibrate
// anything -- a regression on them gives r2 = 0.00. Until a batched run produces real
// numbers, this is a bound rather than a fit, and the floor matters more than the slope.
params.score_time_base_min    = 30
params.score_time_per_mb_sec  = 6
params.score_time_max_hours   = 24

def scoreTime = { regions, attempt ->
    def files = regions instanceof List ? regions : [regions]
    long mb   = Math.max(1L, files.sum { (it.size() as long) }.intdiv(1024L * 1024L))
    long mins = (params.score_time_base_min as long)
                + (mb * (params.score_time_per_mb_sec as long)).intdiv(60L)
    long cap  = (params.score_time_max_hours as long) * 60L
    Duration.of("${Math.min(mins, cap) * attempt} min")
}

def kmerseekMemory = { label, ksize, targetBytes, attempt ->
    if (!isSaturated(label, ksize)) {
        return MemoryUnit.of(params.kmerseek_memory) * attempt
    }
    long   mb   = Math.max(1L, (targetBytes as long).intdiv(1024L * 1024L))
    double frac = Math.min(1.0d, mb / (params.kmerseek_reference_proteome_mb as double))
    long sized  = Math.max(MemoryUnit.of(params.kmerseek_memory_floor).toMega(),
                           (long) (MemoryUnit.of(params.kmerseek_memory_hp_lowk).toMega() * frac))
    MemoryUnit.of("${sized} MB") * attempt
}

// The trace file the MultiQC resource sections read. Resolved through a closure, never
// at parse time: `trace.file` is created by Nextflow's observer as the run starts, so a
// `file()` evaluated while the workflow is still being built can miss it. Calling this
// from inside a channel operator defers it until the upstream process has finished, by
// which point the file exists and holds every completed task.
//
// --report_trace wins: that is a `-entry report` run naming an earlier run's trace, and
// passing it also switches this run's own trace off so the file cannot be truncated out
// from under the read. Otherwise this run's own trace, then the newest one in the launch
// directory, then a sentinel -- a missing trace drops the resource sections and leaves the
// accuracy ones working, rather than failing a whole run over a report.
def resolveTrace = { ->
    for (candidate in [params.report_trace, params.trace_file]) {
        if (!candidate) continue
        def f = file(candidate)
        if (f.exists()) return f
    }
    def found = file("${launchDir}").listFiles()?.findAll { it.name.endsWith('.trace.txt') }
    if (found) return found.max { it.lastModified() }
    return file("${projectDir}/assets/NO_TRACE")
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
    tuple val(species), path(query_db), path(target_db)

    output:
    tuple val(species), path("${species}.identity.tsv")

    script:
    """
    set -euo pipefail
    mkdir -p tmp
    # Permissive on purpose: the job is to MEASURE identity for remote pairs, not to decide
    # whether they are homologous -- the shared Pfam family already decided that. A strict
    # search would silently drop the twilight-zone pairs this axis exists for.
    mmseqs search \\
        ${query_db}/db ${target_db}/db result tmp \\
        --threads ${task.cpus} \\
        -s 7.5 -e 10000 --max-seqs 300

    mmseqs convertalis \\
        ${query_db}/db ${target_db}/db result ${species}.identity.tsv \\
        --threads ${task.cpus} \\
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

process proteomeDisorder {
    /*
     * Per-protein disorder predicted from SEQUENCE, as a second opinion on the pLDDT
     * proxy rather than a replacement for it.
     *
     * pLDDT below 50 is a confidence measurement that correlates with disorder, not a
     * disorder measurement. It also drops when AlphaFold merely modelled a protein badly,
     * which usually means a shallow MSA -- and a shallow MSA independently hurts jackhmmer
     * and hhblits. So "accuracy falls with disorder" read off pLDDT alone could partly be
     * an MSA-depth effect hitting several arms at once. metapredict needs no structure and
     * no alignment, so it shares neither confound.
     *
     * Run over EVERY proteome, not just the human query. An earlier version did query-side
     * only, on the reasoning that disorder is a covariate of the scored object and the
     * scored object is the human domain instance. That reasoning is too narrow: a
     * structure-based tool needs a confident structure on BOTH sides of an alignment, so a
     * disordered TARGET defeats foldseek and reseek exactly as thoroughly as a disordered
     * query. "Does a structure-free method still find these when the target is the
     * unmodellable one" is the sharper form of this benchmark's claim, and it cannot be
     * asked from query-side numbers.
     *
     * Cached per proteome under DB_CACHE, because a target's disorder depends only on that
     * target: the midi and full runs share every target entry, and only the human one
     * differs, which is why the human label carries the query-set digest.
     */
    tag "${label}"
    label 'python'
    container params.metapredict_image
    storeDir "${DB_CACHE}/disorder"

    input:
    tuple val(label), path(fasta)

    output:
    path "${label}.disorder_metapredict.parquet"

    script:
    def thr = params.metapredict_threshold ? "--threshold ${params.metapredict_threshold}" : ""
    """
    predict_disorder_metapredict.py \\
        --fasta       ${fasta} \\
        ${thr} \\
        --out         ${label}.disorder_metapredict.parquet \\
        --summary-out ${label}.disorder_summary.json

    # Nested inside nothing: this is a single FILE output, so storeDir has no directory to
    # collide with -- see the note on kmerseekIndex for why that distinction matters.
    cat ${label}.disorder_summary.json
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
    tuple path(truth), path(hgnc), path(omega), path(structures), path(disorder)

    output:
    path "human_query_covariates.parquet", emit: covariates
    path "covariates_summary.json",        emit: summary

    script:
    def hgnc_arg   = hgnc.name   == 'NO_HGNC'   ? "" : "--hgnc ${hgnc}"
    def omega_arg  = omega.name  == 'NO_OMEGA'  ? "" : "--omega ${omega}"
    def struct_arg = structures.name == 'NO_STRUCTURES' ? "" : "--structures ${structures}"
    def mobidb_arg = params.mobidb_cache ? "--mobidb ${params.mobidb_cache}" : ""
    def mpred_arg  = disorder.name == 'NO_DISORDER' ? "" : "--metapredict ${disorder}"
    """
    build_query_covariates.py \\
        --truth       ${truth} \\
        ${hgnc_arg} ${omega_arg} ${struct_arg} ${mobidb_arg} ${mpred_arg} \\
        --out         human_query_covariates.parquet \\
        --summary-out covariates_summary.json
    """
}

// ===========================================================================
// KMERSEEK — index per (species, alphabet, ksize, low-complexity), then search
// ===========================================================================

process kmerseekIndex {
    /*
     * Build the RocksDB index for one target proteome under one alphabet/ksize/
     * low-complexity setting, and KEEP it under storeDir.
     *
     * This was fused with the search until 2026-08-25, specifically so the index never
     * became a declared output: it was built in the work dir, used once, and deleted
     * before the task exited, which bounded steady-state disk at (maxForks x one index)
     * rather than the whole matrix. An earlier all-vs-all run had died with "No space
     * left on device" doing it the other way, and that failure mode has not gone away --
     * see the disk note below.
     *
     * What changed is that there are now two runs over the same targets. The index
     * depends only on the TARGET proteome, so run-midi (964 chr6 queries) and the full
     * run (19_696 queries) build byte-identical indexes for all 3294 combos. Fused, the
     * full run rebuilds every one of them. Split, with storeDir on DB_CACHE rather than
     * outdir, the full run inherits the whole set and pays only for search.
     *
     * DISK: ~3294 indexes. Measured write volume was a 2.1 GB median per fused task, so
     * budget multiple TB on $SCRATCH and check `df` before launching the full run.
     * `make clean-indexes` drops the cache once the last run over these targets is done.
     *
     * ONE output, and it is the directory. The spectrum, the log and the timing record
     * live INSIDE it. A sibling file next to a directory output under storeDir is the
     * exact shape that produced this repo's recurring "Directory not empty" failure,
     * twice.
     *
     * The timing record is written by the task itself rather than left to the Nextflow
     * trace. A storeDir hit executes NO task, so Nextflow writes no trace row for it, and
     * with 3294 indexes cached in DB_CACHE the hit is the normal case from now on -- the
     * arm would be permanently unmeasured. Nesting the record inside the directory means
     * the measurement is cached alongside the thing it describes. kmerseekSearch carries
     * it back out; see the timings output there.
     *
     * DO NOT "optimise" this further by indexing human once and searching the species
     * against it. It looks like the same win -- 366 index builds instead of 3294 -- but
     * those arms cache a database used in the SAME direction every time, and this one
     * cannot be. kmerseek computes regions on the QUERY side. Making human the target
     * moves the scored interval onto the species protein, and human survives only as
     * transfer coordinates: the side that picks WHICH Pfam domain to transfer, not the
     * side that gets scored. The benchmark's unit is the human domain interval, so that
     * swap answers "what does the mouse region look like" instead of "did the tool find
     * titin's Ig domain". Rejected deliberately on 2026-08-24, not overlooked.
     */
    tag "${species}_${label}_k${ksize}_lc${lowcomp}"
    storeDir "${DB_CACHE}/kmerseek_index"

    memory { kmerseekMemory(label, ksize, species_fasta.size(), task.attempt) }
    // Retries the OOM signals only. Do NOT widen this to exit 1 to catch the
    // "Directory not empty" unstage failure -- that was measured on 2026-08-27 and it does
    // not work. Nextflow reads the store when it CREATES a task and caches that decision
    // on the task itself, so a retry does not re-check: attempts 2 and 3 re-ran the whole
    // index build and died on the identical mv. Retrying exit 1 here would turn one wasted
    // index build into three and still fail the run, while also retrying the genuine
    // errors that exit 1 otherwise means.
    //
    // `stageOutMode 'copy'` was measured the same day and is worse than the disease. It
    // does apply to storeDir, and it does make the run go green, because it swaps the
    // rename for `cp -fRL`, which MERGES into the existing directory instead of failing.
    // Two independent index builds are not guaranteed byte-identical, so the merged entry
    // is neither one: in the test, CURRENT came from the second build and the SST files
    // from both. A RocksDB whose manifest and data files come from different builds is
    // corrupt, and the run reports success. Worse still, the merge writes into an entry a
    // concurrent kmerseekSearch may be reading. A loud failure is the better outcome.
    errorStrategy { task.exitStatus in 128..143 ? 'retry' : 'finish' }
    maxRetries 2

    input:
    tuple val(species), path(species_fasta), val(cli_flag), val(label), val(ksize),
          val(lowcomp)

    output:
    path "${species}.${label}.k${ksize}.lc${lowcomp}.kmerseek.rocksdb"

    script:
    def slug      = "${label}.k${ksize}.lc${lowcomp}"
    def index_dir = "${species}.${slug}.kmerseek.rocksdb"
    def spectrum  = "spectrum.${species}.${slug}.csv.gz"
    // The new CLI treats --remove-low-complexity as a presence-only index flag. Search
    // inherits the index setting when the option is omitted, so false emits nothing and
    // true emits the flag without a value.
    def lc_flag   = lowcomp ? "--remove-low-complexity" : ""
    def timing    = "timing.jsonl"
    """
    set -euo pipefail
    ${KMERSEEK_TIMER_SH}

    _task_t0=\$(_now_ms)

    echo "=== Index: ${species} ${cli_flag} k=${ksize} lc=${lowcomp} ===" | tee index.log
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a index.log

    # --kmer-stats-out writes the k-mer frequency spectrum for this proteome under this
    # alphabet/ksize/low-complexity setting. Kept as a first-class output: the spectra are
    # what show WHY an alphabet behaves as it does, and the with/without low-complexity
    # pair is only interpretable if both spectra exist.
    _cmd_t0=\$(_now_ms)
    kmerseek index \\
        --alphabet ${cli_flag} \\
        --ksize    ${ksize} \\
        --input    ${species_fasta} \\
        --output   ${index_dir} \\
        ${lc_flag} \\
        --kmer-stats-out ${spectrum} \\
        2>&1 | tee -a index.log
    _cmd_s=\$(_elapsed_s \$_cmd_t0)

    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a index.log
    echo "index size: \$(du -sh ${index_dir} | cut -f1)" | tee -a index.log

    # command_s is the `kmerseek index` invocation alone; realtime_s is the whole task, so
    # it is the figure comparable to a Nextflow trace row for any other process.
    printf '{"process":"kmerseekIndex","tag":"%s","cpus":%s,"realtime_s":%s,"command_s":%s,"n_queries_all":null}\\n' \\
        "${species}_${label}_k${ksize}_lc${lowcomp}" "${task.cpus}" \\
        "\$(_elapsed_s \$_task_t0)" "\$_cmd_s" > ${timing}

    # Nested inside the index directory, not emitted alongside it -- see the storeDir note
    # above. kmerseek reads only the RocksDB files it wrote, so these are inert here.
    touch ${spectrum}
    mv ${spectrum} index.log ${timing} ${index_dir}/
    """
}

process kmerseekSearch {
    /*
     * Search human against one stored target index. Emits the region table the benchmark
     * scores, plus the spectrum carried out of the index directory so downstream plotting
     * does not have to reach inside a storeDir path.
     *
     * target_bytes is the TARGET proteome's FASTA size, passed as a value rather than
     * measured from the staged index: the index arrives as a directory symlink, and
     * .size() on a directory returns the dirent size, not the tree. Sizing memory off
     * that silently gave every search the floor allocation.
     *
     * The timings output is how this arm gets a throughput number at all. A storeDir hit
     * runs no task and so leaves no Nextflow trace row, which is why kmerseek was blank in
     * the report's speed sections while every publishDir baseline was populated. Writing
     * the measurement into the store means it is served back on the hit, exactly like the
     * regions parquet it describes, and `-entry report` reads the same files.
     *
     * `optional: true` on that output is load-bearing, not defensive. storeDir decides
     * "already done" by checking the declared outputs exist in the store, so a mandatory
     * third output would miss on every one of the 3294 entries written before this
     * existed and re-run the entire sweep to recover a timing number. Optional lets those
     * entries keep hitting; they simply have no record, which the report reports as a gap
     * rather than as a zero.
     */
    tag "${species}_${label}_k${ksize}_lc${lowcomp}"
    storeDir "${params.outdir}/kmerseek"

    memory { kmerseekMemory(label, ksize, target_bytes, task.attempt) }
    // Retry the OOM signals (128..143), stop the run on anything else. Deliberately not
    // 'ignore': a combo that dies and gets skipped leaves an empty result that reads
    // downstream as "this alphabet found nothing", which is indistinguishable from a real
    // negative. That has already happened once on this project -- 17 combos silently
    // searched ~1000 of 19,696 queries and looked like genuine misses. Failing loudly and
    // resuming costs queue time; a silent partial costs a wrong conclusion.
    errorStrategy { task.exitStatus in 128..143 ? 'retry' : 'finish' }
    maxRetries 2

    input:
    tuple val(species), val(cli_flag), val(label), val(ksize), val(lowcomp),
          val(target_bytes), path(index_dir), path(human_fasta)

    output:
    path "human_vs_${species}.${label}.k${ksize}.lc${lowcomp}.regions.parquet",  emit: regions
    path "spectrum.${species}.${label}.k${ksize}.lc${lowcomp}.csv.gz",           emit: spectrum
    path "human_vs_${species}.${label}.k${ksize}.lc${lowcomp}.timings.jsonl",
         optional: true, emit: timing

    script:
    def slug      = "${label}.k${ksize}.lc${lowcomp}"
    def out_zst   = "human_vs_${species}.${slug}.regions.csv.zst"
    def out_pq    = "human_vs_${species}.${slug}.regions.parquet"
    def log_file  = "human_vs_${species}.${slug}.log"
    def spectrum  = "spectrum.${species}.${slug}.csv.gz"
    def timings   = "human_vs_${species}.${slug}.timings.jsonl"
    def lc_flag   = lowcomp ? "--remove-low-complexity" : ""
    """
    set -euo pipefail
    ${KMERSEEK_TIMER_SH}

    _task_t0=\$(_now_ms)

    echo "=== Search: human vs ${species} (${cli_flag} k=${ksize} lc=${lowcomp}) ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}

    # Carried out of the index directory rather than rebuilt: the spectrum is a property of
    # the target proteome under this alphabet/ksize, which is exactly what kmerseekIndex
    # already computed and stored.
    cp ${index_dir}/${spectrum} ${spectrum} 2>/dev/null || touch ${spectrum}

    # The index's own timing record, carried out of the store the same way. An index built
    # before this record existed simply has none, so the report shows the search cost with
    # no index cost beside it rather than failing.
    cp ${index_dir}/timing.jsonl ${timings} 2>/dev/null || : > ${timings}

    # n_queries_all: every sequence in the FASTA handed to kmerseek, NOT n_queries_ref,
    # the FoldSeek-intersected subset. Recorded for provenance only -- the report divides
    # by its own query count so kmerseek and the baselines are over the same denominator,
    # and this is what catches the case where they are not.
    n_queries=\$(grep -c '^>' ${human_fasta} || true)
    n_queries=\${n_queries:-0}

    # --min-region-score is OR'd with --max-query-pvalue inside kmerseek, so this keeps
    # sub-protein domain hits whose whole-query p-value is unimpressive. Do not "tighten"
    # this by also lowering --max-query-pvalue expecting fewer rows; the OR means the
    # looser of the two governs.
    set +e
    _cmd_t0=\$(_now_ms)
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
        | zstd -T2 -o ${out_zst}
    rc=(\${PIPESTATUS[@]})
    _cmd_s=\$(_elapsed_s \$_cmd_t0)
    set -e

    # kmerseek's own nonzero exit stays tolerated: a combo that finds nothing is a real
    # result. zstd's does NOT. It used to share this `|| true`, so "cannot write block:
    # Cannot allocate memory" left a truncated .zst that polars read as far as it could
    # before dying on "incomplete frame" -- a memory failure surfacing as a parquet error
    # two steps later. Re-raise it as a signal so the retry ladder doubles the allocation.
    if [ "\${rc[1]}" -ne 0 ]; then
        echo "zstd exited \${rc[1]}: the region stream is truncated, not a short result" >&2
        exit 137
    fi
    if [ "\${rc[0]}" -ne 0 ]; then
        echo "note: kmerseek search exited \${rc[0]}; treating as a no-hit result" >&2
    fi

    touch ${out_zst}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "regions csv.zst: \$(du -sh ${out_zst} | cut -f1)" | tee -a ${log_file}

    # Straight to parquet, dropping columns no downstream step reads. Both formats kept
    # around for 3294 result files is the disk blow-up this design avoids.
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
    rm -f ${out_zst}

    echo "regions parquet: \$(du -sh ${out_pq} | cut -f1)" | tee -a ${log_file}

    # realtime_s covers the whole task, csv-to-parquet included, because that is what a
    # Nextflow trace row covers for every baseline -- foldseekSearch's realtime includes
    # its own output normalisation too. Matching the definition is the point: the report
    # divides this by the query count to get queries/s and puts it on the same axis as
    # foldseek's 17.1 and prostt5's 14.4. command_s is the `kmerseek search` invocation
    # alone, kept for attributing the difference rather than for the headline number.
    printf '{"process":"kmerseekSearch","tag":"%s","cpus":%s,"realtime_s":%s,"command_s":%s,"n_queries_all":%s}\\n' \\
        "${species}_${label}_k${ksize}_lc${lowcomp}" "${task.cpus}" \\
        "\$(_elapsed_s \$_task_t0)" "\$_cmd_s" "\$n_queries" >> ${timings}
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

process mmseqsDb {
    /*
     * One MMseqs2 database per FASTA, cached. easy-search rebuilt the human query database
     * in every task: 18 times for mmseqs2Search (2 variants x 9 species) and 9 more for
     * domainIdentity. Each pass is cheap -- createdb on a FASTA is seconds, not the minutes
     * ProstT5 costs -- but the two search variants can share one database for free.
     *
     * Generic on label so the proteome FASTAs and the domain FASTAs both use it; labels
     * must therefore be distinct across callers (human vs human_domains).
     */
    tag "${label}"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    label 'high_cpu'
    storeDir "${DB_CACHE}/mmseqs_db"

    input:
    tuple val(label), path(fasta)

    output:
    path "${label}_mmdb"

    script:
    """
    set -euo pipefail
    mkdir -p ${label}_mmdb
    mmseqs createdb ${fasta} ${label}_mmdb/db
    """
}

process mmseqsDomainDb {
    /*
     * One MMseqs2 database per FASTA, cached. easy-search rebuilt the human query database
     * in every task: 18 times for mmseqs2Search (2 variants x 9 species) and 9 more for
     * domainIdentity. Each pass is cheap -- createdb on a FASTA is seconds, not the minutes
     * ProstT5 costs -- but the two search variants can share one database for free.
     *
     * A near-copy of mmseqsDb rather than a second call to it: DSL2 allows a process to be
     * invoked once per workflow, and aliasing needs it to live in an included module. Kept
     * separate so the domain DBs get their own storeDir too -- different FASTAs, and they
     * should not share a cache namespace with the proteome databases.
     */
    tag "${label}"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    label 'high_cpu'
    storeDir "${params.outdir}/mmseqs_domain_db"

    input:
    tuple val(label), path(fasta)

    output:
    path "${label}_mmdb"

    script:
    """
    set -euo pipefail
    mkdir -p ${label}_mmdb
    mmseqs createdb ${fasta} ${label}_mmdb/db
    """
}

process mmseqsPaddedDb {
    /*
     * The GPU-layout copy of a target database. Separate from mmseqsDb rather than an
     * option on it, because the plain database is the INPUT to the padding pass and both
     * copies have to survive: the CPU arm reads the plain one and a rerun would otherwise
     * have to go back to the FASTA. Measured on the yeast proteome the padded copy is
     * 4_100_850 bytes against the plain 4_106_532, so the cost is one extra copy of the
     * database, not a multiple of it.
     *
     * Only target proteomes are padded. The query side is read unpadded by the GPU
     * prefilter, so the human entry never reaches this process and the HUMAN_LABEL digest
     * convention needs nothing here -- but the label passes through untouched, so a
     * digest-keyed label would land in its own cache entry if that ever changed.
     */
    tag "${label}"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    label 'high_cpu'
    storeDir "${DB_CACHE}/mmseqs_db_gpu"

    input:
    tuple val(label), path(plain_db)

    output:
    path "${label}_mmdb_gpu"

    script:
    """
    set -euo pipefail
    mkdir -p ${label}_mmdb_gpu
    mmseqs makepaddedseqdb ${plain_db}/db ${label}_mmdb_gpu/db --threads ${task.cpus}
    """
}

process mmseqs2Search {
    tag "human_vs_${species} [${variant}${gpu ? ' gpu' : ''}]"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/${variant}", mode: 'copy', pattern: '*.tsv.gz'

    // mode_variant and out_name are computed in the workflow, not here. An `output:` val()
    // holding an expression is evaluated when the process is DECLARED, before any input is
    // bound, so a ternary on `gpu` in that position fails with "No such variable: gpu".
    // path() patterns interpolate per task and would be fine, but keeping both strings on
    // one side of the boundary means the CPU spelling is provably unchanged.
    input:
    tuple val(species), val(variant), val(num_iter), val(gpu), val(mode_variant),
          val(out_name), path(target_db), path(query_db)

    // The mode rides in the `variant` slot, never in the `tool` slot: scoreDomainCalls
    // switches --dedup-fragments and the interval semantics on the tool name, so renaming
    // the tool would silently change how the arm is scored.
    output:
    tuple val(species), val(variant), val(mode_variant), path(out_name)

    script:
    def iter_flag = num_iter > 1 ? "--num-iterations ${num_iter}" : ""
    // -s is dropped in GPU mode rather than passed and ignored, so the command line in
    // .command.sh is a truthful record of what ran.
    def mode_flag = gpu ? "--gpu 1" : "-s ${params.mmseqs2_sensitivity}"
    """
    set -euo pipefail
    mkdir -p mmseqs_tmp
    mmseqs search \\
        ${query_db}/db ${target_db}/db result mmseqs_tmp \\
        --threads ${task.cpus} \\
        ${mode_flag} \\
        ${iter_flag} \\
        --max-seqs 1000 \\
        -e ${params.evalue_report}

    # convertalis reads the SAME target database the search used. makepaddedseqdb renumbers
    # the database keys, so resolving a GPU result against the plain database would emit
    # the wrong accessions rather than fail.
    mmseqs convertalis \\
        ${query_db}/db ${target_db}/db result out.tsv \\
        --threads ${task.cpus} \\
        --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"
    gzip -c out.tsv > ${out_name}
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

process foldseekDb {
    /*
     * One Foldseek structure database per proteome, cached. easy-search rebuilt the human
     * query database inside every task, so the 20_589 human mmCIF files were parsed and
     * 3Di-encoded 9 times instead of once. Cheaper per pass than ProstT5 -- this is file
     * parsing, not inference -- but it is 8 wasted passes over ~20k structures.
     */
    tag "${label}"
    container 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
    label 'high_cpu'
    storeDir "${DB_CACHE}/foldseek_db"

    input:
    tuple val(label), path(structures)

    // Single directory output, label recovered from its name -- see prostt5Db.
    output:
    path "${label}_fsdb"

    script:
    """
    set -euo pipefail
    mkdir -p ${label}_fsdb
    foldseek createdb ${structures}/ ${label}_fsdb/db --threads ${task.cpus}
    """
}

process foldseekPaddedDb {
    /*
     * The GPU-layout copy of a target structure database. foldseek's own makepaddedseqdb
     * is not a thin wrapper around the mmseqs one: it pads the _ss (3Di) sub-database,
     * then renumbers the amino-acid, _ca and _h sub-databases to match, so all four stay
     * consistent. Verified against the pinned image -- one call on a foldseek createdb
     * output produces db_gpu, db_gpu_ss, db_gpu_ca and db_gpu_h.
     *
     * Also verified: a CPU search against the padded database returns results byte-identical
     * to the same search against the plain one, accessions included, so padding on its own
     * changes nothing about the answer. The GPU flag is what changes the prefilter.
     *
     * The copy at the end is load-bearing, not tidiness. Only the _ss sub-database is
     * actually rewritten; foldseek renames the amino-acid, _ca and _h keys with
     * `renamedbkeys --subdb-mode 1`, which writes ABSOLUTE SYMLINKS back into the plain
     * database instead of copying it. Under storeDir that leaves the padded directory
     * pointing at ${DB_CACHE}/foldseek_db, a path Nextflow never declares as an input to
     * foldseekSearch, so Apptainer would not bind it and the links would dangle inside the
     * container -- the same shape as the structures-directory symlink bug documented in
     * the workflow. `cp -RL` dereferences them so the directory stands on its own. The
     * price is one extra copy of each target proteome's coordinate data, which is the
     * bulk of a foldseek database; the mmseqs padded database needs none of this because
     * makepaddedseqdb writes it as real files throughout.
     */
    tag "${label}"
    container 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
    label 'high_cpu'
    storeDir "${DB_CACHE}/foldseek_db_gpu"

    input:
    tuple val(label), path(plain_db)

    output:
    path "${label}_fsdb_gpu"

    script:
    """
    set -euo pipefail
    mkdir -p padded_tmp ${label}_fsdb_gpu
    foldseek makepaddedseqdb ${plain_db}/db padded_tmp/db --threads ${task.cpus}
    cp -RL padded_tmp/. ${label}_fsdb_gpu/
    """
}

process foldseekSearch {
    /*
     * Structure-structure search over two prebuilt databases. No createdb here.
     */
    tag "human_vs_${species}${gpu ? ' [gpu]' : ''}"
    container 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    // mode_variant and out_name come from the workflow -- see the note in mmseqs2Search on
    // why an output val() cannot hold a ternary over an input.
    input:
    tuple val(species), val(gpu), val(mode_variant), val(out_name),
          path(target_db), path(query_db)

    // Tool stays "foldseek" so scoreDomainCalls keeps applying --dedup-fragments; the mode
    // rides in the variant slot. Unlike mmseqs the alphabet is unchanged between modes, so
    // the existing label keeps its meaning and only gains a suffix. The prefilter still
    // changes -- CPU runs `prefilter -s 9.5`, GPU runs `ungappedprefilter` -- which is the
    // reason the two arms are scored separately rather than treated as one result.
    output:
    tuple val(species), val("foldseek"), val(mode_variant), path(out_name)

    script:
    def gpu_flag = gpu ? "--gpu 1" : ""
    """
    set -euo pipefail
    mkdir -p foldseek_tmp
    foldseek search \\
        ${query_db}/db ${target_db}/db result foldseek_tmp \\
        --threads ${task.cpus} \\
        ${gpu_flag} \\
        -e ${params.evalue_report} \\
        --max-seqs 1000

    # Same database on both sides of search and convertalis -- makepaddedseqdb renumbers
    # keys, so a padded search resolved against the plain database yields wrong accessions.
    foldseek convertalis \\
        ${query_db}/db ${target_db}/db result out.tsv \\
        --threads ${task.cpus} \\
        --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"

    # Foldseek names rows by structure filename (AF-<acc>-F<n>.cif). Reduce to the bare
    # accession so every arm keys the same way, and shift the interval by the fragment
    # offset first. AlphaFold models proteins over 2700 aa as overlapping 1400-residue
    # fragments on a 200-residue stride, each numbered from 1, so a hit on F<n> sits
    # (n-1)*200 before its true position. Verified on AF-A0A087WUL8-F2: auth_seq_id
    # 1..1400, SIFTS xref UniProt 201..1600.
    awk -F'\t' 'BEGIN{OFS="\t"} {
        for (i = 1; i <= 2; i++) {
            # One match() only. An earlier version called a helper that ran its own
            # match(), which reset RSTART before this substr() used it and emptied the
            # accession column outright.
            if (match(\$i, /AF-[A-Z0-9]+-F[0-9]+/)) {
                tok = substr(\$i, RSTART, RLENGTH)     # AF-<acc>-F<n>
                split(tok, p, "-")                     # p[2]=accession, p[3]=F<n>
                off = (substr(p[3], 2) + 0 - 1) * 200
                if (off) { \$(2 * i + 1) += off; \$(2 * i + 2) += off }
                \$i = p[2]
            }
        }
        print
    }' out.tsv | gzip -c > ${out_name}
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
    storeDir "${DB_CACHE}/reseek_db"

    input:
    tuple val(species), path(structures)

    output:
    tuple val(species), path("${species}.bca"), path("${species}.mu.fasta")

    script:
    """
    set -euo pipefail
    # -bca, not -bcb: .bca is the binary C-alpha format Reseek recommends for databases.
    # -threads, not -t. Both taken from `reseek` with no arguments in the pinned image,
    # after -bcb failed with "Unknown option bcb" -- the flags used before came from a
    # README summary rather than from the binary.
    # STRUCTS accepts a directory and .cif/.mmcif, both confirmed in that same usage text.
    # A .files list, not the directory. Nextflow stages `path(structures)` as a SYMLINK,
    # and reseek's directory walk does not descend into one: `reseek -convert mouse` on a
    # staged 21_452-structure proteome reported "0 chains" and then SEGFAULTED on the empty
    # input. foldseekDb survives the same staging only because it writes `${structures}/`
    # with a trailing slash, which forces the symlink to resolve.
    #
    # `NAME.files` is a documented STRUCTS form -- "Text file with one STRUCT per line" --
    # so this sidesteps directory walking entirely. find -L follows the staged symlink and
    # the per-file ones underneath it, which was verified inside this exact container.
    find -L ${structures}/ \\( -name '*.cif' -o -name '*.mmcif' -o -name '*.pdb' \\) \\
        | sort > ${species}.files

    # Reseek segfaults on zero structures rather than reporting an error, so an empty list
    # has to be caught here or the failure arrives as exit 139 with nothing to read.
    n_structs=\$(wc -l < ${species}.files)
    echo "reseek ${species}: \${n_structs} structures listed" >&2
    if [ "\$n_structs" -eq 0 ]; then
        echo "no .cif/.mmcif/.pdb found under ${structures}/ -- reseek would segfault on this" >&2
        exit 1
    fi

    reseek -convert ${species}.files -bca ${species}.bca \\
        -threads ${task.cpus} -log ${species}.convert.log

    # Mu is Reseek's 36-letter structural alphabet: -convert2mu encodes each structure's
    # local backbone geometry as a sequence, and -search uses that as a PREFILTER, matching
    # Mu strings to shortlist candidates before doing 3D alignment on the survivors. The
    # binary's own usage says "Accelerates search with -db_mu option".
    #
    # Without it every query is structurally aligned against every target. That is what a
    # 964-query midi search against zebrafish was doing when it reached 1.37 GB of hits in
    # 100 minutes and had not finished after 26 hours.
    #
    # Built here rather than at search time because it depends only on the target proteome,
    # so it caches under storeDir beside the .bca and is paid for once per species.
    #
    # From the .bca when that works, since re-reading every structure is the expensive half
    # of conversion; .bca is listed as a valid STRUCTS form in the same usage text. Falling
    # back to the .files list rather than trusting that: this process already carries a scar
    # from flags taken off a README instead of out of the binary.
    if ! reseek -convert2mu ${species}.bca -fasta ${species}.mu.fasta \\
            -threads ${task.cpus} -log ${species}.mu.log 2> mu.err; then
        cat mu.err >&2
        echo "convert2mu from .bca failed; retrying from the .files list" >&2
        rm -f ${species}.mu.fasta
        reseek -convert2mu ${species}.files -fasta ${species}.mu.fasta \\
            -threads ${task.cpus} -log ${species}.mu.log
    fi
    """
}

process reseekSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/reseek@sha256:24f7c37150dd2c2f2f322b1387a08d2d1a4a279f46f98f1051f1745417675752'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/reseek", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(db), path(db_mu), path(human_bca)

    // Variant is the mode actually run, not a hardcoded string. This said "sensitive"
    // while params.reseek_mode was "verysensitive", so every reseek row in every metrics
    // table named a setting the search had not used -- and switching modes would not have
    // changed the label, so the two arms would have silently collided in the results.
    output:
    tuple val(species), val("reseek"), val(params.reseek_mode),
          path("human_vs_${species}.reseek.tsv.gz")

    script:
    // Bound as locals rather than interpolated inline. `${db}` followed by more flags
    // tripped the Groovy lexer at that column; naming them keeps the script block
    // plain text.
    // Query is now a cached .bca like the target, not a structure directory: reseek's
    // own help lists .bca as the recommended DB format and -search takes the same
    // STRUCTS argument for either side, so human is converted once instead of per task.
    def q_dir   = human_bca.toString()
    def db_file = db.toString()
    def mu_file = db_mu.toString()
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
    # -dbmu is the prefilter. Spelled as the binary's own usage synopsis spells it,
    # "reseek -search STRUCTS -db STRUCTS [-dbmu db_mu.fasta]", not as the prose two lines
    # below it spells it ("-db_mu option"); the synopsis is what the binary prints for its
    # own arguments. If this ever fails with "Unknown option dbmu", the prose was right.
    #
    # -evalue is pinned rather than inherited. The usage says "Max E-value (default 10
    # unless -verysensitive)", so -verysensitive silently relaxes the cutoff, and that
    # governs output volume as much as it governs sensitivity: one midi species reached
    # 1.37 GB of hits in 100 minutes under the relaxed default. Every other arm in this
    # benchmark reports at params.evalue_report, so reseek reporting at something else
    # would have been comparing arms at different thresholds -- and the reseek arm's output
    # is read back by scoreDomainCalls through polars, which does not stream compressed
    # CSV, so an unbounded arm costs scoring memory too.
    reseek -search ${q_dir} -db ${db_file} -dbmu ${mu_file} -${mode} \\
        -evalue ${params.evalue_report} -columns ${cols} \\
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
    container 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
    label 'high_cpu'
    // The model weights depend on nothing this pipeline varies -- not the query set, not
    // the targets -- so they belong in the shared cache rather than being re-downloaded
    // into each run's own outdir.
    // Leaf name kept as `prostt5`, not `prostt5_weights`: this used to store under
    // ${params.outdir}/prostt5, and DB_CACHE defaults to params.outdir, so keeping the leaf
    // means an existing weights download is still found instead of being orphaned and
    // re-fetched. Sharing comes from DB_CACHE, not from renaming the directory.
    storeDir "${DB_CACHE}/prostt5"

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

process prostt5Db {
    /*
     * ProstT5 3Di prediction for ONE proteome, cached. This exists because easy-search
     * rebuilds its query database inside every task: with human as the query against 9
     * targets, the whole human proteome was re-encoded 9 times -- 90_236_496 of the arm's
     * 173_017_199 residues, 52% of its compute, spent recomputing an identical answer.
     * ProstT5 is a neural net, so that waste is inference time, not file parsing.
     *
     * hhblitsBuildDB already had this shape: one channel carrying human plus every
     * species, each built once. This is the same pattern applied to the expensive arm.
     */
    tag "${label}"
    container 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
    cpus   { Math.max(1, (params.prostt5_cpus as int).intdiv(task.attempt)) }
    memory { MemoryUnit.of(params.prostt5_memory) * task.attempt }
    time   { params.prostt5_time }
    errorStrategy { task.exitStatus in 128..143 ? 'retry' : 'finish' }
    maxRetries 3
    storeDir "${DB_CACHE}/prostt5_db"

    input:
    tuple val(label), path(fasta), path(weights)

    // ONE directory output, and the skipped list lives INSIDE it. A directory under
    // storeDir alongside a second output is the shape that produced recurring "Directory
    // not empty" failures elsewhere in this repo; nesting keeps it to a single output.
    // The label is recovered in the workflow from the directory name.
    output:
    path "${label}_prostt5"

    script:
    """
    set -euo pipefail
    mkdir -p ${label}_prostt5 tmp

    # Length filter -- see params.prostt5_max_len. Applied here rather than at search time
    # so query and target proteomes are filtered by one rule in one place.
    printf 'accession\\tlength\\n' > ${label}_prostt5/skipped.tsv
    awk -v maxlen=${params.prostt5_max_len} -v skipped=${label}_prostt5/skipped.tsv '
        function flush(   parts, acc) {
            if (hdr == "") return
            if (length(seq) <= maxlen) { print hdr; print seq; return }
            split(substr(hdr, 2), parts, "|")
            acc = (parts[2] != "") ? parts[2] : parts[1]
            print acc "\t" length(seq) >> skipped
        }
        /^>/ { flush(); hdr = \$0; seq = ""; next }
        { seq = seq \$0 }
        END { flush() }
    ' ${fasta} > filtered.fasta

    n_skipped=\$(( \$(wc -l < ${label}_prostt5/skipped.tsv) - 1 ))
    echo "prostt5 ${label}: dropped \${n_skipped} sequences longer than ${params.prostt5_max_len} aa" >&2

    # createdb is where ProstT5 runs. See the note in prostt5Search on why a nonzero exit
    # has to be inspected rather than trusted.
    set +e
    foldseek createdb filtered.fasta ${label}_prostt5/db \\
        --prostt5-model ${weights} \\
        ${params.prostt5_gpu ? '--gpu 1' : ''} \\
        --threads ${task.cpus} \\
        2> createdb.err
    rc=\$?
    set -e
    cat createdb.err >&2
    if [ "\$rc" -ne 0 ]; then
        if grep -qEi 'killed|cannot allocate|bad_alloc|out of memory' createdb.err; then
            echo "prostt5 ${label}: createdb was OOM-killed; re-raising as 137 to trigger retry" >&2
            exit 137
        fi
        exit "\$rc"
    fi
    """
}

process prostt5Search {
    /*
     * Foldseek over 3Di predicted from SEQUENCE on both sides -- no structures anywhere.
     * That is what makes this the differentiating baseline: it runs on every species,
     * including those where AlphaFold coverage is too thin for Foldseek or Reseek.
     *
     * Both databases arrive prebuilt from prostt5Db, so no ProstT5 inference happens here.
     */
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
    label 'high_cpu'
    publishDir "${params.outdir}/regions/prostt5", mode: 'copy', pattern: '*.tsv.gz'
    publishDir "${params.outdir}/regions/prostt5", mode: 'copy', pattern: '*_skipped.tsv'

    input:
    tuple val(species), path(target_db), path(query_db)

    output:
    tuple val(species), val("prostt5"), val("3di_from_seq"),
          path("human_vs_${species}.prostt5.tsv.gz"),      emit: regions
    path "human_vs_${species}.prostt5_skipped.tsv",        emit: skipped

    script:
    """
    set -euo pipefail
    mkdir -p tmp

    # The coverage gap is published from here rather than from prostt5Db, so one file per
    # comparison names both sides. These instances still count against ProstT5 in the
    # eval's tool-independent reachable denominator -- that is the honest accounting, and
    # this file is what keeps it from reading as unexplained missing recall.
    printf 'accession\\tside\\tlength\\n' > human_vs_${species}.prostt5_skipped.tsv
    awk -v side=query  'NR>1 {print \$1"\t"side"\t"\$2}' ${query_db}/skipped.tsv \\
        >> human_vs_${species}.prostt5_skipped.tsv
    awk -v side=target 'NR>1 {print \$1"\t"side"\t"\$2}' ${target_db}/skipped.tsv \\
        >> human_vs_${species}.prostt5_skipped.tsv

    # A database built this way carries predicted 3Di only, with no Ca coordinates, so
    # TMalign-based alignment types and TM-score/LDDT outputs are unavailable here. The
    # columns below are all sequence-space and unaffected.
    foldseek search \\
        ${query_db}/db ${target_db}/db result tmp \\
        --threads ${task.cpus} \\
        -e ${params.evalue_report} \\
        --max-seqs 1000

    foldseek convertalis \\
        ${query_db}/db ${target_db}/db result out.tsv \\
        --threads ${task.cpus} \\
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
    storeDir "${DB_CACHE}/folddisco_index"

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
    // The nine target databases are identical between the midi and full runs and cost
    // ~36 min each to build, so they belong in the shared cache alongside every other
    // per-target database. The human entry is keyed by query-set digest (HUMAN_LABEL),
    // which is what makes sharing this directory safe -- see the note in the workflow.
    storeDir "${DB_CACHE}/hhblits_db"

    input:
    tuple val(label), path(fasta), val(is_query)

    // Bare path, not a tuple: storeDir supports only val/path outputs and silently does
    // nothing for a tuple. The label comes back off the directory name via label_dbs,
    // the same way every other cached database in this pipeline recovers it.
    output:
    path "${label}_hhdb"

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

    # Target databases also need hhm + cs219; the query database only needs a3m. Driven by
    # the is_query flag, not by matching the label against "human" -- the human label now
    # carries a query-set digest, so a string test against it silently stopped firing.
    if [ "${is_query}" != "true" ]; then
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
    // One task covers every arm for this species now, so the tag names the group rather
    // than a tool. arms is the count because a full sweep is ~376 of them per species.
    tag "${truth_set}: ${species} (${tools.size()} arms)"
    // Its own label, NOT 'python'. Config directives beat script-declared ones, so while
    // this carried the python label that label's flat memory silently overrode the sizing
    // below. The scoring label sets the container and nothing else.
    label 'python_scoring'
    // Memory from the LARGEST arm, time from the SUM. The python loop is sequential, so
    // peak RSS is set by the biggest single regions file while wall clock is additive.
    // Sizing memory on the sum would reserve ~376x what any moment needs.
    memory { scoreMemory(regions instanceof List ? regions.max { it.size() } : regions,
                         task.attempt) }
    time   { scoreTime(regions, task.attempt) }
    errorStrategy { task.exitStatus in 128..143 ? 'retry' : 'finish' }
    maxRetries 3
    publishDir "${params.outdir}/calls",   mode: 'copy', pattern: '*.calls.parquet'
    publishDir "${params.outdir}/metrics", mode: 'copy', pattern: '*.metrics.parquet'
    publishDir "${params.outdir}/curves",  mode: 'copy', pattern: '*.curve.parquet'

    input:
    tuple val(truth_set), val(species), val(tools), val(variants), val(mya),
          path(regions, arity: '1..*'), path(truth), path(domain_map),
          path(covariates), path(identity), path(target_disorder)

    // Globs, because one task now writes a trio per arm. arity '1..*' for the same reason
    // it is on kmerseekIndex's chunk output: a glob emits a bare Path on a single match and
    // a List on several, and a species with one arm would otherwise break the collect.
    output:
    path "*.calls.parquet",   arity: '1..*', emit: calls
    path "*.metrics.parquet", arity: '1..*', emit: metrics
    path "*.curve.parquet",   arity: '1..*', emit: curve

    script:
    // The manifest is built from the STAGED paths, not from the channel's originals. If two
    // arms ever shipped a file of the same name Nextflow would rename one on staging, and a
    // manifest written from the original names would then point at the wrong file.
    //
    // interval-semantics and dedup-fragments used to be decided here per tool. They now live
    // in evaluate_domain_calls.score_one, because a manifest row carries only the tool name
    // and the policy has to be derived from it in exactly one place.
    def tdis_arg = target_disorder.name == 'NO_DISORDER' ? ""
                   : "--target-disorder ${target_disorder}"
    def manifest_rows = [tools, variants, regions].transpose()
        .collect { t, v, r -> "${t}\t${v}\t${r}" }.join("\n")
    """
    cat > manifest.tsv << 'MANIFEST_EOF'
${manifest_rows}
MANIFEST_EOF

    echo "scoring \$(wc -l < manifest.tsv) arms for ${truth_set}/${species}"

    evaluate_domain_calls.py \\
        --manifest     manifest.tsv \\
        --species      ${species} \\
        --species-mya  ${mya} \\
        --truth        ${truth} \\
        --domain-map   ${domain_map} \\
        --covariates   ${covariates} \\
        --identity     ${identity} \\
        ${tdis_arg} \\
        --min-overlap  ${params.min_overlap} \\
        --strict-iou   ${params.strict_iou} \\
        --truth-set    ${truth_set}
    """
}

process scoreHmmscanCeiling {
    /*
     * The annotation ceiling: human queries against Pfam-A directly, no target proteome.
     *
     * Deliberately NOT stratified by identity, and that is why there is no --identity here
     * while scoreDomainCalls has one. That axis is percent identity between a human domain
     * instance and its closest same-family domain IN A GIVEN TARGET PROTEOME, so the
     * identity table is per species and scoreDomainCalls joins it on species. This row is
     * scored once with --species all because hmmscan reads no target proteome at all, so
     * there is no species whose table applies and no principled way to pick one of the
     * nine. Attaching an arbitrary species' identity would stratify the ceiling by, say,
     * human-to-mouse divergence while labelling the row `all`.
     *
     * The consequence is that the ceiling has no rows on the identity axis, so it is
     * absent from the gray-zone sections rather than plotted at a made-up stratum. Passing
     * the NO_IDENTITY sentinel instead would produce the same numbers, but it would say
     * "this run skipped identity", which is a different and untrue statement.
     *
     * evaluate_domain_calls.py already treats --identity as optional: it defaults to None
     * and attach_identity() fills stratum_identity with nulls, the same path the sentinel
     * takes. Until 2026-08-26 the script body interpolated ${identity}, which the input
     * block never declared -- latent because Groovy only resolves it when the process
     * actually executes, and this arm was reached for the first time once the ProstT5 arm
     * was skipped on local runs.
     */
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
    path "all_domain_metrics.parquet", emit: metrics
    path "all_domain_metrics.csv",     emit: csv
    path "all_domain_curves.parquet",  emit: curves

    script:
    """
    aggregate_domain_metrics.py \\
        metrics curves \\
        all_domain_metrics.parquet all_domain_metrics.csv all_domain_curves.parquet
    """
}

process buildMultiqcInputs {
    /*
     * Metrics, curves and the Nextflow trace in; one *_mqc.json per report section out.
     *
     * The trace is read live rather than reconstructed: Nextflow appends a row as each
     * task completes, so by the time this runs every search and scoring task is already
     * recorded. The only rows missing are this task's own and the report task after it,
     * which is why the run-totals section says it counts tasks, not elapsed clock time.
     *
     * The trace cannot see the kmerseek arm, though. It is the only arm on storeDir, and a
     * store hit runs no task and writes no row, so those timings arrive separately as the
     * records kmerseekSearch wrote into its own store. Staged as inputs rather than read
     * from params.outdir directly: under Apptainer only staged paths are bound into the
     * container, so a script reaching into the results directory would find nothing there.
     */
    tag "multiqc inputs"
    label 'python'
    publishDir "${params.outdir}/multiqc", mode: 'copy'

    // The report is the last thing a run does, after every search and every scoring task
    // has already succeeded. Failing the whole run over a plot would throw away a finished
    // sweep, so inside the pipeline this arm is allowed to fail and the run still ends
    // green -- the metrics parquets are published either way.
    //
    // In a `-entry report` run the report IS the work, so there it must fail loudly.
    // params.report_trace is set only by that entry, which makes it the discriminator.
    errorStrategy { params.report_trace ? 'terminate' : 'ignore' }


    input:
    tuple path(metrics), path(curves), path(trace), path(human_fasta)
    path kmerseek_timings, stageAs: 'kmerseek_timings/*'

    output:
    path "multiqc_in", emit: sections

    script:
    def primary = params.multiqc_primary_truth
        ? "--primary-truth ${params.multiqc_primary_truth}" : ""
    // `|| true` because grep exits 1 on no match, which set -e would turn into a task
    // failure reported as a missing output rather than as an empty FASTA.
    //
    // This count is n_queries_all -- every sequence in the human FASTA -- and it is the
    // denominator for every arm's queries/s, kmerseek included. It is NOT n_queries_ref,
    // the FoldSeek-intersected subset the accuracy sections use, and the two must never
    // be swapped: dividing one arm by one and another by the other would put tools on
    // silently different axes in the same scatter.
    //
    // The timings directory does not exist when the sweep produced no records (a run with
    // --skip_kmerseek, or one whose store predates the timings output), which the script
    // treats as "no kmerseek rows" rather than as an error.
    """
    set -euo pipefail
    n_queries=\$(grep -c '^>' ${human_fasta} || true)

    build_multiqc_inputs.py \\
        --metrics      ${metrics} \\
        --curves       ${curves} \\
        --trace        ${trace} \\
        --kmerseek-timings kmerseek_timings \\
        --n-queries    \${n_queries} \\
        --max-tools    ${params.multiqc_max_tools} \\
        --max-lines    ${params.multiqc_max_lines} \\
        --top-kmerseek ${params.multiqc_top_kmerseek} \\
        --outdir       multiqc_in ${primary}
    """
}


process multiqcReport {
    /*
     * Pinned by digest like every other third-party image. `make prefetch-images` builds
     * its pull list by grepping these container lines out of this file, so declaring the
     * image here rather than in nextflow.config is what gets it cached on the cluster.
     * multiqc 1.35, for Methods.
     */
    tag "multiqc"
    container 'quay.io/biocontainers/multiqc@sha256:b65e3fe879df27b92334dda0fd987a6e21bdee09a2848551d4f287099a93b7ac'
    publishDir params.outdir, mode: 'copy'

    // The report is the last thing a run does, after every search and every scoring task
    // has already succeeded. Failing the whole run over a plot would throw away a finished
    // sweep, so inside the pipeline this arm is allowed to fail and the run still ends
    // green -- the metrics parquets are published either way.
    //
    // In a `-entry report` run the report IS the work, so there it must fail loudly.
    // params.report_trace is set only by that entry, which makes it the discriminator.
    errorStrategy { params.report_trace ? 'terminate' : 'ignore' }


    input:
    tuple path(sections), path(mqc_config)

    output:
    path "qfo_pfam_region_multiqc.html",       emit: report
    path "qfo_pfam_region_multiqc_data",       emit: data
    path "qfo_pfam_region_multiqc_plots",      emit: plots

    script:
    // Compute nodes here have no outbound internet, so the update check has nothing to
    // reach and only costs a timeout. MPLCONFIGDIR keeps matplotlib's cache inside the
    // task dir; under Apptainer $HOME can be read-only and the plot export needs it.
    """
    set -euo pipefail
    export MPLCONFIGDIR=\$PWD/.mplconfig

    multiqc ${sections} \\
        --config ${mqc_config} \\
        --filename qfo_pfam_region_multiqc.html \\
        --outdir . \\
        --no-version-check \\
        --no-ansi \\
        --force
    """
}


// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")
    def annotations = file(params.annotations)

    // Every DB_CACHE entry is named after its label, and the midi and full runs share one
    // cache on purpose: the TARGET databases are identical between them, which is the
    // entire point of --db_cache. The HUMAN entry is not identical. midi queries 964 chr6
    // proteins and the full run queries 19_696, and both were writing `human_mmdb`,
    // `human_fsdb`, `human_prostt5` and `human.bca` to the same paths in ../results.
    // Whichever ran first won and the other silently reused it, so kmerseek read the
    // correct query FASTA while every database-backed baseline read the other run's --
    // a difference that lands entirely on the baselines' side of the comparison.
    //
    // Keying the human label by a digest of the query FASTA keeps every target database
    // fully shared while giving each query set its own entry. Derived from the file rather
    // than from a --midi/--full flag, because a flag can be passed wrong and this cannot.
    //
    // Written without an explicit byte[] buffer: Nextflow's strict parser rejects array
    // type declarations, and a parse error there stops it analysing the rest of the file,
    // so the mistake hides every other lint finding behind it.
    def digestOf = { f ->
        java.security.MessageDigest.getInstance('MD5')
            .digest(f.bytes)
            .encodeHex().toString().take(8)
    }
    def HUMAN_LABEL = "human-${digestOf(human_fasta)}"
    log.info "  query set: ${human_fasta.name} -> cache label ${HUMAN_LABEL}"

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
    // Sequence-based disorder, computed from the query FASTA before covariates are built.
    // Optional like every other covariate source: when skipped, the sentinel keeps
    // buildQueryCovariates' signature fixed and the disorder_seq stratum is simply absent
    // from the metrics, which the report already handles by not drawing that section.
    // Every proteome: the human query keyed by its query-set digest, and each target by
    // its own label so the entry is shared between the midi and full runs.
    disorder_all = params.skip_metapredict
        ? Channel.empty()
        : proteomeDisorder(
              Channel.of(tuple(HUMAN_LABEL, human_fasta)).mix(species_ch)
          )
    // Covariates take the human one; the target entries are published for the target-side
    // axis and for anyone plotting proteome disorder directly.
    disorder_ch = params.skip_metapredict
        ? Channel.value(file("${projectDir}/assets/NO_DISORDER"))
        : disorder_all.filter { it.name.startsWith("${HUMAN_LABEL}.") }

    cov_in = truth_out.truth.combine(disorder_ch).map { t, dis ->
        tuple(t,
              optional_or(params.hgnc_file,  'NO_HGNC'),
              optional_or(params.omega_file, 'NO_OMEGA'),
              file("${params.structures}/human").exists()
                  ? file("${params.structures}/human")
                  : file("${projectDir}/assets/NO_STRUCTURES"),
              dis)
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

    // Every DB-once arm below follows hhblitsBuildDB: one channel carrying human plus
    // every target, built once each under storeDir, then combined for the pairwise search.
    // storeDir forbids a tuple output, so each process emits a bare directory and the label
    // comes back off its name.
    def label_dbs = { ch, suffix -> ch.map { d -> tuple(d.name - suffix, d) } }

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

        dom_dbs   = label_dbs(mmseqsDomainDb(dom_fa.map { l, fa -> tuple("${l}_domains", fa) }),
                              '_domains_mmdb')
        human_ddb = dom_dbs.filter { l, _d -> l == 'human' }.map { _l, d -> d }
        target_ddb = dom_dbs.filter { l, _d -> l != 'human' }

        identity_ch = parseIdentity(
            domainIdentity(target_ddb.combine(human_ddb)
                                     .map { sp, tdb, qdb -> tuple(sp, qdb, tdb) })
        )
    }



    // ---- kmerseek: alphabet x ksize x species ----
    // How many (tool, variant) arms each species will produce. scoreDomainCalls groups by
    // (truth_set, species), and groupTuple cannot emit a group until it knows the group is
    // complete -- without a size it waits for the WHOLE channel to close, which means no
    // scoring starts until the last kmerseek search finishes.
    //
    // The count is accumulated at the same lines that build the arms, not re-derived from
    // the skip flags afterwards. A second expression restating "phmmer plus jackhmmer plus
    // two mmseqs variants unless bench_only" is a thing that drifts from the code it
    // describes, and the failure is silent in both directions: too high and the group never
    // emits, too low and it emits early and scores a species on a subset of its arms.
    //
    // It is per species, not one number, because the structure arms only run for species
    // that have structures. groupKey carries a size per key, which is what makes that
    // expressible at all.
    def arms_per_species = [:].withDefault { 0 }
    def countArm = { List labels, int n -> labels.each { arms_per_species[it] += n } }

    kmerseek_regions = Channel.empty()
    // Per-task timings for the report. Separate from the trace because this arm is on
    // storeDir: a store hit runs no task and Nextflow records nothing for it.
    kmerseek_timings = Channel.empty()
    // --gpu_benchmark implies --skip_kmerseek. It measures the baselines' GPU path, and
    // forgetting the flag would queue the 3294-job sweep behind a timing run.
    if (!params.skip_kmerseek && !params.gpu_benchmark) {
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

        // Every combo becomes one kmerseekIndex store entry per species, named
        // <species>.<label>.k<ksize>.lc<lowcomp>.kmerseek.rocksdb. Species are unique by
        // construction -- SPECIES is a findAll over ALL_SPECIES, so `--target_species
        // chicken,chicken` still yields one chicken -- but the combo list is not: it comes
        // straight off `--kmerseek_combos` via tokenize(), which happily accepts
        // `wwmj5:16,wwmj5:16`. That builds two tasks pointing at ONE store entry, and the
        // loser dies at unstage with "Directory not empty".
        //
        // This needs no second pipeline and no concurrency to happen. A single run with a
        // repeated combo reproduces it exactly, even under maxForks 1 with the two tasks
        // running seconds apart, because Nextflow checks the store when it creates a task
        // and never re-checks: not when it dispatches it, and not on a retry.
        //
        // Erroring rather than quietly de-duplicating, for the same reason errorStrategy
        // here is 'finish' rather than 'ignore'. A silent dedup would also swallow the
        // case where a repeated combo is the SYMPTOM of a wrong list, and the run would
        // then report on fewer combos than were asked for without saying so.
        def dupCombos = combos
            .countBy { _cli, label, k, lc -> "${label}.k${k}.lc${lc}" }
            .findAll { _key, n -> n > 1 }*.key
        if (dupCombos) {
            error "Duplicate kmerseek combos: ${dupCombos.join(', ')}. Each names one " +
                  "index store entry per species, so a repeat puts two tasks on one entry " +
                  "and the second fails at unstage. Check --kmerseek_combos for repeats."
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

        countArm(SPECIES*.label, combos.size())
        kmerseek_in = species_ch.combine(Channel.fromList(combos))
            .map { species, fasta, cli_flag, label, ksize, lowcomp ->
                tuple(species, fasta, cli_flag, label, ksize, lowcomp)
            }
        idx_out = kmerseekIndex(kmerseek_in)

        // Rejoin the index to the combo that produced it. kmerseekIndex emits a bare
        // directory so storeDir works, which drops the tuple -- the same trade the region
        // parquet already makes below -- so the key is rebuilt from the directory name and
        // joined back against the input channel. That recovers cli_flag and the target
        // FASTA size, neither of which survives in the name, without reparsing either out
        // of a filename that was never meant to carry them.
        def comboKey = { sp, lab, k, lc -> "${sp}|${lab}|${k}|${lc}" }

        combo_meta = kmerseek_in.map { species, fasta, cli_flag, label, ksize, lowcomp ->
            tuple(comboKey(species, label, ksize, lowcomp),
                  species, cli_flag, label, ksize, lowcomp, fasta.size())
        }

        search_in = idx_out
            .map { d ->
                def m = (d.name =~ /^(.+?)\.(.+)\.k(\d+)\.lc(true|false)\.kmerseek\.rocksdb$/)
                if (!m) error "cannot parse kmerseek index directory name: ${d.name}"
                tuple(comboKey(m[0][1], m[0][2], m[0][3], m[0][4]), d)
            }
            .join(combo_meta)
            .map { _key, d, species, cli_flag, label, ksize, lowcomp, target_bytes ->
                tuple(species, cli_flag, label, ksize, lowcomp, target_bytes, d, human_fasta)
            }

        // Rebuild (species, tool, variant) from the filename. The process emits a bare
        // path so storeDir works; the name carries everything the tuple used to.
        ks_out = kmerseekSearch(search_in)
        kmerseek_timings = ks_out.timing
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
        // Which search modes each GPU-capable arm runs. A normal run does one of them;
        // --gpu_benchmark true does both, in one session against one set of databases, so
        // the per-task numbers are comparable without a cross-run correction.
        def search_modes = params.gpu_benchmark ? [false, true] : [params.gpu_search as boolean]

        // --gpu_benchmark narrows the baseline block to the two arms that HAVE a GPU path.
        // Running phmmer, jackhmmer, hhblits, reseek and folddisco again would add hours and
        // measure nothing: none of them accepts a GPU flag, checked against the pinned
        // binaries. See the note on params.gpu_benchmark.
        def bench_only = params.gpu_benchmark

        pair_ch = species_ch.map { species, fasta -> tuple(species, fasta, human_fasta) }

        phmmer_out    = bench_only ? Channel.empty() : phmmerSearch(pair_ch)
        jackhmmer_out = bench_only ? Channel.empty() : jackhmmerSearch(pair_ch)
        countArm(SPECIES*.label, bench_only ? 0 : 2)

        // Both variants share one pair of databases, so createdb runs once per proteome
        // rather than once per (variant, species).
        mm_dbs    = label_dbs(mmseqsDb(species_ch.mix(Channel.of(tuple(HUMAN_LABEL, human_fasta)))),
                              '_mmdb')
        mm_human  = mm_dbs.filter { l, _d -> l == HUMAN_LABEL }.map { _l, d -> d }
        mm_target = mm_dbs.filter { l, _d -> l != HUMAN_LABEL }

        // The padded copy is built only when a GPU mode is actually requested, and only for
        // targets: the GPU prefilter reads the query database unpadded. mm_target already
        // excludes the human entry, so nothing here has to know about HUMAN_LABEL.
        mm_gpu_target = search_modes.contains(true)
            ? label_dbs(mmseqsPaddedDb(mm_target), '_mmdb_gpu')
            : Channel.empty()

        // (species, tdb, qdb, gpu) for every mode, then crossed with the two mmseqs variants.
        mm_pairs = mm_target.combine(mm_human).map { sp, tdb, qdb -> tuple(sp, tdb, qdb, false) }
        if (search_modes.contains(true)) {
            mm_pairs = mm_pairs.mix(
                mm_gpu_target.combine(mm_human).map { sp, tdb, qdb -> tuple(sp, tdb, qdb, true) }
            )
        }
        // Not redundant: the CPU branch above is always constructed, so this is what drops
        // it again for a GPU-only run (--gpu_search true without --gpu_benchmark).
        mm_pairs = mm_pairs.filter { _sp, _t, _q, gpu -> search_modes.contains(gpu) }

        // The CPU spelling of both the variant label and the output filename is EXACTLY
        // what it was before the GPU arm existed, so a resume, a published region file and
        // an existing metrics row all still match. Only the GPU arm gets new names.
        //
        // The GPU variant is not called s<n>: --gpu 1 replaces the k-mer prefilter with the
        // ungapped one and -s stops having any effect, so a label saying s7 would name a
        // setting that did not apply. It names the prefilter that actually ran.
        def mm_variant = { gpu -> gpu ? "gpu_ungapped" : "s${params.mmseqs2_sensitivity}" }
        def mm_name    = { sp, v, gpu -> "human_vs_${sp}.${v}${gpu ? '.gpu' : ''}.tsv.gz" }

        mmseqs_in = mm_pairs.flatMap { species, tdb, qdb, gpu ->
            [
                tuple(species, "mmseqs2_seqseq", 1, gpu,
                      mm_variant(gpu), mm_name(species, "mmseqs2_seqseq", gpu), tdb, qdb),
                tuple(species, "mmseqs2_iterative", params.mmseqs2_iterations, gpu,
                      mm_variant(gpu), mm_name(species, "mmseqs2_iterative", gpu), tdb, qdb),
            ]
        }
        mmseqs_out = mmseqs2Search(mmseqs_in)
        countArm(SPECIES*.label, 2 * search_modes.size())   // seqseq + iterative, per mode

        // HHblits: build the human query profile DB once, every species target DB once.
        // is_query travels as an explicit flag rather than being inferred from the label.
        // The script used to test `label != "human"` to decide whether to build hhm+cs219,
        // which silently stops meaning anything once the human label carries a digest.
        hhblits_out = Channel.empty()
        if (!bench_only) {
            all_for_hhdb = species_ch.map { l, f -> tuple(l, f, false) }
                .mix(Channel.of(tuple(HUMAN_LABEL, human_fasta, true)))
            hhdb_ch      = label_dbs(hhblitsBuildDB(all_for_hhdb), '_hhdb')
            human_hhdb   = hhdb_ch.filter { label, _db -> label == HUMAN_LABEL }.map { _label, db -> db }
            species_hhdb = hhdb_ch.filter { label, _db -> label != HUMAN_LABEL }

            hhblits_out = hhblitsSearch(species_hhdb.combine(human_hhdb))
            countArm(SPECIES*.label, 1)
        }

        baseline_regions = phmmer_out.mix(jackhmmer_out).mix(mmseqs_out).mix(hhblits_out)

        // ---- ProstT5: 3Di predicted from sequence, no structures on either side ----
        // Deliberately OUTSIDE the structure guard below. This is the arm that can run
        // where AlphaFold coverage is too thin for Foldseek or Reseek, which is         // the regime the invertebrate claim lives in.
        if (!params.skip_prostt5 && !bench_only) {
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

            // Human plus every target, each encoded exactly once. storeDir means a rerun
            // or a second species list reuses them rather than re-running the model.
            p5_dbs = prostt5Db(
                Channel.of(tuple(HUMAN_LABEL, human_fasta))
                    .mix(species_ch)
                    .combine(weights)
            )
            // storeDir forbids a tuple output, so the label comes back off the directory
            // name -- same trick kmerseekIndexAndSearch uses for its filenames.
            p5_labeled  = p5_dbs.map { d -> tuple(d.name - '_prostt5', d) }
            p5_human    = p5_labeled.filter { l, _d -> l == HUMAN_LABEL }.map { _l, d -> d }
            p5_targets  = p5_labeled.filter { l, _d -> l != HUMAN_LABEL }

            prostt5_out = prostt5Search(p5_targets.combine(p5_human))
            baseline_regions = baseline_regions.mix(prostt5_out.regions)
            countArm(SPECIES*.label, 1)
        }

        // ---- foldseek ----
        // Foldseek and Folddisco need actual structure files. An empty or missing
        // directory makes both fail deep inside a container with no usable message --
        // folddisco index exits 1 printing nothing at all -- so the emptiness is checked
        // here, where it can name the cause and the fix. `make sync-data` deliberately
        // does NOT ship structures (they are ~36 GB); `make sync-structures` does.
        // toRealPath() is load-bearing. Nextflow's file(...).list() does NOT follow a
        // symlink to a directory -- it returns the LINK'S OWN NAME, so a perfectly good
        // structures/mouse -> /somewhere/mouse listed as ["mouse"], matched nothing, and
        // this reported "no structures" for a directory full of them. Verified directly:
        // list() gives [mouse], toRealPath().list() gives the AF-*.cif files.
        def has_structs = { label ->
            def d = file("${params.structures}/${label}")
            if (!d.exists()) return false
            def real = file(d.toRealPath().toString())
            real.list().any { it ==~ /(?i)^AF-.*\.cif(\.gz)?$/ }
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
            // Human plus every species with structures, each converted once per tool.
            struct_ch = Channel.fromList(
                struct_species.collect { s -> tuple(s.label, file("${params.structures}/${s.label}")) }
            ).mix(Channel.of(tuple(HUMAN_LABEL, human_structs)))

            fs_dbs    = label_dbs(foldseekDb(struct_ch), '_fsdb')
            fs_human  = fs_dbs.filter { l, _d -> l == HUMAN_LABEL }.map { _l, d -> d }
            fs_target = fs_dbs.filter { l, _d -> l != HUMAN_LABEL }

            // Targets only, and only when a GPU mode is requested -- see mmseqsPaddedDb.
            fs_gpu_target = search_modes.contains(true)
                ? label_dbs(foldseekPaddedDb(fs_target), '_fsdb_gpu')
                : Channel.empty()

            // CPU spelling unchanged, as for mmseqs above.
            def fs_variant = { gpu -> gpu ? "3di_aa_gpu" : "3di_aa" }
            def fs_name    = { sp, gpu -> "human_vs_${sp}.foldseek${gpu ? '.gpu' : ''}.tsv.gz" }

            fs_in = fs_target.combine(fs_human).map { sp, tdb, qdb ->
                tuple(sp, false, fs_variant(false), fs_name(sp, false), tdb, qdb)
            }
            if (search_modes.contains(true)) {
                fs_in = fs_in.mix(
                    fs_gpu_target.combine(fs_human).map { sp, tdb, qdb ->
                        tuple(sp, true, fs_variant(true), fs_name(sp, true), tdb, qdb)
                    }
                )
            }
            // Drops the always-constructed CPU branch for a GPU-only run -- see mmseqs above.
            fs_in = fs_in.filter { _sp, gpu, _v, _n, _t, _q -> search_modes.contains(gpu) }

            foldseek_out     = foldseekSearch(fs_in)
            baseline_regions = baseline_regions.mix(foldseek_out)
            countArm(struct_species*.label, search_modes.size())

            // ---- Reseek: same structures, opposite alphabet direction ----
            // No GPU path exists. The pinned reseek image links no CUDA library, its usage
            // text names no device flag and the binary contains no CUDA symbols, so there is
            // nothing to enable and --gpu_benchmark leaves it out rather than timing a
            // CPU-only tool twice.
            if (!params.skip_reseek && !bench_only) {
                // Human goes through the same cached conversion as the targets; it used to
                // re-parse the whole human structure directory on every search.
                // Only the TARGET side needs a Mu fasta: -dbmu prefilters the database,
                // and the query is read straight from its .bca. Human's mu is built anyway
                // (one process, one output shape) and simply dropped here.
                rs_db     = reseekConvert(struct_ch)
                rs_human  = rs_db.filter { l, _b, _m -> l == HUMAN_LABEL }.map { _l, b, _m -> b }
                rs_target = rs_db.filter { l, _b, _m -> l != HUMAN_LABEL }

                reseek_out = reseekSearch(rs_target.combine(rs_human))
                baseline_regions = baseline_regions.mix(reseek_out)
                countArm(struct_species*.label, 1)
            }

            // ---- folddisco ----
            // Also CPU-only: no GPU flag on `folddisco query`, no CUDA symbols in the
            // binary, no CUDA library linked. Left out of --gpu_benchmark for the same
            // reason as reseek.
            if (!params.skip_folddisco && !bench_only) {
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
                countArm(struct_species*.label, 1)
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

    // One task per (truth_set, species) rather than per (truth_set, species, tool, variant).
    // The per-arm shape was ~10_100 SLURM jobs for a full sweep, each a few seconds of work
    // behind minutes of scheduler latency, and Sherlock rate-limits submission per hour --
    // a run died with "Reached jobs per hour limit" partway through. Grouped, it is 27.
    //
    // truth, domain_map, covariates and identity are identical within a group, so .first()
    // on each is exact rather than an approximation; only tools, variants and regions vary.
    // Each species' own proteome disorder, keyed by the filename proteomeDisorder writes.
    // Falls back to the sentinel per species rather than globally, so a species whose
    // prediction is missing loses only its own target-side axis.
    tdis_by_species = params.skip_metapredict
        ? Channel.empty()
        : disorder_all.map { f -> tuple(f.name.replaceAll(/\.disorder_metapredict\.parquet$/, ''), f) }

    score_in = score_in
        .map { ts, sp, tool, variant, mya, regions, truth, dm, cov, ident ->
            tuple(sp, ts, tool, variant, mya, regions, truth, dm, cov, ident)
        }
        .combine(
            params.skip_metapredict
                ? Channel.fromList(SPECIES.collect {
                      tuple(it.label, file("${projectDir}/assets/NO_DISORDER")) })
                : tdis_by_species,
            by: 0
        )
        .map { sp, ts, tool, variant, mya, regions, truth, dm, cov, ident, tdis ->
            tuple(ts, sp, tool, variant, mya, regions, truth, dm, cov, ident, tdis)
        }

    score_grouped = score_in
        .map { ts, sp, tool, variant, mya, regions, truth, dm, cov, ident, tdis ->
            // groupKey carries the expected size WITH the key, so each (truth_set, species)
            // group is released the moment its own arms are all in rather than when the
            // whole channel closes. Without it, no scoring could start until the last
            // kmerseek search of the last species finished.
            tuple(groupKey(tuple(ts, sp), arms_per_species[sp]),
                  tool, variant, mya, regions, truth, dm, cov, ident, tdis)
        }
        // remainder: true is the safety net for the count being WRONG. If arms_per_species
        // over-counts, the group never reaches its size and would hang forever; with
        // remainder it is released at channel close instead, which is exactly the
        // behaviour this change replaces. An under-count still emits early, which is why
        // the count is accumulated beside the arms rather than restated.
        .groupTuple(by: 0, remainder: true)
        .map { key, tools, variants, myas, regions, truths, dms, covs, idents, tdiss ->
            tuple(key[0], key[1], tools, variants, myas.first(), regions,
                  truths.first(), dms.first(), covs.first(), idents.first(), tdiss.first())
        }

    log.info "  scoring : one task per (truth set, species); arms per species = " +
             "${arms_per_species.collect { k, v -> "${k}:${v}" }.join(', ')}"

    scored = scoreDomainCalls(score_grouped)

    // ---- hmmscan annotation ceiling ----
    ceiling_metrics = Channel.empty()
    ceiling_curves  = Channel.empty()
    // The annotation ceiling is a property of Pfam-A, not of any search arm, so a GPU/CPU
    // timing run has no use for it and would pay for a whole-proteome hmmscan to learn
    // nothing new.
    if (params.run_hmmscan && !params.gpu_benchmark) {
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

    agg = aggregateMetrics(
        scored.metrics.mix(ceiling_metrics).collect(),
        scored.curve.mix(ceiling_curves).collect(),
    )

    if (!params.skip_multiqc) {
        // ifEmpty([]) rather than a bare collect(): collect() on an empty channel emits
        // nothing at all, which would leave buildMultiqcInputs with an input channel that
        // never fires and drop the whole report on any run without kmerseek timings.
        multiqcFromMetrics(agg.metrics, agg.curves, human_fasta,
                           kmerseek_timings.collect().ifEmpty([]))
    }
}


// ---------------------------------------------------------------------------
// One HTML report. Shared by the main workflow and the `report` entry, so a rebuild from
// published results goes through exactly the code path the pipeline itself used.
// ---------------------------------------------------------------------------
workflow multiqcFromMetrics {
    take:
    metrics
    curves
    human_fasta
    kmerseek_timings

    main:
    mqc_in = metrics
        .combine(curves)
        // resolveTrace() runs when this fires, which is after aggregateMetrics finished.
        .map { m, c -> tuple(m, c, resolveTrace(), file(human_fasta)) }

    sections = buildMultiqcInputs(mqc_in, kmerseek_timings).sections
    multiqcReport(sections.combine(Channel.of(file(params.multiqc_config))))
}


// ---------------------------------------------------------------------------
// `nextflow run main.nf -entry report` — rebuild the report from an outdir that already
// holds aggregated metrics, without re-running a single search.
//
// This is the normal way to get the report after a long sweep: the trace is only complete
// once the run has ended, and rerunning the whole pipeline to pick it up would be absurd.
// Point --trace_file at the run whose resource numbers you want.
// ---------------------------------------------------------------------------
workflow report {
    def outdir  = file(params.outdir)
    def metrics = file("${outdir}/all_domain_metrics.parquet")
    def curves  = file("${outdir}/all_domain_curves.parquet")

    if (!metrics.exists()) {
        error """
        |No aggregated metrics at ${metrics}.
        |The report is built from what aggregateMetrics published, so the pipeline has to
        |have run first. Point --outdir at the run you want reported on.
        """.stripMargin()
    }
    if (!curves.exists()) {
        log.warn "No ${curves} -- the PR and ROC sections will be skipped."
    }

    // Without --report_trace this run writes its own trace, and anything found in the
    // launch directory is that empty file rather than the run being reported on. Passing
    // the flag is what turns this run's trace observer off, so it is required rather than
    // inferred -- there is no way to read a trace the observer is also writing.
    //
    // An error, not a warning: a warning would be read after the resource sections had
    // already come out blank.
    if (!params.report_trace) {
        error """
        |No --report_trace given.
        |
        |The report's time, CPU and memory sections come from the trace of the run being
        |reported on, and Nextflow truncates any trace file it is itself writing. Naming
        |the earlier run's trace with --report_trace is what switches this run's trace
        |observer off, so it has to be explicit:
        |
        |    nextflow run main.nf -entry report -profile <profile> \\
        |        --outdir <outdir> --report_trace run/qfo_pfam_region.<date>.trace.txt
        |
        |or just `make multiqc`, which fills it in from the newest trace under run/.
        |To build the accuracy sections alone, pass --report_trace none.
        """.stripMargin()
    }
    def trace = resolveTrace()

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")

    // kmerseek's timings come off disk here, not off a channel, because no kmerseek
    // process runs in this entry. They sit in the search storeDir next to the regions
    // parquet they describe, which is the whole reason they survive a run that executed
    // nothing: `-entry report` reads exactly the files a store hit would have served.
    def ks_timings = file("${outdir}/kmerseek").exists()
        ? file("${outdir}/kmerseek/*.timings.jsonl")
        : []
    log.info "building the report from ${outdir}, trace ${trace}, " +
             "${ks_timings.size()} kmerseek timing records"

    multiqcFromMetrics(
        Channel.of(metrics),
        Channel.of(curves.exists() ? curves : file("${projectDir}/assets/NO_CURVES")),
        human_fasta,
        Channel.of(ks_timings),
    )
}
