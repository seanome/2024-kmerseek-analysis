#!/usr/bin/env nextflow
/*
 * How much of a proteome does conventional sequence search miss?
 *
 * The dark set is every protein in the query proteome that phmmer, jackhmmer and mmseqs2
 * ALL fail to place into the reference at default cutoffs. It is the denominator of the
 * whole proteome-annotate claim: if phmmer already finds the homolog, kmerseek finding it
 * too is not a result. Only the dark set is territory kmerseek can add anything in.
 *
 * The point of this pipeline is that the dark set needs NO answer key. C. gigas has 16
 * reviewed Swiss-Prot entries and Botryllus has 0, so neither can be scored until the
 * structural key exists -- but "no arm hit this protein" is a property of the searches
 * alone. So this runs now, while Chainsaw and InterProScan are still going, and it decides
 * whether the expensive half is worth running at all.
 *
 * THE LADDER. A dark fraction on Botryllus alone has no scale. 45% dark is a lot or a
 * little depending on what the same three arms leave dark on mouse, on fly, on yeast --
 * species whose annotation is deep enough that a dark protein there is much more likely
 * to be a real miss than a spurious gene model. So --species takes a comma-separated
 * list, and the same pipeline runs each QfO species of the midi-plus ladder as the query
 * against reviewed Swiss-Prot minus ITS OWN clade, exactly the construction Botryllus
 * got. One reference per excluded clade, one dark set and one report per species.
 *
 * Two entry points, because they cost very different amounts:
 *
 *   -entry darkSet     reference + three sequence arms + the count. Cheap, no kmerseek,
 *                      no index. This is the number that comes back first.
 *   --with_kmerseek    adds kmerseek on top: a named list of alphabet:ksize combos, or
 *                      with --kmerseek_sweep the whole alphabet x ksize matrix, which is
 *                      what says which alphabet and k to read the Botryllus number at.
 *
 * DIRECTION. Every other pipeline here has human as the query and a proteome as the target.
 * This one is the deployment direction: the query proteome is the QUERY and reviewed
 * Swiss-Prot minus its clade is the reference. kmerseek indexes the TARGET, so the index
 * here is built over ~572_700 reference sequences rather than one proteome -- which is why
 * the mini run is an index-cost probe rather than a small search.
 */

nextflow.enable.dsl = 2

include { compareDarkLengths } from './modules/length.nf'

include { darkSetDisorder } from './modules/disorder.nf'

// The alphabet x ksize matrix, shared with qfo-pfam-region-benchmark so the two pipelines
// sweep one table. keyspaceBits is what sizes both kmerseek processes' memory.
include { allEncodings; extraEncodings; knownEncodings; expandEncodings; keyspaceBits } \
    from '../shared/kmerseek_encodings'

def home = System.getProperty('user.home')

// One species, or a comma-separated list: botryllus, or
// mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli. Every name is a row in
// the registry with annotate_query set; the row supplies the clade to remove from the
// reference and where the proteome is.
params.species        = null
params.registry       = "${projectDir}/../../data/species_metadata.json"
params.qfo_dir        = "${home}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.swissprot_dat  = "${home}/data/uniprot/uniprot_sprot.dat.gz"
params.outdir         = "${home}/data/invertebrate-dark-set/results"
params.reference_cache = null         // shared across species; see buildReference
// Single-species overrides. Both are refused with more than one species, because there
// is no way to say which one they mean.
params.query_fasta    = null          // overrides the registry path
params.exclude_clade  = null          // overrides the registry default rung

params.evalue_report        = 10.0
params.evalue_call          = 1e-3    // what counts as "found something" for the dark set
params.jackhmmer_iterations = 3
params.mmseqs2_sensitivity  = 7

// Queries per task. 45_339 botryllus proteins against 572_700 reference sequences is
// ~1-3 s/query for phmmer and several times that for jackhmmer at 3 iterations, so the
// whole proteome as one task is 12-38h for the cheapest arm and well past that for the
// dearest -- past the 12h walltime, and a scheduler kill reports exitStatus
// Integer.MAX_VALUE, so it would retry into the same wall. 2_000 gives 23 tasks that each
// finish inside an hour or two.
params.query_chunk_size = 2000

// alphabet:ksize pairs. Both defaults are arms with a measured identity-axis row, so a
// zero here can be read against a known baseline rather than guessed at. hp_pbotc is the
// designated best HP arm but was renamed in PR #43 and its post-rename CLI flag has not
// been resolved against the binary, so it is not defaulted in -- add it once confirmed.
params.kmerseek_alphabets = 'hp_thomas_dill2:23,protein20:10'

// The whole matrix instead of the named pairs: every alphabet in
// ../shared/kmerseek_encodings.nf over its entropy-derived ksize range, 183 combos, plus
// polarity4 and funcgroups8 (20 more) when --kmerseek_extra_encodings is on. This is
// the run that says which alphabet and k the dark-set number should be read at; the
// two named defaults above are for reading ONE species against an arm that already has
// an identity-axis row. --kmerseek_encodings names a subset of alphabets and keeps each
// one's full ksize range, for a run scoped to the HP family alone.
params.kmerseek_sweep           = false
params.kmerseek_extra_encodings = false
params.kmerseek_encodings       = null
// alphabet:ksize pairs appended to a sweep's table, for combos the table's k ranges
// leave out. The 0.4 ladder wants the two-letter HP alphabets at k=12, six below the
// table's HP floor, because only an extension arm can use k-mers that short. Ignored
// unless --kmerseek_sweep or --kmerseek_encodings is on; without a sweep,
// --kmerseek_alphabets already names every pair outright.
params.kmerseek_sweep_plus      = ''

// The low-complexity mask runs ON and OFF as a PAIR by default, not as a sweep dimension.
// BHF's seven flagship matches included polar-biased low-complexity segments (ZNF292
// pppphpppppppphhppp), so a dark-set hit that does not survive masking is not a finding.
// Reporting one setting alone makes that unfalsifiable. A sweep may narrow this to
// `false`: the region benchmark's own ON/OFF sweep did not move its metrics, and the
// Botryllus pair lost 0 (HP) and 4 (protein20) dark proteins to the mask, so on the
// ladder the pair doubles the search count to guard against something two measurements
// have already bounded. The pair panel in the report is simply omitted when only one
// side ran; the sweep panel does not need it.
params.kmerseek_lowcomp = 'true,false'

params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_query_pvalue = 0.05
params.min_region_score = 1.3
params.index_cache        = null
// Where the sequence arms' hits are stored and looked up (see phmmerSearch); defaults
// to this run's --outdir, and a second run points it at the first run's.
params.hits_cache         = null
params.with_kmerseek      = false

// kmerseek 0.4 arms. Every value below is a list, and the arms are the product.
//
// --scaled is an INDEX-time setting (FracMinHash: keep the k-mers whose hash falls in
// the lowest 1/N of the hash space), so it is a dimension of the index, named in the
// index directory, and search reads it back. Indexing memory and index size fall almost
// linearly with N; what it costs is a short match none of whose k-mers survive.
params.kmerseek_scaled = '1'
// Extension is a SEARCH-time setting over the same index: `exact` is a region as a
// maximal run of shared k-mers, `extend` grows each run outward through mismatches
// (X-drop) and ranks it with a Karlin-Altschul E-value, chaining colinear runs.
params.kmerseek_extension = 'exact'
// Mismatch penalties C for the `extend` arms, one search arm per value. A number is used
// as given; `opt` is the alphabet's own optimum, C_best = -ln(1 - kappa) /
// ln(1 + kappa / f - kappa) with equal class shares f = 1 / classes (the log-odds
// scoring behind BLOSUM, Altschul 1991, written for a +1 / -C scheme; see
// docs/kmerseek_evalue_explainer, equation 2b). kappa is the copy rate measured in
// aligned Pfam pairs at 20-30% identity (analysis notebook 230), in kappaTable below. With
// two letters a match is weak evidence and a mismatch must count for a lot (C 1.63);
// with twenty a match is strong evidence and a mismatch is what most positions of a
// true pair look like (C 0.14). 2 is the value every kmerseek #54 benchmark used.
params.kmerseek_mismatch_penalty = '2'
// The X-drop scales with C so that the walk still ends after the same run of
// mismatches: X = xdrop_per_penalty x C, 8 at C = 2.
params.kmerseek_extend_xdrop_per_penalty = 4
params.kmerseek_chain_max_gap            = 30
params.kmerseek_chain_max_shift          = 10
// An extended region's E-value needs lambda and K fitted on ITS index. The fit runs at
// index time on this many of the reference's own sequences and is stored in the index,
// so every search chunk reads the same fit; 0 skips it, and an `extend` search would
// then have to fit its own. Only done when an `extend` arm is wanted.
params.kmerseek_ka_queries = 500
// For the calibrateStore entry only: which penalties to fit on every stored index;
// which clades' indexes (comma list, empty = every clade in the store); and a regex
// naming indexes to leave alone (e.g. the k=12 HP pairs a sweep no longer runs).
params.calibrate_penalties = 'opt'
params.calibrate_clades    = ''
params.calibrate_exclude   = ''
// The dark-gain step counts a dark protein as reached when ANY region lands on it, which
// saturates (see kmerseek_dark_gain.py). An `extend` arm also carries region_evalue, so
// it is counted again at each of these cutoffs; an `exact` arm has no E-value and is
// counted at none.
params.kmerseek_evalue_max = '0.01,0.0001'
// Landmark pairs, species:query:target, one per entry. A pair the report tracks by name
// across every arm: did the sequence arms and did each kmerseek arm put this query on
// this target, and how well. Human BCL2 (P10415) and C. elegans CED-9 (P41958) are the
// classic remote Bcl-2 family pair (about 25% identical, the same fold), one of the
// cases this project exists for; each is searched from its own species against the
// reference that still holds the other. Borrelia P66 (H7C7N8) mimics human CD47's
// (Q08722) SIRPalpha-binding face without sequence similarity (Tal et al. 2024): P66 is
// the query from its own proteome, CD47 is in Swiss-Prot minus Spirochaetia. A target of
// `*` means every target: the arm-by-arm table then lists what each arm finds for the
// query at all, which is what the Botryllus histocompatibility factor BHF
// (FUN008084_FUN008084; BLAST 0, HMMer 0) needs, having no named partner. Accessions
// are bare, as splitQuery and buildReference write every header.
params.landmarks = 'human:P10415:P41958,worm:P41958:P10415,borrelia:H7C7N8:Q08722,botryllus:FUN008084_FUN008084:*'

// Memory for the two kmerseek processes, sized per task rather than as a flat ladder.
// The index is sized from the keyspace; the SEARCH is sized from the index's own k-mer
// spectrum, which every kmerseekIndex task writes and which is the only thing that
// predicted the ladder run's peaks (see kmerseekSearchMemory). These are the knobs.
//
// Retries multiply the first ask by kmerseek_memory_retry_factor per attempt, up to
// kmerseek_memory_max. hns has 121 nodes with 192-256 GB and 15 bigmem nodes with
// 1-1.5 TB; ask for more than any node has and SLURM rejects the job outright rather
// than queueing. kmerseek_memory_first_max used to hold the first ask at 250 GB so it
// could land on the small nodes, with the bigmem nodes reserved for retries. The ladder
// run (amazing_koch, 2026-09-16) showed what that costs: gbmr7 k11-12 asked the 250 GB
// cap, died, asked 375 GB, died again, and only the third attempt could reach 500 GB --
// three queue waits and up to 12 h of walltime for one search. Both caps are now the
// ceiling: a task the model says needs a bigmem node goes there on its first attempt.
// The retry factor is 2, not 1.5, for the same reason: the retries that completed in
// that run peaked at 1.09-1.49x their first ask, and the 43 that died a second time
// needed more than 1.5x, so a 1.5x step bought one more queue wait and nothing else.
params.kmerseek_memory_first_max    = '500 GB'
params.kmerseek_memory_max          = '500 GB'
params.kmerseek_memory_retry_factor = 2.0
params.kmerseek_search_memory_floor = '24 GB'
params.kmerseek_index_memory_floor  = '48 GB'
// The Karlin-Altschul fit that `kmerseek index` runs when an extend arm is wanted is a
// search of --ka-queries reference sequences against the index just built, and once
// the k-mer space is crowded that search, not the build, sets the task's peak. Peak RSS
// of whole builds in the ladder-0.4 run (2026-09-20, 8 CPUs, 500 queries):
//   2-letter k19 (19 keyspace bits), scaled 1 / 2 / 5 / 10:  44 / 25 / 14 / 10 GB
//   2-letter k23 (23 bits):                                  42 / 24 / 12 /  8 GB
//   protein20 k10 (43 bits):                                 18 /  9 /  4 /  2 GB
//   2-letter k12 (12 bits): scaled 10 = 61-70 GB, scaled 5 = 106 GB; scaled 1 and 2
//     never finished, cgroup-killed at the 74 GB ask and the 149 GB retry (67 kills,
//     every OOM in that run). The keyspace model above asked 73 GB for them because
//     12 bits fills nothing -- the build IS small; the fit is what needs the memory.
// Fit peak ~ 300 x 2^(-(bits - 12) / 2.5) x scaled^-0.7 GB: 43 GB at 19 bits (measured
// 44), 300 at 12 bits and scaled 1 (extrapolated along the scaled curve), 60 at 12 bits
// and scaled 10 (measured 61-70). Scaling with the query count is assumed linear.
params.kmerseek_index_build_gb              = 45
params.kmerseek_ka_fit_memory_gb_at_12_bits = 300
params.kmerseek_ka_fit_memory_halving_bits  = 2.5
params.kmerseek_ka_fit_memory_scaled_power  = 0.7
// Search-memory model, peak GB = headroom x (base + slope x sqrt(load)); see
// kmerseekSearchMemory for what load is and where these numbers come from.
params.kmerseek_search_memory_base     = 36
params.kmerseek_search_memory_slope    = 2.34
// 1.5 is what the ladder sweep ran with (passed on the command line on 2026-09-16), and
// the asks the per-alphabet factors below were measured against; the audited failure
// rates and peak/ask ratios are all relative to headroom 1.5, so the two numbers move
// together. 2.0 was the earlier default and over-asked for the alphabets that never die.
params.kmerseek_search_memory_headroom = 1.5
// A per-alphabet multiplier on top of the model. The spectrum load is an INDEX-side
// number and cannot see how many hits a query chunk will materialise, and that is where
// the model was wrong: in the ladder run (6_143 first attempts audited on 2026-09-17,
// requested --mem against peak_rss) the first attempt was cgroup-killed on 74% of gbmr7
// searches, 68% of hp_kyte_doolittle2, 42% of hp_thomas_dill_no_c2, 34% of gbmr4 and 32%
// of hp_thomas_dill2, against 0-1% for protein20, uniprot18 and hp_lehninger2. Tasks that
// did complete used a median 63% (p90 87%) of their ask, so the model is not padded, it
// is mis-shaped for those alphabets. Those asks were headroom 1.5 x the model, which is
// the default now. The factors are the smallest step of 1.25 above the worst peak/ask
// ratio measured for each alphabet; where most of the alphabet's tasks
// still had no completed attempt (gbmr7 193 of 268, hp_kyte_doolittle2 226 of 318) the
// ratio is censored at the 1.5x retry that also died, and the factor is 2. An alphabet
// not listed gets 1.0.
params.kmerseek_search_memory_alphabet_factor = [
    gbmr7: 2.0, hp_kyte_doolittle2: 2.0, hp_thomas_dill_no_c2: 2.0, gbmr4: 2.0,
    hp_thomas_dill2: 1.5, dayhoff6: 1.5, mmseqs12: 1.5, sdm12: 1.5, wwmj5: 1.5, hsdm17: 1.5,
    hp_lehninger_hpc3: 1.25, hp_pbotc_1st_ed2: 1.25, wass14: 1.25,
]
// A combo whose PREDICTED median peak, times this, is above kmerseek_memory_max is not
// searched at all; see the skip in the workflow. 2.0 is the worst chunk-to-median ratio
// the ladder run measured (2.54, zebrafish and fly) rounded down, so a combo that passes
// has a real chance on its last attempt rather than a certainty of ending the run.
params.kmerseek_skip_factor = 2.0

// Length of the dark proteins against the placed ones. On by default and cheap -- it reads
// the query FASTA and the dark parquet and nothing else -- because the dark fraction should
// not be quoted without it. Botryllus is a 2026 annotation, and a new gene set's tail of
// fragments and spurious ORF calls is dark for reasons that have nothing to do with
// homology detection being hard. Junk models are short, so this says whether shortness is
// what is inflating the number.
params.with_length_comparison = true

// Disorder is ON by default, unlike kmerseek. It is one metapredict pass over the query
// proteome with no index to build, so it costs a rounding error next to the three search
// arms -- and it is a check on the dark set's own composition, which every reading of the
// dark count depends on. See modules/disorder.nf.
params.with_disorder      = true

// The MultiQC report. On by default: it costs one python task and one multiqc task at the
// end of a run that has already done every search, and a run whose numbers exist only as
// JSON on a scratch filesystem is a run nobody reads.
params.with_multiqc        = true
params.multiqc_dark_config = "${projectDir}/assets/multiqc_dark_config.yaml"

// Set only by `-entry darkReport`, where the report IS the work and a failure has to be
// loud. Inside a full run the report is the last step after every search has already
// succeeded, so failing the whole run over a plot would throw away a finished proteome --
// there it is allowed to fail and the run still ends green. The errorStrategy that reads
// this lives in nextflow.config, not in the process body: a config-level setting always
// beats a directive declared in the process.
params.report_only        = false

// The report lives in its own module file. main.nf is where every other arm is also being
// added, and a report process defined here would put three sets of edits in one hunk.
include { darkReportFrom } from './modules/report'

HMMER   = 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
MMSEQS  = 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'

// --- kmerseek memory ---------------------------------------------------------------------
//
// INDEX: grows with how many DISTINCT k-mers the reference has, which the keyspace caps.
// Measured on the ladder run of 2026-09-14 (495 builds, 572k-sequence references): 51-57
// GB below 20 bits, rising to a plateau of 74-83 GB past ~40 bits, once every one of the
// reference's ~200M k-mers is its own key. Floor plus a term that saturates near 28 bits,
// times 1.4 -- the spread across builds of one combo is small because nothing about the
// index depends on the query.
//
// SEARCH: the other way round, and NOT a function of the keyspace. A small keyspace means
// each query k-mer matches many reference k-mers, and the per-query match set is what
// fills memory -- but how many it matches depends on how SKEWED the reference's k-mer
// spectrum is, not on how many keys exist. gbmr7 puts G and P alone and 40% of residues
// in one class, so at k=9 its most common k-mer occurs 380_000 times in the reference;
// the entropy-bits model that sized the first ladder run put gbmr7 k16 at 29 GB and it
// died three times at 65. That run OOM-killed 259 of 1_057 searches, every one in a
// skewed alphabet at low k (gbmr7, gbmr4, hp_kyte_doolittle2, hp_lehninger_hpc3, and the
// kmin of everything else), and ended the run.
//
// What did predict the 618 uncapped peaks (r2 0.60, against 0.56 for the best keyspace
// model) is the LOAD of the index's own spectrum: the expected posting-list length hit by
// a random reference k-mer, sum(occ^2 x n_kmers) / sum(occ x n_kmers), read from the
// spectrum.csv.gz that kmerseekIndex writes into every index. peak ~ 36 + 2.34 x
// sqrt(load) GB at the median; the chunk-to-median ratio ran up to 2.5 (zebrafish, fly),
// and worm's chunks sit 1.6x over the cross-species median throughout, so the first ask
// carries 2.0x headroom and the retries take the rest. The mask ON arm sits ~20% under
// mask OFF. Peaks that came out exactly at the 24 GB floor are the cgroup limit read back
// as RSS -- RocksDB's file-backed pages fill whatever is allowed -- not a need for 24 GB.
//
// The spectrum is read once per index, in the workflow's skip filter, and cached BY INDEX
// NAME. The name is the key on purpose: inside a process directive a path input is a
// TaskPath that knows only its staged name (toAbsolutePath() throws
// UnsupportedOperationException, resolve() is relative to the staged name), so the
// memory closure cannot open the file itself. The filter sees the real path, runs before
// any search task is created, and leaves the number here for the closure to look up.

def SPECTRUM_LOAD = java.util.Collections.synchronizedMap([:])

def spectrumLoad = { Path index_dir ->
    def key = index_dir.name
    def hit = SPECTRUM_LOAD[key]
    if (hit != null) return hit
    def f = index_dir.resolve('spectrum.csv.gz')
    if (!f.exists()) {
        error "no spectrum.csv.gz inside ${index_dir}: kmerseekIndex writes one into every " +
              "index and the search memory model reads it. An index without one was built " +
              "by something else."
    }
    double num = 0.0d, den = 0.0d
    new java.util.zip.GZIPInputStream(f.newInputStream()).withReader('UTF-8') { r ->
        r.eachLine { line ->
            if (line.startsWith('#') || line.startsWith('moltype')) return
            def c = line.split(',')
            double occ = c[2] as double, n = c[3] as double
            num += occ * occ * n
            den += occ * n
        }
    }
    double load = den > 0 ? num / den : 0.0d
    SPECTRUM_LOAD[key] = load
    load
}

// The model's median prediction, before headroom -- the number the skip decision uses.
def searchMedianGb = { double load, String lowcomp ->
    double base  = params.kmerseek_search_memory_base as double
    double slope = params.kmerseek_search_memory_slope as double
    (base + slope * Math.sqrt(load)) * (lowcomp == 'true' ? 0.8d : 1.0d)
}

def memoryLadder = { double firstGb, int attempt ->
    long firstCapMb = MemoryUnit.of(params.kmerseek_memory_first_max).toMega()
    long capMb      = MemoryUnit.of(params.kmerseek_memory_max).toMega()
    double f        = params.kmerseek_memory_retry_factor as double
    long first      = Math.min(firstCapMb, (long) (firstGb * 1024L))
    long askMb      = (long) (first * Math.pow(f, attempt - 1))
    MemoryUnit.of("${Math.min(capMb, askMb)} MB")
}

// Median peak of the Karlin-Altschul fit inside kmerseekIndex, before headroom; 0 when
// no extend arm wants a fit. Constants and their measurements: params.kmerseek_ka_fit_*.
def kaFitGb = { String label, int ksize, int scaled ->
    int nq = params.kmerseek_ka_queries as int
    if (nq <= 0 || !resolveExtensions().any { it.startsWith('extend:') }) return 0.0d
    double bits = keyspaceBits(label, ksize)
    double at12 = params.kmerseek_ka_fit_memory_gb_at_12_bits as double
    double halv = params.kmerseek_ka_fit_memory_halving_bits as double
    double pw   = params.kmerseek_ka_fit_memory_scaled_power as double
    at12 * Math.pow(2.0d, -(bits - 12.0d) / halv) * Math.pow((double) scaled, -pw) * (nq / 500.0d)
}

// Build and fit run one after the other in the same task, so the ask is the larger of
// the two models, not their sum. The build term was `52 + 36 x filled` GB, fit on the
// 0.3 builder, where filled = min(1, 2^(bits - 28)); 0.3 did peak at 74 GB on protein20
// k10. The 0.4 builder peaks at 18 GB there, 42 GB (max 56) at 23 bits and 44 GB
// (max 48) at 19 bits, at scaled 1 -- three points that do not follow the keyspace at
// all -- so for 0.4 the build term is a flat params.kmerseek_index_build_gb that covers
// every one of them, and the 2x retry reaches the old ask if a combo exceeds it. Refit
// once the full-table run has measured a build for every alphabet.
def kmerseekIndexMemory = { String label, int ksize, int scaled, int attempt ->
    double build   = params.kmerseek_index_build_gb as double
    double gb      = 1.4d * Math.max(build, kaFitGb(label, ksize, scaled))
    double floorGb = MemoryUnit.of(params.kmerseek_index_memory_floor).toGiga()
    memoryLadder(Math.max(floorGb, gb), attempt)
}

def kmerseekSearchMemory = { Path index_dir, String alphabet, String lowcomp, int attempt ->
    def load = SPECTRUM_LOAD[index_dir.name]
    if (load == null) {
        error "no spectrum load cached for ${index_dir.name}: the skip filter in the workflow " +
              "is what fills the cache, and every kmerseekSearch input has to pass through it"
    }
    double factor  = (params.kmerseek_search_memory_alphabet_factor[alphabet] ?: 1.0) as double
    double gb      = (params.kmerseek_search_memory_headroom as double) * factor
                     * searchMedianGb(load as double, lowcomp)
    double floorGb = MemoryUnit.of(params.kmerseek_search_memory_floor).toGiga()
    memoryLadder(Math.max(floorGb, gb), attempt)
}

// The reference is keyed by the CLADE REMOVED, not by the species asking for it. Chordata
// is one file shared by six query species and Metazoa by nine, so a second species costs
// nothing once the first has built it.
process buildReference {
    tag "minus_${clade}"
    storeDir { params.reference_cache ?: "${params.outdir}/reference" }
    label 'python_scoring'
    cpus 2
    memory '8 GB'

    input:
    tuple val(clade), path(swissprot)

    output:
    tuple val(clade), path("minus_${clade}")

    script:
    """
    set -euo pipefail
    mkdir -p minus_${clade}
    build_clade_excluded_reference.py \\
        --swissprot-dat ${swissprot} \\
        --exclude-clade ${clade} \\
        --out-prefix minus_${clade}/reference \\
        --summary-out minus_${clade}/summary.json
    """

    stub:
    """
    mkdir -p minus_${clade}
    printf '>Q6GZX4\\nMKLLVLLAAGGLLSAQ\\n>P12345\\nMSTEQKLISEEDLNGAQ\\n' > minus_${clade}/reference.fasta
    echo '{"excluded_clade": "${clade}", "entries_kept": 2}' > minus_${clade}/summary.json
    """
}

// Headers rewritten to the bare accession, then chunked. Every step after this one reads
// the rewritten proteome, never the registry's file: the three arms and kmerseek all
// report a UniProt-style header differently (see bin/split_query_fasta.py), and one
// rewrite here is what gives them a common key.
process splitQuery {
    tag "${species}"
    label 'python_scoring'
    cpus 1
    memory '4 GB'

    input:
    tuple val(species), val(clade), path(query)

    output:
    tuple val(species), val(clade), path("${species}.query.fasta"), emit: proteome
    tuple val(species), val(clade), path("chunks/chunk_*.fasta"),   emit: chunks

    script:
    """
    set -euo pipefail
    mkdir -p chunks
    split_query_fasta.py --in ${query} --outdir chunks \\
        --proteome-out ${species}.query.fasta \\
        --chunk-size ${params.query_chunk_size}
    """

    stub:
    """
    mkdir -p chunks
    printf '>${species}_A\\nMKLLVLLAAGGLLSAQ\\n>${species}_B\\nMSTEQKLISEEDLNGAQ\\n' > ${species}.query.fasta
    printf '>${species}_A\\nMKLLVLLAAGGLLSAQ\\n' > chunks/chunk_0000.fasta
    printf '>${species}_B\\nMSTEQKLISEEDLNGAQ\\n' > chunks/chunk_0001.fasta
    """
}

process phmmerSearch {
    tag "${species}.${chunk.simpleName}_vs_minus_${clade}"
    container HMMER
    label 'high_cpu'
    // storeDir, not publishDir: a chunk's hits against a reference do not depend on
    // anything but the two of them, so a second run (the 0.4 ladder, from its own launch
    // and results directories) finds them here and skips the task instead of repeating
    // hours of jackhmmer. --hits_cache points at another run's results directory; the
    // file name is the same one publishDir wrote there before this change. One output
    // per process, on purpose: storeDir with two outputs is the "Directory not empty"
    // failure this project has had twice.
    storeDir "${params.hits_cache ?: params.outdir}/${species}/hits"

    input:
    tuple val(species), val(clade), path(chunk), path(ref_dir)

    // No arm label in the output: storeDir can only pre-resolve a `val` that is an input,
    // so a literal val('phmmer') here made every task "storeDir can only be used with
    // `val` and `path` outputs" and nothing was ever stored. The file name says the arm.
    output:
    tuple val(species), path("${species}.${chunk.simpleName}.phmmer.tsv.gz")

    script:
    """
    set -euo pipefail
    phmmer --domtblout /dev/stdout --tblout /dev/null -o /dev/stderr --noali \\
        -E ${params.evalue_report} --cpu ${task.cpus} \\
        ${chunk} ${ref_dir}/reference.fasta \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$4 "\\t" \$1 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > ${species}.${chunk.simpleName}.phmmer.tsv.gz
    """

    stub:
    """
    printf '${species}_A\\tQ6GZX4\\t1\\t10\\t50\\t1e-9\\n' | gzip -c > ${species}.${chunk.simpleName}.phmmer.tsv.gz
    """
}

process jackhmmerSearch {
    tag "${species}.${chunk.simpleName}_vs_minus_${clade}"
    container HMMER
    label 'high_cpu'
    // storeDir, not publishDir: a chunk's hits against a reference do not depend on
    // anything but the two of them, so a second run (the 0.4 ladder, from its own launch
    // and results directories) finds them here and skips the task instead of repeating
    // hours of jackhmmer. --hits_cache points at another run's results directory; the
    // file name is the same one publishDir wrote there before this change. One output
    // per process, on purpose: storeDir with two outputs is the "Directory not empty"
    // failure this project has had twice.
    storeDir "${params.hits_cache ?: params.outdir}/${species}/hits"

    input:
    tuple val(species), val(clade), path(chunk), path(ref_dir)

    // No arm label in the output: storeDir can only pre-resolve a `val` that is an input,
    // so a literal val('jackhmmer') here made every task "storeDir can only be used with
    // `val` and `path` outputs" and nothing was ever stored. The file name says the arm.
    output:
    tuple val(species), path("${species}.${chunk.simpleName}.jackhmmer.tsv.gz")

    script:
    """
    set -euo pipefail
    jackhmmer -N ${params.jackhmmer_iterations} \\
        --domtblout /dev/stdout --tblout /dev/null -o /dev/stderr --noali \\
        -E ${params.evalue_report} --cpu ${task.cpus} \\
        ${chunk} ${ref_dir}/reference.fasta \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$4 "\\t" \$1 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > ${species}.${chunk.simpleName}.jackhmmer.tsv.gz
    """

    stub:
    """
    printf '${species}_A\\tQ6GZX4\\t1\\t10\\t50\\t1e-9\\n' | gzip -c > ${species}.${chunk.simpleName}.jackhmmer.tsv.gz
    """
}

process mmseqs2Search {
    tag "${species}.${chunk.simpleName}_vs_minus_${clade}"
    container MMSEQS
    label 'high_cpu'
    // storeDir, not publishDir: a chunk's hits against a reference do not depend on
    // anything but the two of them, so a second run (the 0.4 ladder, from its own launch
    // and results directories) finds them here and skips the task instead of repeating
    // hours of jackhmmer. --hits_cache points at another run's results directory; the
    // file name is the same one publishDir wrote there before this change. One output
    // per process, on purpose: storeDir with two outputs is the "Directory not empty"
    // failure this project has had twice.
    storeDir "${params.hits_cache ?: params.outdir}/${species}/hits"

    input:
    tuple val(species), val(clade), path(chunk), path(ref_dir)

    // No arm label in the output: storeDir can only pre-resolve a `val` that is an input,
    // so a literal val('mmseqs2') here made every task "storeDir can only be used with
    // `val` and `path` outputs" and nothing was ever stored. The file name says the arm.
    output:
    tuple val(species), path("${species}.${chunk.simpleName}.mmseqs2.tsv.gz")

    script:
    """
    set -euo pipefail
    mmseqs createdb ${chunk} qdb
    mmseqs createdb ${ref_dir}/reference.fasta tdb
    # --num-iterations 3 is the iterative arm. The dark set is defined against the STRONGEST
    # sequence search available, not the cheapest -- a protein that iterative search reaches
    # is not dark, and counting it as dark would inflate the headline.
    mmseqs search qdb tdb res tmp -s ${params.mmseqs2_sensitivity} \\
        --num-iterations 3 -e ${params.evalue_report} --threads ${task.cpus}
    mmseqs convertalis qdb tdb res out.tsv \\
        --format-output 'query,target,tstart,tend,bits,evalue'
    gzip -c out.tsv > ${species}.${chunk.simpleName}.mmseqs2.tsv.gz
    """

    stub:
    """
    printf '${species}_A\\tQ6GZX4\\t1\\t10\\t50\\t1e-9\\n' | gzip -c > ${species}.${chunk.simpleName}.mmseqs2.tsv.gz
    """
}

// Indexes the SAME clade-excluded reference the sequence arms search. Indexing all of
// reviewed Swiss-Prot instead and dropping self-clade hits afterwards is safe for the two
// invertebrates -- Ascidiacea is 0.02% of the database and Bivalvia 0.05%, so the index-
// time IDF barely moves -- but it would leave kmerseek searching a target set the other
// arms never saw, and a win from a hit the baselines had no access to is not a win. On
// the ladder it is not even safe: Gammaproteobacteria is 21.56% of Swiss-Prot and Mammalia
// larger still, and an IDF computed with the query's own clade in the pool is a different
// statistic. So: one index per clade per combo, and the confound never arises.
process kmerseekIndex {
    tag "minus_${clade}.${alphabet}.k${ksize}${scaled == 1 ? '' : '.s' + scaled}.lc${lowcomp}"
    storeDir { params.index_cache ?: "${params.outdir}/kmerseek_index" }
    container params.kmerseek_image
    // Sized per combo in the body -- see kmerseekIndexMemory. No `label` here on purpose:
    // a withLabel selector in the config beats a body directive, and the generic
    // process block does not, which is the one arrangement under which this closure is
    // the memory that actually applies. cpus and time come from `withName: kmerseekIndex`
    // in nextflow.config, which sets neither memory nor label.
    memory { kmerseekIndexMemory(alphabet, ksize as int, scaled as int, task.attempt) }

    input:
    tuple val(clade), path(ref_dir), val(alphabet), val(ksize), val(lowcomp), val(scaled)

    output:
    tuple val(clade), val(alphabet), val(ksize), val(lowcomp), val(scaled),
          path("minus_${clade}.${alphabet}.k${ksize}${scaled == 1 ? '' : '.s' + scaled}.lc${lowcomp}.kmerseek.rocksdb")

    script:
    // scaled=1 keeps the name every index built before 0.4 has, so a store of those
    // still hits; any other value names itself.
    def idx = "minus_${clade}.${alphabet}.k${ksize}${scaled == 1 ? '' : '.s' + scaled}.lc${lowcomp}.kmerseek.rocksdb"
    def lc  = lowcomp == 'true' ? '--remove-low-complexity' : ''
    // The Karlin-Altschul fit lives in the index and is only wanted when an `extend` arm
    // will read it. One fit per mismatch penalty: `kmerseek index` stores the first, and
    // `kmerseek calibrate` adds one per further penalty, before the index is made
    // read-only below. Each fit searches --ka-queries reference sequences against the
    // index it just built, so this task then costs a search's memory as well (see the
    // memory directive).
    def nq   = params.kmerseek_ka_queries as int
    def pens = (nq > 0 && resolveExtensions().any { it.startsWith('extend:') }) ? penaltiesFor(alphabet) : []
    def ka   = pens ? "--extend-mismatch-penalty ${pens[0]} --extend-xdrop ${xdropFor(pens[0])} --ka-queries ${nq}"
                    : '--ka-queries 0'
    def more = pens.drop(1).collect { c ->
        "kmerseek calibrate --target ${idx} --extend-mismatch-penalty ${c} " +
        "--extend-xdrop ${xdropFor(c)} --ka-queries ${nq} 2>&1 | stamp | tee -a index.log"
    }.join('\n    ')
    """
    set -euo pipefail
    # One timestamp per log line. The index build and the Karlin-Altschul E-value fit
    # both happen inside the one `kmerseek index` call and it prints no times, so
    # without this the two cannot be told apart afterwards (2026-09-20: the fit turned
    # out to be 25x the CPU of the build at hp_pbotc_1st_ed2 k19, and only the task
    # total was measurable). Build = from "start" to "Fitting Karlin-Altschul"; fit =
    # from there to "Stored in the index". Bash's own printf stamps, so it costs no
    # process per line and needs nothing from the image.
    stamp() { while IFS= read -r line; do printf '%(%Y-%m-%dT%H:%M:%S)T %s\\n' -1 "\$line"; done; }
    echo start | stamp | tee index.log
    kmerseek index \\
        --alphabet ${alphabet} --ksize ${ksize} --scaled ${scaled} \\
        --input  ${ref_dir}/reference.fasta \\
        --output ${idx} ${lc} ${ka} \\
        --kmer-stats-out ${idx}/spectrum.csv.gz 2>&1 | stamp | tee -a index.log
    ${more}
    # An index is immutable once built. A search that opens it read-write (the image
    # before 2026-09-13-rocksdb-4gb-readonly did) rewrites CURRENT, MANIFEST and LOG,
    # and Nextflow hashes a directory input from its files' names, sizes and mtimes, so
    # every finished search on that index silently loses its cache on the next -resume.
    # On 2026-09-17 one accidental run with the old image cost 272 mouse searches that
    # way. Read-only permissions turn that into a loud open failure. To delete an index,
    # chmod -R u+w it first (make unlock-indexes).
    #
    # The files go read-only here; the directory itself stays writable, because
    # Nextflow's storeDir step is a rename into another parent, and renaming a
    # directory rewrites its `..` entry, which needs write permission on that
    # directory. With the directory locked too, every index task on 2026-09-19 built
    # its index and then died at stage-out with `mv: ... Permission denied` (147 tasks,
    # each retried). Read-only files are enough for the loud failure: a read-write open
    # starts by opening LOCK for writing, which EACCES stops. `make lock-indexes` closes
    # the directory bit afterwards, in place.
    chmod -R a-w ${idx}
    chmod u+w ${idx}
    """

    stub:
    // A spectrum with the shape the real one has, skewed enough at low k that the skip
    // filter fires for the stub run too: occurrence counts scale down with ksize.
    def top = Math.max(1, (long) (400000 / Math.pow(2, ksize as int)))
    """
    d=minus_${clade}.${alphabet}.k${ksize}${scaled == 1 ? '' : '.s' + scaled}.lc${lowcomp}.kmerseek.rocksdb
    mkdir -p \$d
    touch \$d/CURRENT
    printf '# stub\nmoltype,ksize,occurrences,n_kmers\n${alphabet},${ksize},1,1000000\n${alphabet},${ksize},${top},10\n' \
        | gzip -c > \$d/spectrum.csv.gz
    """
}

// One stored index gets the Karlin-Altschul fits its alphabet's penalties need, in
// place. For the calibrateStore entry: the fit an index carries is looked up by exact
// penalty and X-drop, so whenever a penalty changes after indexes are built (the kappa
// table gained thirteen alphabets and corrected hp_pbotc_1st_ed2 on 2026-09-21) every
// stored index would refuse the new arm as unfitted until it is fitted for it. The fit
// is a --ka-queries search against the index, so it is sized as the index task's fit is.
// Idempotent: calibrate replaces an earlier fit for the same pair; the files are
// unlocked for the write and locked again after.
process calibrateStoredIndex {
    tag "${file(index_path).name.replace('.kmerseek.rocksdb', '')}"
    container params.kmerseek_image
    memory { kmerseekIndexMemory(alphabet, ksize as int, scaled as int, task.attempt) }

    input:
    tuple val(index_path), val(alphabet), val(ksize), val(scaled), val(penalties)

    output:
    path "${file(index_path).name}.calibrate.log"

    script:
    def name = file(index_path).name
    def nq   = params.kmerseek_ka_queries as int
    def fits = penalties.collect { c ->
        "kmerseek calibrate --target ${index_path} --extend-mismatch-penalty ${c} " +
        "--extend-xdrop ${xdropFor(c)} --ka-queries ${nq} 2>&1 | stamp | tee -a ${name}.calibrate.log"
    }.join('\n    ')
    """
    set -euo pipefail
    stamp() { while IFS= read -r line; do printf '%(%Y-%m-%dT%H:%M:%S)T %s\\n' -1 "\$line"; done; }
    echo "start ${name}: penalties ${penalties.join(' ')}" | stamp | tee ${name}.calibrate.log
    chmod -R u+w ${index_path}
    ${fits}
    chmod -R a-w ${index_path}
    chmod u+w ${index_path}
    """

    stub:
    """
    echo stub > ${file(index_path).name}.calibrate.log
    """
}

process kmerseekSearch {
    tag "${species}.${chunk.simpleName}.${alphabet}.k${ksize}.s${scaled}.lc${lowcomp}.${armName(ext, alphabet)}"
    container params.kmerseek_image
    publishDir "${params.outdir}/${species}/kmerseek", mode: 'copy', pattern: '*.zst'
    // Sized per combo in the body -- see kmerseekSearchMemory and the note on
    // kmerseekIndex for why there is no label. On 2026-09-13 a flat 64 GB first tier was
    // cgroup-killed on 7 of 23 Botryllus chunks of hp_thomas_dill2 k23 with the mask OFF
    // (exit 137) and every retry cost a full requeue; the per-combo model asks for what the
    // measured peak needs the first time.
    memory { kmerseekSearchMemory(index_dir, alphabet, lowcomp, task.attempt) }

    input:
    tuple val(species), val(clade), path(chunk), val(alphabet), val(ksize), val(lowcomp), val(scaled),
          val(ext), path(index_dir)

    output:
    tuple val(species), val(alphabet), val(ksize), val(lowcomp), val(scaled), val(ext),
          path("${chunk.simpleName}.${alphabet}.k${ksize}.s${scaled}.lc${lowcomp}.${armName(ext, alphabet)}.queries.tsv"),
          path("${chunk.simpleName}.${alphabet}.k${ksize}.s${scaled}.lc${lowcomp}.${armName(ext, alphabet)}.landmarks.csv"), emit: queries
    path "*.regions.csv.zst", emit: regions

    script:
    def slug = "${chunk.simpleName}.${alphabet}.k${ksize}.s${scaled}.lc${lowcomp}.${armName(ext, alphabet)}"
    def lc   = lowcomp == 'true' ? '--remove-low-complexity' : ''
    def flags = extensionFlags(ext, alphabet)
    // The landmark queries of this species, for awk: a space-separated list, or empty.
    def lmq  = landmarksFor(species).collect { it[0] }.unique().join(' ')
    """
    set -euo pipefail
    set +e
    kmerseek search \\
        --alphabet ${alphabet} --ksize ${ksize} \\
        --query  ${chunk} --target ${index_dir} ${lc} ${flags} \\
        --threshold        ${params.threshold} \\
        --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} \\
        --min-region-score ${params.min_region_score} \\
        2> ${slug}.log | zstd -T2 -o ${slug}.regions.csv.zst
    status=(\${PIPESTATUS[@]})
    set -e
    # One failure is not a failure of the task: an `extend` search refused because the
    # index holds no Karlin-Altschul fit for its penalty (the index-time fit found too
    # few score bins; it happens on 20-letter alphabets from k=8 or so). That arm cannot
    # be searched against that index at all, so it is recorded as unfitted rather than
    # as zero reach: the queries file starts with a #nofit line, the gain step lists the
    # arm under unfitted_arms and draws no row for it, and the regions file is a header.
    if [ "\${status[0]}" -ne 0 ] && grep -q "no Karlin-Altschul fit" ${slug}.log; then
        printf '#nofit\\t%s\\n' "\$(grep -m1 'no Karlin-Altschul fit' ${slug}.log | tr '\\t' ' ')" > ${slug}.queries.tsv
        printf 'query_name,target_name,region_evalue\\n' | zstd -T2 -f -o ${slug}.regions.csv.zst
        printf 'query_name,target_name,region_evalue\\n' > ${slug}.landmarks.csv
        exit 0
    fi
    [ "\${status[0]}" -eq 0 ] && [ "\${status[1]}" -eq 0 ] || exit 1
    # No `|| true` for any other failure, on purpose. A search that finds nothing exits 0
    # (checked against the 0.4.0 binary), so tolerating a non-zero exit protects nothing -- and on
    # 2026-09-12 it hid 32 of 92 tasks dying on a RocksDB LOCK race (kmerseek PR #53) as
    # 32 empty result files, which the gain step then counted as "kmerseek found none".
    # Under pipefail a failed search or a failed zstd fails the task, and the retry above
    # gets three attempts before the run stops.

    # The queries that got any region, one per line with the smallest region_evalue any
    # of its regions carries (inf on an `exact` arm, where no E-value exists). Extracted
    # with zstd + awk rather than by reading the .zst in polars: scan_csv on a zstd CSV
    # inflates the whole file in RAM (1.9 GB -> 184.7 GB once, on this project). Both
    # columns are found by HEADER NAME, never by position, so a column order change cannot
    # silently shift them. Splitting on a bare comma is safe only because splitQuery
    # rewrote every header to the bare accession: kmerseek writes the WHOLE header into
    # query_name, and a QfO description line carries commas.
    printf 'query_name\\tmin_region_evalue\\n' > ${slug}.queries.tsv
    if [ -s ${slug}.regions.csv.zst ]; then
        zstd -dc ${slug}.regions.csv.zst \\
        | awk -F, 'NR==1 { for (i=1;i<=NF;i++) { if (\$i=="query_name") c=i; if (\$i=="region_evalue") e=i }
                           if (!c) { print "no query_name column" > "/dev/stderr"; exit 3 }
                           next }
                   c { v = e ? \$e : "inf"; if (v == "" ) v = "inf";
                       if (!(\$c in m) || (v+0) < (m[\$c]+0)) m[\$c] = v }
                   END { for (q in m) print q "\\t" m[q] }' \\
        | sort >> ${slug}.queries.tsv
    fi

    # Every row of this species' landmark queries, all columns, for the landmark table in
    # the report. Header only when the species has no landmark or the query hit nothing.
    if [ -s ${slug}.regions.csv.zst ]; then
        zstd -dc ${slug}.regions.csv.zst \\
        | awk -F, -v q="${lmq}" 'BEGIN { n = split(q, a, " "); for (i = 1; i <= n; i++) want[a[i]] = 1 }
                   NR==1 { for (i=1;i<=NF;i++) if (\$i=="query_name") c=i; print; next }
                   (\$c in want) { print }' > ${slug}.landmarks.csv
    else
        printf 'query_name,target_name,region_evalue\\n' > ${slug}.landmarks.csv
    fi
    """

    stub:
    def slug = "${chunk.simpleName}.${alphabet}.k${ksize}.s${scaled}.lc${lowcomp}.${armName(ext, alphabet)}"
    def ev   = ext == 'exact' ? 'inf' : '1e-5'
    """
    printf 'query_name,target_name,region_evalue\\n${species}_B,Q6GZX4,${ev}\\n' | zstd -o ${slug}.regions.csv.zst
    printf 'query_name\\tmin_region_evalue\\n${species}_B\\t${ev}\\n' > ${slug}.queries.tsv
    printf 'query_name,target_name,region_evalue\\n' > ${slug}.landmarks.csv
    """
}

process kmerseekDarkGain {
    tag "${species}"
    label 'python_scoring'
    memory '16 GB'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(query), path(dark_parquet), path(query_lists, stageAs: 'queries/*')

    output:
    tuple val(species),
          path("${species}_kmerseek_dark_gain.parquet"),
          path("${species}_kmerseek_dark_gain.json")

    script:
    """
    set -euo pipefail
    kmerseek_dark_gain.py \\
        --dark ${dark_parquet} --query ${query} --species ${species} \\
        --queries queries/*.queries.tsv \\
        --evalue-max ${params.kmerseek_evalue_max} \\
        --out ${species}_kmerseek_dark_gain.parquet \\
        --summary-out ${species}_kmerseek_dark_gain.json
    """

    stub:
    """
    touch ${species}_kmerseek_dark_gain.parquet
    echo "{\\"species\\": \\"${species}\\", \\"n_query_lists\\": \$(ls queries | wc -l)}" \\
        > ${species}_kmerseek_dark_gain.json
    """
}

// One species' landmark pairs (see params.landmarks) across every arm: found or not, and
// how well, from the sequence arms' hit tables and the kmerseek searches' landmark rows.
process landmarkSummary {
    tag "${species}"
    label 'python_scoring'
    memory '8 GB'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(dark_parquet), path(hits, stageAs: 'hits/*'), path(landmark_csvs, stageAs: 'landmarks/*')

    output:
    tuple val(species), path("${species}_landmarks.parquet"), path("${species}_landmarks.json")

    script:
    def pairs = landmarksFor(species).collect { q, t -> "${q}:${t}" }.join(',')
    """
    set -euo pipefail
    landmark_summary.py \\
        --species ${species} --pairs '${pairs}' --dark ${dark_parquet} \\
        --hits hits/*.tsv.gz --landmark-csvs landmarks/*.landmarks.csv \\
        --out ${species}_landmarks.parquet --summary-out ${species}_landmarks.json
    """

    stub:
    """
    touch ${species}_landmarks.parquet
    echo '{"species": "${species}", "pairs": []}' > ${species}_landmarks.json
    """
}

process computeDarkSet {
    tag "${species}"
    label 'python_scoring'
    memory '16 GB'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(query), path(hits, stageAs: 'hits/*')

    output:
    tuple val(species), path("${species}_dark_set.parquet"), path("${species}_dark_summary.json")

    script:
    """
    set -euo pipefail
    compute_dark_set.py \\
        --query ${query} --species ${species} \\
        --evalue-call ${params.evalue_call} \\
        --hits hits/*.tsv.gz \\
        --out ${species}_dark_set.parquet \\
        --summary-out ${species}_dark_summary.json
    """

    stub:
    """
    touch ${species}_dark_set.parquet
    echo "{\\"species\\": \\"${species}\\", \\"n_hit_files\\": \$(ls hits | wc -l)}" \\
        > ${species}_dark_summary.json
    """
}

// --- species resolution ------------------------------------------------------------------

def loadRegistry() {
    def f = file(params.registry)
    if (!f.exists()) error "no registry at ${params.registry}"
    new groovy.json.JsonSlurper().parse(f.toFile())
}

// One resolved row per requested species: the label, the clade removed from the
// reference, and the proteome file. The registry says where a proteome is in one of two
// ways -- `staged_fasta`, a path under --qfo_dir for a proteome staged by hand (Botryllus),
// or the QfO release's own <subdir>/<proteome>_<taxon>.fasta for every QfO species -- and
// both are resolved here so nothing downstream knows the difference.
def resolveSpecies() {
    if (!params.species) {
        error "--species is required: one registry label, or a comma-separated list " +
              "(e.g. botryllus, or mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli)"
    }
    def reg   = loadRegistry()
    def known = reg.findAll { k, v -> v instanceof Map && v.annotate_query }.keySet()
    def names = params.species.toString().tokenize(',')*.trim().findAll { it }
    def dup   = names.countBy { it }.findAll { _n, c -> c > 1 }*.key
    if (dup) error "--species lists ${dup.join(', ')} more than once"
    def unknown = names - known
    if (unknown) {
        error "unknown species: ${unknown.join(', ')}. Registry query species: ${known.join(', ')}"
    }
    if (names.size() > 1 && (params.query_fasta || params.exclude_clade)) {
        error "--query_fasta and --exclude_clade override ONE species' registry row; " +
              "with ${names.size()} species there is no way to say which"
    }

    names.collect { sp ->
        def row   = reg[sp]
        def clade = params.exclude_clade ?: row.annotate_clade
        if (!clade) error "no clade to exclude for ${sp}; pass --exclude_clade"

        def qpath = params.query_fasta
        if (!qpath && row.staged_fasta) {
            qpath = "${params.qfo_dir}/${row.staged_fasta}"
        }
        if (!qpath && row.qfo_proteome && row.qfo_subdir && row.taxon_id) {
            qpath = "${params.qfo_dir}/${row.qfo_subdir}/${row.qfo_proteome}_${row.taxon_id}.fasta"
        }
        if (!qpath) error "no query FASTA for ${sp}: the registry row has neither staged_fasta " +
                          "nor qfo_proteome+qfo_subdir+taxon_id; pass --query_fasta"
        def query = file(qpath)
        if (!query.exists()) error "query proteome missing for ${sp}: ${query}"
        [label: sp, clade: clade, query: query]
    }
}

// --- kmerseek combos ---------------------------------------------------------------------

// (alphabet, ksize, lowcomp) triples. The lowcomp axis is applied to whichever of the
// three sources of alphabet:ksize pairs is in force.
def resolveCombos() {
    def lcs = params.kmerseek_lowcomp.toString().tokenize(',')*.trim().findAll { it }
    def bad = lcs.findAll { !(it in ['true', 'false']) }
    if (bad) error "--kmerseek_lowcomp takes true and/or false, not '${bad.join(', ')}'"
    if (!lcs) error "--kmerseek_lowcomp is empty; it needs at least one of true, false"

    List pairs
    if (params.kmerseek_sweep || params.kmerseek_encodings) {
        def table = params.kmerseek_extra_encodings ? knownEncodings() : allEncodings()
        if (params.kmerseek_encodings) {
            def wanted  = params.kmerseek_encodings.toString().tokenize(',')*.trim().findAll { it }
            def unknown = wanted - knownEncodings()*.get(0)
            if (unknown) {
                error "Unknown encoding(s) in --kmerseek_encodings: ${unknown.join(', ')}. " +
                      "Known: ${knownEncodings()*.get(0).join(', ')}"
            }
            // Looked up over the whole known table, so an EXTRA alphabet can be named
            // outright without also turning --kmerseek_extra_encodings on.
            table = wanted.collect { w -> knownEncodings().find { it[0] == w } }
        }
        pairs = expandEncodings(table).collect { cli, _label, k -> [cli, k] }
        def plus = params.kmerseek_sweep_plus.toString().tokenize(',')*.trim().findAll { it }.collect { spec ->
            def parts = spec.tokenize(':')
            if (parts.size() != 2 || !(parts[1] ==~ /\d+/)) {
                error "--kmerseek_sweep_plus entries are alphabet:ksize, not '${spec}'"
            }
            if (!(parts[0] in knownEncodings()*.get(0))) {
                error "Unknown alphabet in --kmerseek_sweep_plus: ${parts[0]}. " +
                      "Known: ${knownEncodings()*.get(0).join(', ')}"
            }
            [parts[0], parts[1] as Integer]
        }
        pairs = (pairs + plus).unique()
    }
    else {
        pairs = params.kmerseek_alphabets.tokenize(',')*.trim().findAll { it }.collect { spec ->
            def parts = spec.tokenize(':')
            if (parts.size() != 2 || !(parts[1] ==~ /\d+/)) {
                error "--kmerseek_alphabets entries are alphabet:ksize, not '${spec}'"
            }
            [parts[0], parts[1] as Integer]
        }
    }
    def scaleds = params.kmerseek_scaled.toString().tokenize(',')*.trim().findAll { it }
    def badS = scaleds.findAll { !(it ==~ /\d+/) || (it as int) < 1 || (it as int) > 10 }
    if (badS) error "--kmerseek_scaled takes integers 1..10, not '${badS.join(', ')}'"
    if (!scaleds) error "--kmerseek_scaled is empty; it needs at least one value"
    pairs.collectMany { a, k -> lcs.collectMany { lc -> scaleds.collect { sc -> [a, k, lc, sc as int] } } }
}

// kappa, the copy rate: the fraction of aligned positions in a Pfam pair at 20-30%
// identity where the target carries the query's class because it was conserved, over
// and above chance agreement. One number per alphabet, from analysis notebook 230's
// Pfam seed pairs (all 17 sweep alphabets; see assets/kappa_by_alphabet.tsv for the
// values, their confidence intervals and how the three the notebook left out were
// added). It is a property of the alphabet and of homologs, not of a database: the
// database-dependent numbers, K and lambda, are fitted on every index at build time.
// A function, not a top-level map: a script-level `def` is not in scope inside the
// functions below (only the closures capture it).
def kappaTable() {
    def table = [:]
    file("${projectDir}/assets/kappa_by_alphabet.tsv").readLines().each { line ->
        if (line.startsWith('#') || line.startsWith('alphabet') || !line.trim()) return
        def f = line.split('\t')
        table[f[0]] = f[2] as double
    }
    table
}

// The number of classes is the trailing integer of every 0.4 alphabet name.
def alphabetClasses(String alphabet) {
    def m = alphabet =~ /(\d+)$/
    if (!m) error "cannot read the class count off alphabet name '${alphabet}'"
    m[0][1] as int
}

// C_best for one alphabet, equal class shares. Rounded to two decimals so the same
// string reaches `kmerseek calibrate` at index time and `kmerseek search` later: the
// stored fit is looked up by exact penalty and X-drop.
def optimalPenalty(String alphabet) {
    def kappa = kappaTable()[alphabet]
    if (kappa == null) {
        error "no kappa measured for ${alphabet}, so no `opt` mismatch penalty for it; " +
              "measured: ${kappaTable().keySet().join(', ')}. Give a number instead."
    }
    double f = 1.0d / alphabetClasses(alphabet)
    double c = -Math.log(1.0d - kappa) / Math.log(1.0d + kappa / f - kappa)
    penaltyString(c)
}

// 2 -> "2", 1.63 -> "1.63", 0.14 -> "0.14": the shortest string that still names the
// value, used in file names, arm labels and on the kmerseek command line alike.
def penaltyString(double c) {
    def sf = String.format('%.2f', c)
    sf.contains('.') ? sf.replaceAll(/0+$/, '').replaceAll(/\.$/, '') : sf
}

// The mismatch penalties one alphabet's `extend` arms use, as strings. `opt` needs a
// measured kappa, and only five alphabets have one (kappaTable), so on a run over the
// whole table `opt` is an arm those five get and the others do not: it is dropped here
// for an alphabet without a kappa, and armsFor below drops the matching search arm.
// The plan log names every alphabet this happens to.
def hasOptimalPenalty(String alphabet) { kappaTable()[alphabet] != null }

def penaltiesFor(String alphabet) {
    params.kmerseek_mismatch_penalty.toString().tokenize(',')*.trim().findAll { it }.collect { spec ->
        spec == 'opt' ? (hasOptimalPenalty(alphabet) ? optimalPenalty(alphabet) : null)
                      : penaltyString(spec as double)
    }.findAll { it != null }.unique()
}

// The search arms one alphabet's indexes get: every requested arm, minus `extend:opt`
// where no kappa is measured.
def armsFor(String alphabet) {
    resolveExtensions().findAll { it != 'extend:opt' || hasOptimalPenalty(alphabet) }
}

def xdropFor(String penalty) {
    penaltyString((params.kmerseek_extend_xdrop_per_penalty as double) * (penalty as double))
}

// The landmark pairs of one species, as [query, target] accession pairs.
def landmarksFor(String species) {
    params.landmarks.toString().tokenize(',')*.trim().findAll { it }.collect { spec ->
        def parts = spec.tokenize(':')
        if (parts.size() != 3) error "--landmarks entries are species:query:target, not '${spec}'"
        parts
    }.findAll { it[0] == species }.collect { [it[1], it[2]] }
}

// The search-time arms over one index, as specs: `exact`, or `extend:<C>` with C a number
// or `opt`. Validated once, here, so a typo does not surface as a failed search hours in.
// The numeric C of an `opt` arm depends on the alphabet and is resolved in the process.
def resolveExtensions() {
    def exts = params.kmerseek_extension.toString().tokenize(',')*.trim().findAll { it }
    def bad = exts.findAll { !(it in ['exact', 'extend']) }
    if (bad) error "--kmerseek_extension takes exact and/or extend, not '${bad.join(', ')}'"
    if (!exts) error "--kmerseek_extension is empty; it needs at least one of exact, extend"
    def pens = params.kmerseek_mismatch_penalty.toString().tokenize(',')*.trim().findAll { it }
    def badP = pens.findAll { !(it == 'opt' || it ==~ /\d+(\.\d+)?/) }
    if (badP) error "--kmerseek_mismatch_penalty takes numbers and/or opt, not '${badP.join(', ')}'"
    if ('extend' in exts && !pens) error "--kmerseek_mismatch_penalty is empty but an extend arm is wanted"
    exts.collectMany { e -> e == 'exact' ? ['exact'] : pens.collect { "extend:${it}".toString() } }
}

// The part of a search's file name and arm label that says which arm it is: `exact`, or
// `extend-c<C>` with the numeric penalty.
def armName(String spec, String alphabet) {
    if (spec == 'exact') return 'exact'
    def c = spec.substring('extend:'.length())
    "extend-c${c == 'opt' ? optimalPenalty(alphabet) : penaltyString(c as double)}"
}

// The kmerseek search flags one arm adds. `exact` adds nothing, so its command line is
// the one every earlier run used.
def extensionFlags(String spec, String alphabet) {
    if (spec == 'exact') return ''
    def c = armName(spec, alphabet).substring('extend-c'.length())
    // --ka-queries 0: the E-value's lambda and K come from the fit the index build stored
    // for this penalty and X-drop, or the search is refused. The binary's default is to
    // refit on 200 reference sequences when no fit is stored, which is 200 extra
    // searches in every chunk and, where the index-time fit failed for want of score
    // bins, fails the same way after doing them. The refusal is handled in the search
    // script: the arm is recorded as unfitted for that index.
    "--extend-mismatch-penalty ${c} --extend-xdrop ${xdropFor(c)} --ka-queries 0 " +
    "--chain-max-gap ${params.kmerseek_chain_max_gap} " +
    "--chain-max-shift ${params.kmerseek_chain_max_shift}"
}

// Every path output declared as a glob arrives as a single path when it matched one
// file and as a list otherwise; this makes it a list either way.
def asList = { it instanceof List ? it : [it] }

workflow darkSet {
    def SPECIES = resolveSpecies()
    SPECIES.each { s ->
        log.info "  query    : ${s.label} (${s.query.name})"
        log.info "  reference: reviewed Swiss-Prot minus ${s.clade}"
    }

    // The scale of what is about to be submitted, so `make ladder-plan` (-preview) says
    // it before anything is queued. Chunk counts come from counting FASTA records now
    // rather than from splitQuery, which -preview never runs.
    def n_chunks_of = SPECIES.collectEntries { s ->
        int n = 0
        s.query.eachLine { line -> if (line.startsWith('>')) n++ }
        [(s.label): Math.max(1, (int) Math.ceil(n / (params.query_chunk_size as double)))]
    }
    def total_chunks = n_chunks_of.values().sum()
    log.info "  chunks   : ${total_chunks} of ${params.query_chunk_size} queries -- " +
             n_chunks_of.collect { k, v -> "${k} ${v}" }.join(', ')

    def COMBOS = params.with_kmerseek ? resolveCombos() : []
    def EXTENSIONS = params.with_kmerseek ? resolveExtensions() : []
    if (params.with_kmerseek) {
        def n_alpha  = COMBOS.collect { it[0] }.unique().size()
        def n_lc     = COMBOS.collect { it[2] }.unique().size()
        def n_sc     = COMBOS.collect { it[3] }.unique().size()
        def n_clades = SPECIES*.clade.unique().size()
        def idx_gb   = COMBOS.collect { a, k, _lc, sc -> kmerseekIndexMemory(a, k as int, sc as int, 1).toGiga() }
        log.info "  kmerseek : ${COMBOS.size()} index combos -- ${n_alpha} alphabet(s), " +
                 "${COMBOS.size().intdiv(n_lc * n_sc)} alphabet x ksize pair(s), mask setting(s): " +
                 "${COMBOS.collect { it[2] }.unique().join(',')}, scaled: " +
                 "${COMBOS.collect { it[3] }.unique().join(',')}; search arms per index: " +
                 "${EXTENSIONS.join(',')}"
        if (EXTENSIONS.any { it.startsWith('extend:') }) {
            COMBOS.collect { it[0] }.unique().each { a ->
                log.info "             ${a}: extend penalties C = ${penaltiesFor(a).join(', ') ?: 'none'}" +
                         (hasOptimalPenalty(a) ? " (kappa ${kappaTable()[a]}, ${alphabetClasses(a)} classes)"
                                               : (EXTENSIONS.contains('extend:opt') ? ' (no kappa measured: the opt arm is skipped)' : ''))
            }
        }
        log.info "             ${n_clades * COMBOS.size()} index builds (${n_clades} clade(s) x combos), " +
                 "first-attempt memory ${idx_gb.min()}-${idx_gb.max()} GB"
        log.info "             up to ${total_chunks * COMBOS.sum { armsFor(it[0]).size() }} searches (chunks x combos x arms); each is " +
                 "sized from its index's k-mer spectrum once that index exists, " +
                 "${params.kmerseek_search_memory_floor}-${params.kmerseek_memory_first_max} on the first " +
                 "attempt, and a combo predicted past ${params.kmerseek_memory_max} is skipped and logged"
    }

    // One reference per clade removed, however many species ask for it.
    ref_ch = buildReference(
        Channel.fromList(SPECIES*.clade.unique().collect { cl -> tuple(cl, file(params.swissprot_dat)) }))

    staged = splitQuery(Channel.fromList(SPECIES.collect { s -> tuple(s.label, s.clade, s.query) }))

    // (species, rewritten proteome). Every non-search step reads this file, never the
    // registry's -- see splitQuery.
    proteome = staged.proteome.map { sp, _cl, q -> tuple(sp, q) }

    // One chunk per emission, keyed by CLADE first so it can be combined with the
    // reference and the indexes, both of which are per clade.
    chunks = staged.chunks.flatMap { sp, cl, cs -> asList(cs).collect { c -> tuple(cl, sp, c) } }

    // How many chunks each species has. groupTuple() on its own waits for the WHOLE
    // channel to close before emitting anything, which with nine species of very
    // different sizes means yeast's dark set waits on arabidopsis's last jackhmmer chunk.
    // groupKey(species, n) lets each species' group close as soon as its n items are in.
    n_chunks = staged.chunks.map { sp, _cl, cs -> tuple(sp, asList(cs).size()) }

    in_ch = chunks.combine(ref_ch, by: 0).map { cl, sp, c, r -> tuple(sp, cl, c, r) }

    // Every chunk's hits from every arm land in one list per species, so computeDarkSet
    // still sees the whole proteome at once. A protein is dark only if NO arm placed it in
    // ANY chunk, and that judgement cannot be made per chunk.
    hits = phmmerSearch(in_ch)
        .mix(jackhmmerSearch(in_ch))
        .mix(mmseqs2Search(in_ch))
        .combine(n_chunks, by: 0)
        .map { sp, f, n -> tuple(groupKey(sp, n * 3), f) }
        .groupTuple()
        .map { k, fs -> tuple(k.getGroupTarget(), fs) }

    dark = computeDarkSet(proteome.join(hits))

    // Everything the report can draw besides the headline, as (species, file). Each
    // optional arm mixes its own products in where it runs, so a run without that arm
    // contributes nothing and the report omits the section by name instead of drawing an
    // empty one.
    report_extra = Channel.empty()

    dark_with_query = dark.map { sp, dp, _js -> tuple(sp, dp) }.join(proteome)
        .map { sp, dp, q -> tuple(sp, q, dp) }

    if (params.with_length_comparison) {
        len = compareDarkLengths(dark_with_query)
        report_extra = report_extra.mix(len.flatMap { sp, pq, js -> [tuple(sp, pq), tuple(sp, js)] })
    }

    // Is the dark set more disordered than the placed set?
    if (params.with_disorder) {
        dis = darkSetDisorder(dark_with_query)
        report_extra = report_extra.mix(dis.flatMap { sp, pq, js -> [tuple(sp, pq), tuple(sp, js)] })
    }

    // kmerseek is opt-in. The dark set is defined by the sequence arms alone and is worth
    // having on its own; adding kmerseek costs an index over 572_700 sequences per clade
    // per alphabet/ksize/mask, which is the expensive part of this pipeline.
    if (params.with_kmerseek) {
        combos = Channel.fromList(COMBOS.collect { a, k, lc, sc -> tuple(a, k, lc, sc) })

        idx = kmerseekIndex(ref_ch.combine(combos).map { cl, r, a, k, lc, sc -> tuple(cl, r, a, k, lc, sc) })

        // Combos no node can search are dropped HERE, with the index in hand, rather than
        // discovered three OOM kills later. gbmr7 at k=9 against a 572k-sequence reference
        // has a predicted median peak of ~360 GB and a worst chunk near 900; nothing on
        // hns takes that, and the region benchmark's entropy-derived k floors were set on
        // 20k-protein targets, thirty times smaller. A dropped combo is named in the log
        // and is simply absent from the sweep panels -- an absent point, not a zero.
        double capGb = MemoryUnit.of(params.kmerseek_memory_max).toGiga()
        double skipF = params.kmerseek_skip_factor as double
        runnable = idx.filter { cl, a, k, lc, sc, i ->
            double load = spectrumLoad(i)
            double med  = searchMedianGb(load, lc)
            if (med * skipF > capGb) {
                log.warn "skipping minus_${cl}.${a}.k${k}.s${sc}.lc${lc}: spectrum load ${Math.round(load)}, " +
                         "predicted median peak ${Math.round(med)} GB x ${skipF} is over the " +
                         "${Math.round(capGb)} GB ceiling"
                return false
            }
            true
        }

        // Every runnable index is searched once per extension arm its alphabet has.
        ks = kmerseekSearch(
            chunks.combine(runnable, by: 0)
                  .flatMap { cl, sp, c, a, k, lc, sc, i -> armsFor(a).collect { ext -> tuple(sp, cl, c, a, k, lc, sc, ext, i) } })

        // How many combos each species actually searches, AFTER the skip above. The
        // groupKey size below has to be exact: groupTuple discards a group that never
        // reaches its size (remainder is false by default), so counting the skipped
        // combos in would silently drop that species' gain step, and counting too few
        // would emit early on a partial set -- the "17 combos searched ~1000 queries and
        // looked like real negatives" failure this project has already had once. The
        // count is known once every index build of a clade has finished, which is long
        // before its searches are.
        n_combos = runnable
            .map { cl, a, _k, _lc, _sc, _i -> tuple(cl, armsFor(a).size()) }
            .groupTuple()
            .flatMap { cl, arms -> SPECIES.findAll { it.clade == cl }.collect { s -> tuple(s.label, arms.sum()) } }

        // Every chunk and every combo of a species reaches the gain step together: a
        // protein counts as rescued only against the whole dark set, and the dark set is
        // proteome-wide. Closed per species by groupKey, as for the hits above.
        q_lists = ks.queries
            .map { sp, _a, _k, _lc, _sc, _ext, f, _lm -> tuple(sp, f) }
            .combine(n_chunks, by: 0)
            .combine(n_combos, by: 0)
            .map { sp, f, n, m -> tuple(groupKey(sp, n * m), f) }
            .groupTuple()
            .map { k, fs -> tuple(k.getGroupTarget(), fs) }

        gain = kmerseekDarkGain(dark_with_query.join(q_lists))
        report_extra = report_extra.mix(gain.flatMap { sp, pq, js -> [tuple(sp, pq), tuple(sp, js)] })

        // The landmark rows, grouped exactly as the query lists are (same tasks, same count).
        lm_lists = ks.queries
            .map { sp, _a, _k, _lc, _sc, _ext, _f, lm -> tuple(sp, lm) }
            .combine(n_chunks, by: 0)
            .combine(n_combos, by: 0)
            .map { sp, lm, n, m -> tuple(groupKey(sp, n * m), lm) }
            .groupTuple()
            .map { k, fs -> tuple(k.getGroupTarget(), fs) }
        landmarks = landmarkSummary(dark.map { sp, dp, _js -> tuple(sp, dp) }.join(hits).join(lm_lists))
        report_extra = report_extra.mix(landmarks.flatMap { sp, pq, js -> [tuple(sp, pq), tuple(sp, js)] })
    }

    if (params.with_multiqc) {
        // join with remainder, not combine: a species with no optional arm at all has no
        // row in report_extra, and the report must still be built for it with an empty
        // list. combine() would also CONCATENATE the collected list into the tuple and hand
        // the process N positional arguments, the bug that already broke kmerseekDarkGain
        // here once.
        extras = report_extra.groupTuple()
        // The overview quotes the clade removed and the reference's own entry counts, so
        // each species' row is joined to its clade and then to that clade's summary.json.
        clade_of    = Channel.fromList(SPECIES.collect { s -> tuple(s.label, s.clade) })
        ref_summary = ref_ch.map { cl, dir -> tuple(cl, dir.resolve('summary.json')) }
        darkReportFrom(dark.map { sp, _dp, js -> tuple(sp, js) }
            .join(extras, remainder: true)
            .map { sp, js, ex -> tuple(sp, js, ex ?: []) }
            .join(clade_of)
            .map { sp, js, ex, cl -> tuple(cl, sp, js, ex) }
            .combine(ref_summary, by: 0)
            .map { cl, sp, js, ex, rs -> tuple(sp, cl, js, rs, ex) })
    }
}

/*
 * Report only, over a results directory a previous run already published.
 *
 * The searches are the expensive part and they are already done; re-rendering the report
 * after a plot changes should not need the work directory that produced them. So this
 * entry reads the published products straight out of --outdir and builds the report from
 * whichever of them exist. `make multiqc-dark-set SPECIES=<x>` is this entry; SPECIES may
 * be a list here too.
 */
workflow darkReport {
    if (!params.species) error "--species is required"

    def names = params.species.toString().tokenize(',')*.trim().findAll { it }
    if (names.size() > 1 && params.exclude_clade) {
        error "--exclude_clade overrides ONE species' registry row; with ${names.size()} " +
              "species there is no way to say which"
    }
    def reg      = loadRegistry()
    def ref_root = params.reference_cache ?: "${params.outdir}/reference"

    def rows = names.collect { sp ->
        def dir = file("${params.outdir}/${sp}")
        def summary = file("${dir}/${sp}_dark_summary.json")
        if (!summary.exists()) {
            error "no dark summary at ${summary}\n" +
                  "  This entry reports on a finished run; it does not compute one.\n" +
                  "  Run the dark set first:  make run-dark-set SPECIES=${sp}"
        }

        // The clade the run removed, and that reference's own summary. The overview
        // section quotes both; a re-render that cannot find the reference the run used
        // is pointed at --reference_cache rather than left to describe the wrong one.
        def clade = params.exclude_clade ?: reg[sp]?.annotate_clade
        if (!clade) error "no clade to exclude for ${sp} in the registry; pass --exclude_clade"
        def ref_summary = file("${ref_root}/minus_${clade}/summary.json")
        if (!ref_summary.exists()) {
            error "no reference summary at ${ref_summary}\n" +
                  "  The report quotes how many Swiss-Prot entries the reference kept and " +
                  "removed.\n" +
                  "  Pass --reference_cache <dir> pointing at where the run built " +
                  "minus_${clade}/ (make passes REF_CACHE)."
        }

        // Named suffixes, not a glob. A glob over the directory would also sweep in
        // <species>_dark_set.parquet -- one row per dark protein, staged for nothing -- and
        // would quietly pick up whatever else a future arm publishes there, including
        // files this report has no idea how to read.
        def optional_products = [
            "_kmerseek_dark_gain.json", "_kmerseek_dark_gain.parquet",
            "_length_summary.json", "_length_comparison.parquet",
            "_disorder_summary.json", "_disorder.parquet",
        ].collect { file("${dir}/${sp}${it}") }.findAll { it.exists() }

        log.info "  reporting on : ${dir}"
        log.info "  optional arms: " + (optional_products
            ? optional_products*.name.join(', ') : "none found, their sections are omitted")
        tuple(sp, clade, summary, ref_summary, optional_products)
    }

    darkReportFrom(Channel.fromList(rows))
}

/*
 * Fit every stored index for the penalties its alphabet's `extend` arms use now.
 *
 * Reads the index store (--index_cache or --outdir/kmerseek_index), parses alphabet,
 * ksize and scaled off each directory name, and runs `kmerseek calibrate` for each
 * penalty of --calibrate_penalties (`opt` and/or numbers, same spelling as
 * --kmerseek_mismatch_penalty; default `opt`, since C = 2 was fitted at build time).
 * `make calibrate-store-0.4` is this entry. Nothing else in the store is touched.
 */
workflow calibrateStore {
    def store = file(params.index_cache ?: "${params.outdir}/kmerseek_index")
    if (!store.isDirectory()) error "no index store at ${store}"
    def specs = params.calibrate_penalties.toString().tokenize(',')*.trim().findAll { it }
    if (!specs) error "--calibrate_penalties is empty"
    def clades  = params.calibrate_clades.toString().tokenize(',')*.trim().findAll { it }
    def exclude = params.calibrate_exclude.toString()
    def dirs = store.listFiles().findAll { it.isDirectory() && it.name.endsWith('.kmerseek.rocksdb') }
    def skipped = 0
    def items = dirs.findAll { d ->
        def keep = (!clades || clades.any { d.name.startsWith("minus_${it}.") }) && !(exclude && d.name =~ exclude)
        if (!keep) skipped++
        keep
    }.collect { d ->
        def m = d.name =~ /^minus_[A-Za-z]+\.(.+?)\.k(\d+)(?:\.s(\d+))?\.lc(true|false)\.kmerseek\.rocksdb$/
        if (!m) error "cannot parse ${d.name}"
        def alphabet = m[0][1]
        def pens = specs.collect { spec -> spec == 'opt' ? optimalPenalty(alphabet) : penaltyString(spec as double) }.unique()
        tuple(d.toAbsolutePath().toString(), alphabet, m[0][2] as int, (m[0][3] ?: '1') as int, pens)
    }
    log.info "calibrating ${items.size()} stored index(es) under ${store} for penalties ${specs.join(',')}" +
             (skipped ? " (${skipped} left alone: not in --calibrate_clades or matching --calibrate_exclude)" : '')
    items.groupBy { it[1] }.each { a, l -> log.info "  ${a}: ${l.size()} index(es), C = ${l[0][4].join(', ')}" }
    calibrateStoredIndex(Channel.fromList(items))
}

workflow { darkSet() }
