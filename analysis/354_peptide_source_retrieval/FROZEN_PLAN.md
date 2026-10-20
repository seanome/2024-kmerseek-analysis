# 354. Does a reduced-alphabet search find the source protein of a de novo peptide call better than a full-alphabet search?

Frozen 2026-10-20, before any method was run on any data. Changes after this date go in a
dated section at the end, never as edits above it.

## Question

A de novo sequencing run gives a peptide call: a string of 7 to 30 residues read from one
mass spectrum, with errors. For each call, can a search that first merges similar residues
(a reduced alphabet) find the protein the peptide came from with fewer wrong candidates, or
with much less computing, than a search on all 20 residues? This is lookup of the source
protein only. It says nothing about what the peptide does.

## Data the experiment needs

| Input | File | Schema checked by |
|---|---|---|
| Frozen protein collection: human reference proteome plus a common-contaminant list (cRAP) | one FASTA per part, listed in `inputs.json` | `validate` |
| 500 de novo peptide calls from one completed run | `calls.tsv` | `validate` |
| The accepted peptide for each call, from an independent database search, with its evidence | `accepted.tsv` | `validate` |
| Per-residue confidence and alternative residues, when the caller gives them | columns of `calls.tsv` | `validate` |

`inputs.json` also states `data_origin`: `experimental` or `synthetic_fixture`. Any run on
`synthetic_fixture` data ends in INSUFFICIENT_EVIDENCE whatever its numbers are.

### `calls.tsv`

| Column | Required | Meaning |
|---|---|---|
| `call_id` | yes | unique |
| `run_id`, `scan_id` | yes | which spectrum the call came from |
| `called_sequence` | yes | 20 canonical letters plus `X`; modifications stripped |
| `call_score` | yes | the caller's own peptide score, higher is better |
| `residue_scores` | no | comma-separated, one per residue |
| `alternatives` | no | `pos:RES;pos:RES`, 1-based, RES = other residues the caller allowed at that position |

### `accepted.tsv`

| Column | Required | Meaning |
|---|---|---|
| `call_id` | yes | joins to `calls.tsv` |
| `accepted_peptide` | yes | bare sequence from the database search of the same spectrum |
| `evidence_tool` | yes | e.g. MSFragger, Comet, Sage |
| `evidence_qvalue` | yes | the search's PSM q-value (a PSM is one peptide-spectrum match) |
| `evidence_score` | no | the search's score |

Calls whose accepted peptide has a q-value above 0.01 are excluded before anything else,
and the count is reported. Calls that are wrong, partly wrong, or ambiguous stay in. No
call is edited and no error is simulated.

## The valid source set

For each call, the valid source set is every protein in the frozen collection that contains
the accepted peptide, with I and L treated as the same letter (they have the same mass, so
no spectrum can tell them apart). A peptide found in 8 histone proteins has 8 valid sources.
Any one of them counts as correct; none is singled out as the right one. A call whose
accepted peptide is in no protein of the collection is excluded and counted.

## Development and evaluation split

1. Join proteins into groups: two proteins are in one group when some call has both in its
   valid source set (connected components, so the link is transitive).
2. A group goes to development when `sha256("354:" + smallest accession in the group)`,
   read as a number in [0, 1), is below 0.5. Otherwise it goes to evaluation.
3. A call goes where its group goes. No protein is a valid source for calls on both sides.

## Source-absent cases

Before any method runs, 20% of the calls on each side are marked source-absent: those with
the smallest `sha256("354-absent:" + call_id)`, rounded down to a whole number of calls.
For a source-absent call, every protein in its valid source set is removed from the
collection for that call only. Every other call still sees those proteins.

This is done by a per-call exclusion list that each method applies before it ranks or
decides anything. Every score is computed for one candidate at a time, so this gives the
same candidates as building a separate collection without those proteins. A test rebuilds
the collection and checks this on the fixtures.

## Methods

Every method receives the same input: the called sequence, its alternatives, and I = L.
Every candidate any method returns is scored by the same rule, the verifier below.

| Method | What finds the candidates |
|---|---|
| `exact` | Proteins that contain the call, or a version of it with allowed alternatives, as an exact substring. Found through a 5-mer index of the 20-letter collection, then checked letter by letter. |
| `align` | Every protein, scored by local alignment of the call against it (Smith-Waterman, PAM30, gap open 9, gap extend 1: BLAST's settings for short queries). Nothing is skipped. |
| `exact_then_align` | `exact`; when it finds no protein, `align`. |
| `reduced_<alphabet>_k<k>` | Proteins that share at least one k-mer with the call in the reduced alphabet, ranked by the number of positions where the call and the protein fall in the same residue class along the best diagonal. Reported twice: alone (ranked by that count) and with verification (ranked by the verifier). |

Reduced settings, fixed now: `massspec18` k = 5 (merges I/L and Q/K, the pairs a mass
spectrum confuses most), `sdm12` k = 6, `dayhoff6` k = 8. These are the alphabets defined
in kmerseek `src/rust/alphabets.rs`, copied in nb300's `alphabets.py`.

### The verifier

Local alignment of the call against the protein (PAM30, gap open 9, extend 1), where the
score at each call position is the best score over the called residue and its allowed
alternatives, and I = L. `align` runs it on every protein; the seeded methods run it on a
window of the protein around the seed diagonal, 10 residues wider on each side. The verified
score is divided by the call's score against itself, giving a number from 0 to 1 called the
match fraction. Reported candidates are ranked by match fraction. A region E-value is never
used as the chance that an assignment is right.

## Measures

- Reporting threshold `t`: for each method, the smallest match fraction at which at most 1%
  of the development source-absent calls have any candidate at or above `t`. Fitted on
  development calls only, then frozen and applied to evaluation calls.
- Rank of the best valid source: 1 + the number of non-source candidates whose match fraction
  is at or above it. Ties with other valid sources do not count against it; ties with wrong
  proteins do.
- Recovered: a source-present call whose best valid source has rank ≤ 5 and match fraction ≥ `t`.
- Confident false assignment: a source-absent call with any candidate at or above `t`.
- Surviving candidates: every candidate at or above `t`, counted in full. Lists are never cut
  to five to meet gate 3.
- Proportions get Wilson 95% intervals. Differences between two methods on the same calls
  get a paired bootstrap 95% interval (2,000 resamples of calls, seed 354).

## Pass rule

PASS needs all three gates for the best reduced setting with verification:

1. At least 95% of source-present evaluation calls are recovered.
2. At most 1% of source-absent evaluation calls get a confident false assignment.
3. The median number of surviving candidates over source-present evaluation calls is at most 5.

and one of:

- A. Recovery at least 10 percentage points above the strongest full-alphabet method
  (`exact`, `align`, `exact_then_align`), each at its own threshold from the development calls.
- B. End-to-end runtime at least 5 times lower than the fastest full-alphabet method whose
  recovery is no more than 1 percentage point below the reduced setting's, with that method
  also meeting gate 2.

FAIL when any gate fails, when neither A nor B holds, when the reduced setting only gains
before verification, or when a full-alphabet method already meets gates 1 to 3 and the
reduced setting meets neither A nor B.

INSUFFICIENT_EVIDENCE when the data are not experimental, or when fewer than 500 calls
remain after exclusions.

With about 50 source-absent evaluation calls, 0 false assignments still leaves a 95% Wilson
upper bound of 7.1% (0 of 50). Passing gate 2 on the observed count does not show the true rate is below 1%;
the report gives the upper bound beside the count.

## Controls and strata

- Each evaluation call shuffled keeping its residue counts, and shuffled keeping its
  adjacent-residue pairs (an Euler path through the pairs; first and last residue fixed).
  Residue scores move with their residues. A shuffle that is still a substring of some
  protein is redrawn up to 20 times, then dropped and counted. Reported: the share given a
  confident assignment.
- Recovery and false assignments reported separately for: low-complexity calls (two residues
  make up at least half of the call), calls with `X` or with alternatives, calls that differ
  from their accepted peptide versus calls that match it, length bins 7-9, 10-12, 13-16,
  17-20, 21+, and call-score quartiles (edges set on development calls).

## Runtime

Measured per method: index build (wall time, CPU time, peak memory, index size on disk), a
cold batch (a new process that loads the index from disk and runs every call once) and a
warm batch (the same calls again in that process). Cold here means a new process; the
operating-system file cache is not flushed, because that needs root. CPU-hours are user plus
system time. The amortized daily cost assumes one batch of 500 calls a day and one index
build every 30 days: build / 30 + one warm batch.

Compute budget: one CPU, 16 GB memory, 4 hours in total for the full run on the real
collection (about 20,600 human proteins plus about 120 contaminants).
