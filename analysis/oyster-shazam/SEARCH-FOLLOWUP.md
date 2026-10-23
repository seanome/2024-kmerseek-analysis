# Searches after full oyster extension

Requested by Olga on 2026-10-23: search every extended sequence using DNA, protein, Dayhoff and HP, then update the report.

- Extension array: 49191753, 22 batches. First ten batches had completed at the initial search setup check.
- Index validation and integration-test job: 49194487 (replaces failed 49194074).
- Search array: 49194526, `afterok:49191753:49194487` (replaces automatically cancelled 49194102), 22 tasks, concurrency two.
- Read and follow current Sherlock instructions at each session and after compaction. No local heavy analysis.

The earlier index job 49190585 built DNA scaled5, then failed because `sig check` requires a picklist. A subsequent `sig manifest` export wrote six manifest records but then crashed (job 49194074, exit 139). The real four-alphabet integration test passed before this application crash. The repaired builder removes the unnecessary export and directly validates exactly six reference names from search results, requiring each library to recover its own full sketch at containment 1. It reuses the already-built DNA directory only after those checks. It builds the other three encodings at scaled5; searches downsample to scaled10 and 100. A real synthetic positive/ambiguous-sequence integration test runs first in the same Slurm job. A failed test prevents downstream searches.

Reference coverage is six trimmed, first-million-pair oyster libraries from the discovery study. This is self-study sequence occurrence, not independent corroboration or global SRA context. The full bivalve inventory has 17,623 runs and is not indexed. Broader searches must be repeated against that collection when available.

Queries are every Compactors row longer than its anchor. Separate record IDs retain batch, anchor, original Compactors row ID and full DNA. Translated sketches use all six reading frames for each DNA record, with no CDS prediction and no concatenation of records. Identical or overlapping sequences may remain as separate records; they are not distinct genes. All nonzero-overlap comparisons are retained, including one-hash matches, which are candidates requiring confirmation. Empty sketches are counted separately at every encoding and scale; they are not negative searches.

Index job: 4 CPUs, 16 GB, 2 hours. Prior DNA index build occupied 11 minutes; original job maximum resident memory was under 6 GB. Search job: 4 CPUs, 16 GB, 4 hours per batch, two concurrent. This is a conservative model estimate, not measured full-batch performance. Query generation and searches run under `/usr/bin/time -v`; inspect these measurements before scaling further.

## Report follow-up

On completion, verify all 22 `search-trimmed-all/batch-NNN/validated.json` files against corresponding extension hashes and CSV hashes. Sum extended records and distinguish record counts from unique DNA and genes. Check all 12 encoding/scale settings per batch. Aggregate query mappings and CSV exports with original IDs, and report numbers with sketches and with overlap for each setting. Preserve reference scope, self-study status and limitations. Add the validated full-extension results to the explorer or a downloadable complete table; do not silently leave it showing only the original 300-anchor subset. Keep original pilot results labelled separately.

Report builder: `analysis/oyster-shazam/build_report.py`; local Site checkout `/Users/olga/code/oyster-shazam-report-site`. Publish privately to the existing project from `.openai/hosting.json`, then commit/push results and report source on PR 202. A thread heartbeat is intended to perform this follow-up, not a process or agent supervisor on Sherlock. Notify only on meaningful change or completion.
