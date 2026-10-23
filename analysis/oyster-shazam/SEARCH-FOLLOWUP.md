# Searches after full oyster extension

Requested by Olga on 2026-10-23: search every extended sequence using DNA, protein, Dayhoff and HP, then update the report.

- Extension array: 49191753, 22 batches. All 22 batches completed successfully by the 08:32 PDT follow-up. Independent aggregation is queued.
- Index validation and integration-test job: 49201967 (replaces 49194487, OOM-killed at 16 GB).
- Search array: 49201976, `afterok:49201967` (replaces automatically cancelled 49194526), 22 tasks, concurrency two.
- Read and follow current Sherlock instructions at each session and after compaction. No local heavy analysis.

The earlier index job 49190585 built DNA scaled5, then failed because `sig check` requires a picklist. A subsequent `sig manifest` export wrote six manifest records but then crashed (job 49194074, exit 139). The real four-alphabet integration test passed before this application crash. The repaired builder removes the unnecessary export and directly validates exactly six reference names from search results, requiring each library to recover its own full sketch at containment 1. It reuses the already-built DNA directory only after those checks. It builds the other three encodings at scaled5; searches downsample to scaled10 and 100. A real synthetic positive/ambiguous-sequence integration test runs first in the same Slurm job. A failed test prevents downstream searches.

Reference coverage is six trimmed, first-million-pair oyster libraries from the discovery study. This is self-study sequence occurrence, not independent corroboration or global SRA context. The full bivalve inventory has 17,623 runs and is not indexed. Broader searches must be repeated against that collection when available.

Queries are every Compactors row longer than its anchor. Separate record IDs retain batch, anchor, original Compactors row ID and full DNA. Translated sketches use all six reading frames for each DNA record, with no CDS prediction and no concatenation of records. Identical or overlapping sequences may remain as separate records; they are not distinct genes. All nonzero-overlap comparisons are retained, including one-hash matches, which are candidates requiring confirmation. Empty sketches are counted separately at every encoding and scale; they are not negative searches.

Original index request: 4 CPUs, 16 GB, 2 hours; see the memory correction below. Prior DNA index build occupied 11 minutes; original job maximum resident memory was under 6 GB. Search job now requests 4 CPUs, 32 GB, 4 hours per batch, two concurrent. This is a conservative model estimate, not measured full-batch performance. Query generation and searches run under `/usr/bin/time -v`; inspect these measurements before scaling further.

## Report follow-up

On completion, verify all 22 `search-trimmed-all/batch-NNN/validated.json` files against corresponding extension hashes and CSV hashes. Sum extended records and distinguish record counts from unique DNA and genes. Check all 12 encoding/scale settings per batch. Aggregate query mappings and CSV exports with original IDs, and report numbers with sketches and with overlap for each setting. Preserve reference scope, self-study status and limitations. Add the validated full-extension results to the explorer or a downloadable complete table; do not silently leave it showing only the original 300-anchor subset. Keep original pilot results labelled separately.

Report builder: `analysis/oyster-shazam/build_report.py`; local Site checkout `/Users/olga/code/oyster-shazam-report-site`. Publish privately to the existing project from `.openai/hosting.json`, then commit/push results and report source on PR 202. A thread heartbeat is intended to perform this follow-up, not a process or agent supervisor on Sherlock. Notify only on meaningful change or completion.

## Memory correction, 2026-10-23

Job 49194487 was OOM-killed at 16 GB during six-library dense-sketch self-search (16,776,808 KiB recorded). It did not validate the production index, so dependent searches were cancelled. Replacement 49201967 uses the existing scaled100 source sketches for exact self-containment validation, rather than loading all six dense source sketches. It requests 32 GB, twice the failed ask; search tasks also request 32 GB. The real synthetic integration test still exercises all twelve encoding/scale combinations. `summarize_extensions.py` independently checks all batch hashes, selection coverage and counts before exporting every extended record and its source ID. A full synthetic 213,658-anchor partition test passed and a tampered output was rejected. Production aggregation remains pending.
