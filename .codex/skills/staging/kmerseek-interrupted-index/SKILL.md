---
name: kmerseek-interrupted-index
description: Resume a kmerseek v0.4.0 index/search after a Slurm memory kill, especially when CURRENT exists but calibration failed.
---

Observed 2026-10-23 in notebook 301, PR 182: seven 96 GiB jobs saved RocksDB and then
ran out of memory during the 200-query Karlin–Altschul fit. Nine others failed in search.

1. Read sacct step records and per-command GNU time logs. A killed RSS is capped by the
   allocation and is only a lower bound. Record the failed stage as calibration or search.
2. Do not treat idx/CURRENT as completion. Require `Indexing completed successfully!`
   in index.log. Preserve an interrupted index and its logs; rebuild it in a clean path.
3. Size calibration separately from indexing: these failures occurred after RocksDB
   had finished. PR 182 requests 256 GiB for calibration failures and 192 GiB for search
   failures (previous failed request 96 GiB). Those requests have not yet succeeded.
4. Preserve chunk boundaries on resume. Changing 100-sequence chunks to 25 while reusing
   chunk000 output assigns old results to different queries. Use a separate output directory
   if changing boundaries, or check a manifest of query IDs per chunk.
5. Promote filtered output only after both search and filter succeed. Run gzip integrity
   checks before reuse; clean up a FIFO reader when search fails.
6. Recompute requests from the current logs at submission. Keep completed settings out
   of the submission list. Tests for the interrupted-CURRENT and completed-chunk paths
   live in tests/test_301_recovery.py.
