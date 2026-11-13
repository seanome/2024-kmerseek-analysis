---
name: branchwater-empty-results
description: Use when Branchwater manysearch succeeds but emits a zero-byte result file or CSV DictWriter fails with fieldnames None.
---

On 2026-10-24, production public bivalve workers failed with TypeError because Branchwater0.9.15 can emit a zero-byte CSV when no hashes match. SRR5945967 reproduced the case after processing all354316queries.

Check process exit0 AND the log confirmation of the expected query count before interpreting an empty file as zero matches. A missing file or incomplete log is an error. Write an explicit header-only compressed output for valid empty results, using the pinned version's observed columns. Preserve query empty-sketch counts separately. Reject foreign run IDs, duplicate query/run pairs and nonpositive shared-hash rows. Test zero/one/duplicate/invalid/incomplete cases. Preserve interrupted outputs before rerunning, and validate a real zero-match run before releasing held workers.
