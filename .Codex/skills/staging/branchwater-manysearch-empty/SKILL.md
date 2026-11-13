---
name: branchwater-manysearch-empty
description: Handle empty results and native CSV columns when changing Botryllus search or evidence joins using Sourmash 4.9.4 and Branchwater 0.9.15.
---

Check the pinned executable with identical and disjoint signature collections before
changing candidate retrieval. On Branchwater 0.9.15, `manysearch -A` includes
zero-overlap comparisons for ZIP targets. For positive-overlap candidates, omit `-A`
and explicitly retain rows with `intersect_hashes > 0`. Keep raw comparisons separately.

A successful search against a wholly disjoint target writes a zero-byte CSV. After
checking the process exit status, preserve that raw file and write an explicit empty
result table with a known schema. Reject malformed nonempty tables; do not turn a
failed command or missing output into zero matches.

Native CSVs contain `scaled`. When adding search metadata, merge dictionaries with
explicit field precedence instead of passing both `scaled=` and `**row` to `dict()`.
Keep regression fixtures using the native columns and both matching/disjoint targets.

Added 2026-10-25 during PR 203. Job 49513576 reached the final evidence join and failed
on the duplicate `scaled` argument. Job 49514822 caught the zero-byte result in the
pinned disjoint-target test before running the workflow. Historical search outputs
remain unchanged; these fixes apply to the new migration protocol.
