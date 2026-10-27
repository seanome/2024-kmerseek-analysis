---
name: botryllus-reference-identifiers
description: Validates gene-named Botryllus reference sequences without confusing annotated isoforms with quantified sequences. Use when changing Botryllus FASTA/GTF mapping, recovering gene-count preflights, or interpreting coordinate reconstruction failures.
---

# Botryllus reference identifiers

Added 2026-10-25 after the production preflight rejected gene-named reference sequences
because the GTF annotated more than one transcript for some genes.

1. Inspect the FASTA identifiers and both `gene_id` and `transcript_id` namespaces.
   Count actual quantified sequences per gene separately from annotated isoforms.
2. An exact gene identifier can map to itself even when its GTF gene has multiple
   isoforms. For example, FASTA `g304` may be the only quantified sequence for gene
   `g304`, while the GTF contains `transcript-304` and `transcript-305` for that gene.
3. Reject names that resolve to different genes through the two namespaces. Reject
   multiple quantified reference sequences assigned to one gene unless a separately
   reviewed length-aware aggregation method is implemented. Never invent a mapping.
4. Verify every output count against its quantification input and retain the exact
   mapping table. The current integer conversion is part of the frozen pipeline;
   changing it requires a separate analysis decision.
5. Reconstruct coordinate evidence per annotated transcript. Merging all intervals
   by gene can combine mutually exclusive isoforms and falsely report a mismatch.
   A gene-ID mapping check does not establish coordinate equivalence.
6. After a late preflight failure, preserve successful measurements and outputs.
   Use a new recovery directory, retain the failed allocation's diagnosis and peak
   memory, and rerun only failed checks plus relevant integration tests. Never turn
   a failed allocation into a reported successful job.

Implementation and regression checks are in
`scripts/botryllus_shazam/nextflow_stages.py`, `test_nextflow_stages.py`, and
`recover_production_preflight.py`. Resource recovery keeps the larger scheduler peak
when it exceeds the individual command peak.

## Frozen metadata after Git checkout

Added 2026-10-27 after controller 49525592 failed before submitting tasks.
Git normalized the sample TSV from CRLF to LF while the frozen design retained
the original byte hash. Before launching, check the actual archived input against
that hash. Use `frozen_samples.restore_frozen_samples` to restore line endings only
when they reproduce the exact expected hash. Write into a new launch input file;
never change the frozen hash to accept changed labels. Test a changed treatment
label as a rejection case. Preserve the failed launch for diagnosis.
