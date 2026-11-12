# First comparisons with Figure 4

Use the new production directory only:
`$SCRATCH/botryllus-shazam-PEC-nextflow-20261025/results`.

1. `counts/genes.tsv.gz`: check all 64 library columns and all 72,113 reference gene
   rows. Compare library totals and gene counts against the supplied HTSeq matrix.
   Kallisto estimates from coding sequences and HTSeq gene counts are different
   measurements; agreement is descriptive and does not establish pipeline equivalence.
2. `models/young_2h/genes/result/all-features.tsv.gz`, and corresponding `young_24h`
   and `young_312h` paths: require `model.complete.txt` and a completed trace entry.
   Join by exact gene ID to the supplied-count adjusted refits, keeping absent and
   untestable genes. Compare effect direction, effect size and adjusted P separately.
   These frozen contrasts contain 4, 4 and 3 genotypes, from 12, 12 and 6 libraries.
3. Join all predefined immune genes and uniquely resolved published hub memberships.
   Retain ambiguous hub labels separately. Show denominators for tested, significant,
   missing and untestable genes. Do not redefine gene sets from the new results.

These outputs support an early gene-level comparison without waiting for SPLASH,
k-mer models, recruited reads, query Pfam or the final evidence join. They do not
cover all six published comparison bins or implement the unresolved Botryllus score.
Keep the supplied-count condition-only reconstruction, supplied-count adjusted
analysis and new read-derived adjusted analysis as separate columns. Their count
sources and normalization must be stated before interpreting differences.

Planning estimate: 2–6 hours after the production controller starts, with child-job
queue delays additional. Reassess from real task traces once quantifications and
models begin. Do not treat this estimate as a promised completion time.
