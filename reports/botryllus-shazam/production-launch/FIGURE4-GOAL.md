# Figure 4 reproduction

The project goal is to reproduce Figure 4. The 64-library Nextflow workflow supplies
independent sequence discovery and gene-abundance evidence toward that goal. Finishing
this workflow alone does not reproduce the figure.

The user supplied the supplementary workbook and appendix on 2026-10-25.
Their measured fingerprints are in `supplied-sources.json`. The workbook is identical
to the source used for the earlier published-count extraction.

Keep these comparisons separate:

- Panels A and B: published differential-gene lists, the six published time/cohort
  comparisons, functional-group memberships, directions, and counts.
- Panel E: the predefined proimmune/cell-repair gene lists, sample memberships,
  expression transformation, signed score, and the authors' tests.
- Our frozen genotype/collection-adjusted contrasts: independent gene and k-mer tests
  that can disagree with the published analysis.
- SPLASH and DE-kupl recovery: sequences linked to paper genes, with ambiguous and
  missing mappings retained. Genome alignment and direct query Pfam calls add sequence
  evidence; reference annotations do not establish query function.

The detailed panel requirements remain in `../FIGURE4-REPRODUCTION.md`.
Do not tune selection rules or change the frozen model to recover the published result.
The exact original reference and Botryllus score implementation remain checks to resolve.
