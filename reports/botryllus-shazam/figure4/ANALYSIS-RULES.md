# Figure 4 reproduction and adjusted comparison

Written before running new models on the supplied counts, 2026-10-24.

The workbook is an external published result, not an independent validation. Preserve its counts, DESeq2 tables, curated gene lists and manually declared hub counts separately.

Replot all six visible functional groups in Figure 4A–B from the supplementary count sheet. Report the omitted Other category in the download. This redraw is not an independent reproduction of the gene selection or functional annotation.

Refit the six supplied comparison groups on their supplied HTSeq gene counts with DESeq2 `~ condition`, default size-factor estimation and dispersion fitting, Treatment versus Control. Drop all-zero genes; exclude libraries with fewer than 10,000 assigned gene counts. Preserve both raw-P and BH-adjusted-P counts at ≤0.05 and absolute log2 fold change >1. Compare gene-level effects and P values to the supplied tables and retain disagreements. The original DESeq2 version, formula and filtering settings are not supplied; this is a condition-only reconstruction, not a claim of exact original code execution.

Separately apply the already frozen genotype/collection design to the same supplied counts: three young contrasts at 2 h, 24 h and 13 d; paired collection strata summed within each genotype/condition; one genotype is one unit; ≥5 counts in ≥2 units; fit `~ genotype + condition`; global BH over all eligible genes per contrast without independent filtering; primary significance adjusted P strictly <0.05. No new fold-change cutoff. Collection means the recorded genotype, age and treatment-time stratum, not a verified calendar date. Report the 4/4/3 genotype denominators.

This supplied-count analysis is independent of the Kallisto-derived gene counts and DE-kupl k-mer counts in the full-depth branch. Do not merge their count matrices or treat them as independent biological replicates.

Retain all 146 supplementary immune gene IDs (13 proimmune, 133 cell-repair), including non-significant and missing genes. Do not tune the list to obtain the published score pattern. Companion mammalian score code is pinned for inspection, but the Botryllus-specific transformed expression matrix, weighting/cluster implementation and per-sample score output were not found in that repository. Panel E exact reproduction remains unresolved until those details are established. Any reconstructed score must label the transform and deviations explicitly.

For recovery, compare exact Botryllus gene IDs from supported SPLASH mappings and exact DE-kupl 31-mer matches to the reference transcriptome. Count ambiguous/multi-gene hits separately. Gene-name similarity and Swiss-Prot sketch matches do not establish recovery. Join the genotype-adjusted effect and adjusted P, and distinguish sequence recovery, significant k-mer abundance, gene-level DE, and splice/sequence-change candidates.
