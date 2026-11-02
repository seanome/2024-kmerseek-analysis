# Do the k-mer routes find the paper's most changed genes? (notebook 361)

Asked 2026-11-02: do SPLASH, DESeq2 on anchor counts, or KaMRaT find the most differentially
expressed genes, and what kinds of genes do they find?

![Reached genes against the paper's fold change, expression and p-value](reached_vs_paper_de.png)

**a, b.** One dot per gene with the paper's baseMean > 0. x: the paper's baseMean (mean normalised
count). y: the paper's DESeq2 log2 fold change, PEC over control. Light grey: not reached. Blue:
reached by SPLASH. Vermillion: reached by DESeq2 on anchor counts (p <= 0.05, 2-fold). **c.** Share
of genes reached in each baseMean bin. **d.** Among genes with baseMean >= 10, share reached in bins
of the paper's p-value. The bins below p = 0.001 hold only 1 to 17 genes per contrast.

Code: `scripts/reached_genes.py` (Sherlock job 50777735, 2026-11-02) and
`scripts/plot_reached_vs_paper_de.py`. Numbers: `reached_genes.tsv` (one row per contrast,
route and gene) and `reached_vs_paper_de_bins.tsv` (the numbers behind c and d).

A gene is reached when one of the route's peptides (20 or more residues) aligns to its old
gene-model protein with E <= 1e-5 and query coverage >= 50 %, the same rule as `../paper_recovery`.

## KaMRaT and DE-kupl

Both ran in all six real contrasts (KaMRaT 1.2 t-test, DE-kupl 1.3.5 limma and DESeq2) and kept
0 k-mers after correction, so they reach no genes. In the simulated control they kept 196,784
(KaMRaT) and 226,785 (DE-kupl limma) k-mers.

## What decides whether a gene is reached: its expression, not its fold change

- Reach rises steeply with expression (c). In young 24 h, SPLASH reaches 7 % of genes with
  baseMean below 1 and 97 % of genes above 1,000; anchor counts reach 1 % and 54 %.
- Reach barely depends on the paper's p-value once expression is held above 10 (d). In young 24 h,
  SPLASH reaches 40 to 64 % in every p-value bin. Only young 2 h shows a trend: SPLASH reaches 62 %
  of the 8 genes with p < 0.001 and 20 % of genes with p > 0.2.
- In a and b, the reached genes sit at the high-expression end, spread over all fold changes. The
  genes with the largest fold changes are mostly lowly expressed and mostly not reached.

So neither route picks out the paper's most changed genes. Both pick out highly expressed genes,
which have the most reads, so any small difference in their k-mers passes a test.

## What kinds of genes

Gene names from the paper's sheet, genes with baseMean >= 10 and a real name, tallied by name
prefix (a rough grouping, not a GO analysis):

| group (name prefix) | young 2 h, SPLASH | young 24 h, anchor counts | all expressed genes | genes with baseMean >= 100 |
|---|---|---|---|---|
| ribosome, translation (RPL, RPS, EEF, EIF) | 7.2 % | 17.7 % | 2.5 % | 8.7-9.7 % |
| cytoskeleton, muscle (ACT, TUB, MYH, MYL, TPM, TNN, ...) | 4.1 % | 10.8 % | 1.5 % | 3.3-3.4 % |
| mitochondria, energy (ATP5, COX, NDUF, UQCR, ...) | 2.8 % | 3.0 % | 1.3 % | 3.2-3.8 % |

SPLASH's genes look like highly expressed genes in general. Anchor counts in young 24 h have more
ribosome and cytoskeleton genes than even the highly expressed set: these are large families of
similar genes, so an anchor shared by several paralogs can reach all of them.

Where the paper's own small p-values are, both routes do reach some of the paper's top genes. In
young 2 h, both reach Cldn1, Mfge8, HMCN1, Uqcrc2, GSTM4 and TPM1 (paper p 0.0006 to 0.005, all
lower in PEC). In young 24 h, both reach COL12A1 and KLK6 (higher in PEC).

## Limits

- "Reached" is a peptide match, not a measured change in that gene.
- The p-value bins in d use the paper's uncorrected p-values; the paper's corrected p-values keep
  almost nothing (see `../kmer_methods/kmer_methods_counts.md`).
