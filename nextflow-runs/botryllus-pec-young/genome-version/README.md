# Is the genome version why we cannot reproduce Domen et al. 2026?

The paper counted reads on an older Botryllus genome (`botznik-chr.fa`, gene models
`nc_nsi_nf.gtf`, 72,113 genes named `g1`, `g2`, ...) with HTSeq. Our runs count on the
April 2024 genome (35,541 genes named `FUN_000001`, ...) with Salmon. So two things
changed at once: the genome and gene models, and the read counter. This folder asks how
much of the gap each one explains.

The question is how many of the paper's 782 listed genes (Dataset S1, sheet
"annotated DE all groups young", four young-colony comparisons) each count table finds:
p < 0.05 with the fold change in the paper's direction. Every table is fitted the same
way, as the paper did: each comparison alone, `~ treatment`, DESeq2's default size
factors, the paper's 38 libraries.

![Listed genes found per genome and counter; the paper's found genes by genome](figures/genome_version_domen2026_dataset_s1.png)

| Listed genes found, of 782 | Counted as HTSeq does | Counted by Salmon |
|---|---:|---:|
| The paper's own table (HTSeq) | 631 | |
| Our reads, paper's genome (`g` genes) | 280 | 155 |
| Our reads, April 2024 genome (`FUN_` genes) | 187 | 183 |

"Counted as HTSeq does" is featureCounts with HTSeq's rule: a read pair counts for a gene
only when it aligns to one place and overlaps that one gene. Strand is ignored, because
the libraries are unstranded.

- **Genome version: part of the gap, not most of it.** With HTSeq-style counting,
  going back to the paper's genome finds 280 genes instead of 187, so 93 more. The gap
  between the paper's table and the new genome is 444 (631 - 187), so 93 is 21% of it.
  With Salmon the paper's genome finds 28 fewer than the new one.
- Gene by gene (panel b) the genome matters more than the net 93 shows. Of the 631
  genes the paper's table finds, 153 are found only on the paper's genome and 59 only on
  the April 2024 genome.
- **HTSeq instead of Salmon: matters only on the paper's genes.** On the paper's genome,
  HTSeq-style counting finds 125 more than Salmon (280 against 155). On the April 2024
  genome the two differ by 4 (187 against 183).
- **Most of the gap is left even with the paper's genome and the paper's way of
  counting.** The gap is still 351 genes (631 - 280), and 308 of the 631 genes the
  paper's table finds are found on neither genome. Per library, our HTSeq-style count on the
  paper's genome holds 62% as many read pairs as the paper's table (median over 38
  libraries; PR 223, `count_split_libraries.tsv`).
- Over the listed genes, the rank correlation of signed -log10 p with the paper's fit is
  0.86 on the paper's genome and 0.77 on the April 2024 genome (HTSeq-style; 0.81 and
  0.75 with Salmon). These genes were picked by the paper's p-values, so this is not a
  correlation over all genes.

What is left points at the paper's own counting run, not the genome. In PR 223 the
paper's HTSeq tallies for one library (SRR32233970) add up to 28.1 million reads for 18.1
million read pairs. That fits HTSeq counting the two mates of a pair as two reads, which
would raise every count up to 2-fold. PR 223 is testing that now (Sherlock job 52056949,
written before it ran): counting each mate as its own read should find at least 456
listed genes on the paper's genome if this is the main cause, and fewer than 368 if it is
not.

## Files

- `genome_version.py`: reads `inputs/`, prints the numbers above to
  `genome_version.out`, writes the two tables and the figure.
  Run with `python genome_version.py` in the 2025-kmerseek-analysis environment.
- `inputs/count_split_*.tsv`: copied unchanged from
  `/scratch/users/olgabot/botryllus-pec-young/count-split/` on Sherlock, written by PR
  223's `count_split.R` (job 51959161, 2026-11-11). PR 223 has the code that aligned the
  reads to the paper's genome (nf-core/rnaseq 3.27.0, `params-rnaseq-paper-genome.yaml`)
  and fitted the tables.
- `genome_version_summary.tsv`: listed genes found and tested per table, and the rank
  correlation with the paper's fit.
- `genome_version_split.tsv`: the paper's 631 found genes by which genome finds them.
