# Recovering the paper's gene lists (notebook 361)

The paper (Domen et al. 2026) compares Botryllus colonies given a pulsatile electrical current
(PEC) with untreated controls. Its named lists (SI Dataset S1, sheets "annotated DE all groups
young/old") were made at uncorrected p <= 0.05 and |log2 fold change| > 1 on its own
reference-based DESeq2 (a test of read counts per gene). This folder asks how many genes on each
list our reference-free routes reach, against a null fixed in `design/ANALYSIS_PLAN.md`
(2026-10-30) before the comparison was run.

Words used below:
- k-mer: a 31-letter piece of a read. Contig: overlapping k-mers joined into a longer sequence.
- SPLASH: a k-mer tool that finds 31-letter anchors whose following sequence differs between
  samples. Compactor: an anchor extended with the reads that carry it.
- DE-kupl: a k-mer differential tool. It counts every k-mer, tests each one with limma (a linear
  model on log counts) or DESeq2, and joins kept k-mers into contigs.
- KaMRaT: a k-mer tool that tests each k-mer with a t-test and joins neighbouring k-mers with
  similar counts into contigs.
- corrected p: p-value after the Benjamini-Hochberg correction for testing many sequences.
  Uncorrected p: the p-value of each test alone, the paper's own cutoff.

How a gene is reached: a route's sequences (compactors or contigs) are translated, matched by
sourmash to the old gene-model proteins (`botznik`, gene IDs g<N>), and kept when the alignment
check supports the match (E <= 1e-5, query coverage >= 50 %). The null: 1,000 random gene sets of
the list's size, drawn from the genes in the paper's DESeq2 sheet of that contrast, matching the
list's count in each baseMean (mean normalised count) decile. Both list and pool are limited to
genes with an old gene-model protein (all listed genes have one). Code: `scripts/paper_recovery.py`,
figure `scripts/plot_paper_recovery.py`.

## Result, 2026-11-02: KaMRaT and DE-kupl added

![Recovery of the paper's lists by six routes](paper_recovery.png)

Every route except SPLASH has a direction per sequence (higher or lower in PEC), so its up list
is compared only with genes reached by sequences higher in PEC, and its down list with sequences
lower in PEC. SPLASH has no direction and is compared with both lists alike.

How many of the 12 lists (6 contrasts x up and down) each route reaches more often than
baseMean-matched random genes. "Beat random" means a share of random sets at or above the real
list of 0.05 or less; the stricter column corrects for the 12 lists (Bonferroni, 0.05 / 12 =
0.004).

| route | lists scored | beat random (<= 0.05) | after correcting for 12 lists (<= 0.004) |
|---|---|---|---|
| SPLASH anchors, corrected p <= 0.05 | 12 | 3 | 2 |
| anchor counts, DESeq2, p <= 0.05 and 2-fold | 12 | 5 | 4 |
| KaMRaT contigs, t-test, p < 0.05 | 12 | 6 | 1 |
| KaMRaT contigs, DESeq2, p < 0.05 | 12 | 9 | 7 |
| DE-kupl k-mers, limma, p < 0.05 | 10 | 3 | 2 |
| DE-kupl k-mers, DESeq2, p < 0.05 | 6 so far | 4 | 3 |

KaMRaT contigs tested with DESeq2 beat random on the most lists, 9 of 12, and 7 of 12 after
correction. In every contrast it reaches the up list more often than random; it reaches no gene
on the down lists of old 24 h, young 13-49 d and old 41-93 d. Its null sits near zero (it reaches
0.007 to 1.3 % of the genes in the pool), so a list reached by a few genes stands out; the real
lists are reached at 5 to 44 %.

The KaMRaT t-test reaches mostly down-list genes (29 of 98 in young 24 h); with 6 PEC and 6 control
samples a t-test on log counts finds few contigs higher in PEC (young 2 h: 1,972 up, 27,329 down).

DE-kupl's limma test reaches no gene on the up lists of young 2 h, young 24 h, old 24 h and
old 41-93 d. It keeps few k-mers at p < 0.05 (young 2 h: 108,577 of 48 million tested, 0.2 %).

Numbers: `paper_recovery.tsv`, written by `scripts/paper_recovery.py` in Slurm job 50801147
(2026-11-02). A share of random sets of 0.000 means none of the 1,000 reached the real share.
In that run the random draws of SPLASH and anchor counts were not in a fixed order; the
committed script now fixes the order (one stream per route and contrast, deciles and genes
sorted), and its rerun was not copied back before the connection to Sherlock was lost. Rerunning
the old script on the same files moved 12 of 24 shares by 0.001 to 0.022 and moved no list
across 0.05 or 0.004.

### How each new route was made

- KaMRaT contigs, t-test (`kamrat_ttest_p05`): KaMRaT 1.2 merges every k-mer of DE-kupl's count
  table into contigs; it reports only corrected p, which keeps no contig in any real contrast.
  `select_kamrat_nominal.py ... 0` recomputes KaMRaT's own Welch t-test on log2(count + 1) per
  contig and keeps p < 0.05, no fold-change cutoff.
- KaMRaT contigs, DESeq2 (`kamrat_deseq2_p05`): KaMRaT has no DESeq2 test, so its contigs' counts
  are tested with PyDESeq2 the way anchor counts are (`~ pair + condition`; young 132 d
  `~ genotype + condition`, since its samples are not paired; size factors from all contigs;
  contigs with a count in fewer than half the samples are dropped). KaMRaT stores counts scaled
  to 1e9 per sample; `kamrat_contig_counts.py` multiplies them back by each sample's total k-mer
  count / 1e9 and rounds. Check: for 2,000 contigs made of one k-mer, the rebuilt counts match
  that k-mer's raw counts (median difference 6e-7 of the count in young 2 h and young 24 h).
  Kept: p < 0.05, no fold-change cutoff (`select_kamrat_deseq.py`).
- DE-kupl k-mers, limma (`dekupl_limma_p05`): DE-kupl keeps only k-mers at corrected p <= 0.05 and
  2-fold, none in any real contrast, but writes the uncorrected p of every tested k-mer
  (`raw_pvals.txt.gz`), so no rerun was needed. `select_dekupl_nominal.py` keeps p < 0.05,
  rebuilds DE-kupl's fold change exactly as its limma script does (checked: identical for all
  10,419 k-mers DE-kupl kept in the simulated control), splits the k-mers by direction and joins
  each set into contigs with DE-kupl's own `mergeTags -k 31 -n`.
- DE-kupl k-mers, DESeq2 (`dekupl_deseq2_p05`): only the simulated control had a DE-kupl DESeq2
  run. `slurm/19_dekupl_deseq2_p05.sbatch` runs DE-kupl's DESeq2 test step
  (`/dekupl/bin/DESeq2_diff_method.R` in the dekupl-run 1.3.5 image) on the limma run's k-mer
  counts, with its filters opened (corrected p <= 1, |log2FC| >= 0) so every k-mer keeps its
  DESeq2 fold change, then the same selection and joining. Young 2 h (48 M k-mers) and young 24 h
  (53 M) were still running when this was written: 16 of 49 and 16 of 53 chunks of 1 M k-mers
  were done 47 minutes after the start (at 10:53), so the test should end near 13:15; the
  selection, joining and protein search add about an hour.

Direction: up = higher in PEC in every route. Checked on the KaMRaT t-test's strongest up and
down contigs in young 132 d: the up contig has 0 counts in all 3 controls and 84-86 in all 3 PEC
samples, the down contig the reverse.

### DE-kupl in young 132 d: no result

DE-kupl's sample scaling factors for young 132 d (`DEkupl_result/metadata/sample_conditions_full.tsv`)
are 0 for JA162 and JA169 and 1e219 to 1e262 for the other four samples. Its DESeq2 test stops
on them ("all(value > 0) is not TRUE"), and its limma p-values rest on them, so neither DE-kupl
route is scored there. In the other contrasts the factors span 5 to 200-fold between samples
(young 2 h: 0.078 to 15.8; young 24 h: 0.043 to 7.6), wider than the 2.6-3.4x difference in read
depth between control and PEC.

### SPLASH anchors (corrected p <= 0.05)

| list | reached | real share | null median (95 % range) | random sets at or above |
|---|---|---|---|---|
| young 2 h up | 11/43 | 0.26 | 0.19 (0.09-0.33) | 0.220 |
| young 2 h down | 56/237 | 0.24 | 0.17 (0.13-0.22) | 0.004 |
| young 24 h up | 115/252 | 0.46 | 0.41 (0.36-0.46) | 0.037 |
| young 24 h down | 36/98 | 0.37 | 0.33 (0.24-0.41) | 0.180 |
| old 24 h up | 21/57 | 0.37 | 0.04 (0.00-0.09) | 0.000 |
| old 24 h down | 0/3 | 0.00 | 0.00 (0.00-0.33) | 1.000 |
| young 13-49 d up | 2/18 | 0.11 | 0.06 (0.00-0.17) | 0.204 |
| young 13-49 d down | 4/48 | 0.08 | 0.04 (0.00-0.10) | 0.151 |
| young 132 d up | 3/40 | 0.07 | 0.03 (0.00-0.10) | 0.115 |
| young 132 d down | 1/46 | 0.02 | 0.02 (0.00-0.09) | 0.747 |
| old 41-93 d up | 1/19 | 0.05 | 0.00 (0.00-0.16) | 0.479 |
| old 41-93 d down | 0/17 | 0.00 | 0.00 (0.00-0.12) | 1.000 |

Beat random: young 2 h down, young 24 h up, old 24 h up; after correction young 2 h down and
old 24 h up.

### Anchor counts, DESeq2, at the paper's cutoff (p <= 0.05, |log2FC| > 1)

The anchors whose anchor-count DESeq2 fit passes the paper's own cutoff (primary design
~ pair + condition; young 132 d ~ genotype + condition) were extended into compactors
(process `compactors_nominal`) and matched to proteins like the SPLASH route.

| contrast | anchors fitted | kept | up | down | compactors |
|---|---|---|---|---|---|
| young_2h | 4654616 | 34741 | 7509 | 27232 | 25367 |
| young_24h | 4129843 | 26245 | 8372 | 17873 | 36107 |
| old_24h | 814823 | 51565 | 18068 | 33497 | 53007 |
| young_13_49d | 210910 | 6772 | 5699 | 1073 | 2799 |
| young_132d | 244983 | 12998 | 6056 | 6942 | 1968 |
| old_41_93d | 220616 | 3940 | 3048 | 892 | 1811 |

| list | reached | real share | null median (95 % range) | random sets at or above |
|---|---|---|---|---|
| young 2 h up | 6/43 | 0.14 | 0.02 (0.00-0.09) | 0.002 |
| young 2 h down | 40/237 | 0.17 | 0.05 (0.03-0.08) | 0.000 |
| young 24 h up | 10/252 | 0.04 | 0.02 (0.00-0.04) | 0.023 |
| young 24 h down | 1/98 | 0.01 | 0.02 (0.00-0.06) | 0.896 |
| old 24 h up | 22/57 | 0.39 | 0.02 (0.00-0.05) | 0.000 |
| old 24 h down | 0/3 | 0.00 | 0.00 (0.00-0.33) | 1.000 |
| young 13-49 d up | 1/18 | 0.06 | 0.00 (0.00-0.11) | 0.292 |
| young 13-49 d down | 0/48 | 0.00 | 0.00 (0.00-0.02) | 1.000 |
| young 132 d up | 5/40 | 0.12 | 0.03 (0.00-0.07) | 0.002 |
| young 132 d down | 1/46 | 0.02 | 0.02 (0.00-0.07) | 0.613 |
| old 41-93 d up | 0/19 | 0.00 | 0.00 (0.00-0.05) | 1.000 |
| old 41-93 d down | 0/17 | 0.00 | 0.00 (0.00-0.06) | 1.000 |

Beat random: young 2 h up and down, young 24 h up, old 24 h up, young 132 d up; after correction
all but young 24 h up.

### KaMRaT contigs, t-test (p < 0.05)

| contrast | contigs tested | p < 0.05 | up | down |
|---|---|---|---|---|
| young_2h | 4442363 | 29301 | 1972 | 27329 |
| young_24h | 6432394 | 118268 | 2444 | 115824 |

| list | reached | real share | null median (95 % range) | random sets at or above |
|---|---|---|---|---|
| young 2 h up | 1/43 | 0.02 | 0.00 (0.00-0.02) | 0.050 |
| young 2 h down | 9/237 | 0.04 | 0.02 (0.00-0.03) | 0.017 |
| young 24 h up | 1/252 | 0.00 | 0.00 (0.00-0.00) | 0.115 |
| young 24 h down | 29/98 | 0.30 | 0.07 (0.03-0.12) | 0.000 |
| old 24 h up | 0/57 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| old 24 h down | 0/3 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| young 13-49 d up | 2/18 | 0.11 | 0.00 (0.00-0.06) | 0.012 |
| young 13-49 d down | 0/48 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| young 132 d up | 1/40 | 0.03 | 0.00 (0.00-0.03) | 0.030 |
| young 132 d down | 6/46 | 0.13 | 0.04 (0.00-0.11) | 0.017 |
| old 41-93 d up | 1/19 | 0.05 | 0.00 (0.00-0.05) | 0.101 |
| old 41-93 d down | 0/17 | 0.00 | 0.00 (0.00-0.06) | 1.000 |

Beat random: young 2 h up (0.050) and down, young 24 h down, young 13-49 d up, young 132 d up and
down; after correction young 24 h down. The same contigs at the earlier 2-fold cutoff
(`kamrat_nominal`, in the table, not drawn) give the same reached counts in every list.

### KaMRaT contigs, DESeq2 (p < 0.05)

| contrast | contigs fitted | p < 0.05 | up | down | corrected p <= 0.05 |
|---|---|---|---|---|---|
| young_2h | 2015561 | 27781 | 9694 | 18087 | 24 |
| old_24h | 643205 | 9246 | 2684 | 6562 | 0 |
| young_13_49d | 224704 | 5415 | 5142 | 273 | 0 |
| young_132d | 177389 | 12741 | 7410 | 5331 | 452 |
| old_41_93d | 246989 | 2191 | 1149 | 1042 | 5 |

(young 24 h: `contrasts/young_24h/kamrat_deseq2_p05/summary.json` on Sherlock, not copied back.)

| list | reached | real share | null median (95 % range) | random sets at or above |
|---|---|---|---|---|
| young 2 h up | 6/43 | 0.14 | 0.00 (0.00-0.05) | 0.000 |
| young 2 h down | 16/237 | 0.07 | 0.02 (0.00-0.03) | 0.000 |
| young 24 h up | 12/252 | 0.05 | 0.01 (0.00-0.03) | 0.000 |
| young 24 h down | 6/98 | 0.06 | 0.01 (0.00-0.04) | 0.008 |
| old 24 h up | 6/57 | 0.11 | 0.00 (0.00-0.02) | 0.000 |
| old 24 h down | 0/3 | 0.00 | 0.00 (0.00-0.67) | 1.000 |
| young 13-49 d up | 8/18 | 0.44 | 0.06 (0.00-0.17) | 0.000 |
| young 13-49 d down | 0/48 | 0.00 | 0.00 (0.00-0.02) | 1.000 |
| young 132 d up | 4/40 | 0.10 | 0.00 (0.00-0.03) | 0.000 |
| young 132 d down | 8/46 | 0.17 | 0.04 (0.00-0.11) | 0.000 |
| old 41-93 d up | 3/19 | 0.16 | 0.00 (0.00-0.11) | 0.005 |
| old 41-93 d down | 0/17 | 0.00 | 0.00 (0.00-0.06) | 1.000 |

Beat random: 9 lists, all but the down lists of old 24 h, young 13-49 d and old 41-93 d; after
correction 7 (young 24 h down at 0.008 and old 41-93 d up at 0.005 drop out).

### DE-kupl k-mers, limma (p < 0.05)

| list | reached | real share | null median (95 % range) | random sets at or above |
|---|---|---|---|---|
| young 2 h up | 0/43 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| young 2 h down | 13/237 | 0.05 | 0.01 (0.00-0.03) | 0.000 |
| young 24 h up | 0/252 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| young 24 h down | 13/98 | 0.13 | 0.05 (0.01-0.09) | 0.001 |
| old 24 h up | 0/57 | 0.00 | 0.00 (0.00-0.02) | 1.000 |
| old 24 h down | 0/3 | 0.00 | 0.00 (0.00-0.33) | 1.000 |
| young 13-49 d up | 2/18 | 0.11 | 0.00 (0.00-0.06) | 0.010 |
| young 13-49 d down | 0/48 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| young 132 d | no result (scaling factors) | | | |
| old 41-93 d up | 0/19 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| old 41-93 d down | 1/17 | 0.06 | 0.00 (0.00-0.06) | 0.144 |

Beat random: young 2 h down, young 24 h down, young 13-49 d up; after correction young 2 h down
and young 24 h down.

### DE-kupl k-mers, DESeq2 (p < 0.05), partial

| list | reached | real share | null median (95 % range) | random sets at or above |
|---|---|---|---|---|
| young 2 h, young 24 h | still running (see above) | | | |
| old 24 h up | 18/57 | 0.32 | 0.00 (0.00-0.04) | 0.000 |
| old 24 h down | 0/3 | 0.00 | 0.00 (0.00-0.00) | 1.000 |
| young 13-49 d up | 4/18 | 0.22 | 0.00 (0.00-0.11) | 0.000 |
| young 13-49 d down | 3/48 | 0.06 | 0.00 (0.00-0.04) | 0.002 |
| young 132 d | no result (scaling factors) | | | |
| old 41-93 d up | 2/19 | 0.11 | 0.00 (0.00-0.05) | 0.015 |
| old 41-93 d down | 0/17 | 0.00 | 0.00 (0.00-0.00) | 1.000 |

Beat random: 4 of the 6 lists scored so far; after correction 3.

## The same routes on the simulated control

`../control/control_recall.png` (`scripts/control_truth.py`, summary
`../control/control_truth_summary_p05.json`): 3,000 simulated genes, 6 control and 6 PEC samples
in pairs, 150 genes raised 4x, 150 lowered 4x, 100 with two isoforms switched 9x, 2,600 unchanged.
A gene is found when a kept sequence sits in that gene only; every unchanged gene found is a false
find.

| route | raised 4x | lowered 4x | isoforms switched | unchanged found (false) |
|---|---|---|---|---|
| KaMRaT contigs, DESeq2, p < 0.05 | 147/150 | 129/150 | 85/100 | 149/2600 |
| KaMRaT contigs, DESeq2, corrected p <= 0.05 | 136/150 | 104/150 | 76/100 | 14/2600 |
| KaMRaT contigs, t-test, p < 0.05 | 148/150 | 139/150 | 91/100 | 275/2600 |
| DE-kupl k-mers, limma, p < 0.05 | 148/150 | 135/150 | 88/100 | 1013/2600 |
| DE-kupl k-mers, DESeq2, p < 0.05 | 148/150 | 140/150 | 93/100 | 914/2600 |
| anchor counts, DESeq2, corrected p <= 0.05, 2-fold | 121/150 | 80/150 | 54/100 | 380/2600 |

KaMRaT + DESeq2 finds nearly as many planted genes as the DE-kupl routes at p < 0.05 with about a
sixth of their false finds; at corrected p it keeps 14 false finds. In the control every
sequence is either changed or not; the real contrasts have far fewer changed genes, so the share
of false finds among kept sequences there is likely higher.

## Run time and memory

Asked for: wall time, CPU hours and peak memory of each route, per contrast. Done so far, from
the Nextflow traces of `nextflow-runs/pec-domen2026` (runs 2026-10-31 and 2026-11-01; CPU hours =
realtime x %cpu): the steps the three large contrasts share.

| step | contrast | wall | CPU hours | peak memory | used by |
|---|---|---|---|---|---|
| SPLASH | young 2 h | 18 min | 2.5 | 41 GB | SPLASH, anchor counts |
| SPLASH | young 24 h | 14 min | 2.1 | 84 GB | SPLASH, anchor counts |
| SPLASH | old 24 h | 3 min | 0.4 | 81 GB | SPLASH, anchor counts |
| compactors | young 2 h / 24 h / old 24 h | 10 / 17 / 25 min | 0.3 / 1.0 / 0.5 | 23 / 42 / 56 GB | SPLASH |
| protein search of compactors | young 2 h / 24 h / old 24 h | 22 / 58 / 7 min | 1.2 / 3.3 / 0.3 | 30 / 67 / 31 GB | SPLASH |
| anchor counts + PyDESeq2 (3 designs) | young 2 h / 24 h / old 24 h | 2.0 / 2.2 / 0.2 h | 14.6 / 20.3 / 1.7 | 27 / 24 / 6 GB | anchor counts |
| compactors of kept anchors | young 2 h / 24 h / old 24 h | 4 / 3 / 26 min | 0.1 / 0.2 / 0.6 | 11 / 17 / 55 GB | anchor counts |
| protein search, kept anchors | young 2 h / 24 h / old 24 h | 3 / 6 / 6 min | 0.1 / 0.2 / 0.3 | 5 / 4 / 11 GB | anchor counts |
| DE-kupl counting + limma | young 2 h / 24 h / old 24 h | 86 / 73 / 11 min | 7.8 / 7.4 / 1.6 | 25 / 25 / 4 GB | every DE-kupl and KaMRaT route |
| KaMRaT index, t-test, merge | young 2 h / 24 h / old 24 h | 53 / 62 / 4 min | 0.9 / 1.0 / 0.07 | 10 / 12 / 0.9 GB | both KaMRaT routes |

The new steps (KaMRaT t-test selection, contig counts + PyDESeq2, DE-kupl selection and joining,
DE-kupl's DESeq2 test, and their protein searches) wrote a `/usr/bin/time -v` file per step on
Sherlock; `scripts/route_resources.py` collects them, the traces and Slurm's records (`sacct`)
into `resources.tsv`. It was not run: the connection to Sherlock was lost first. Whole Slurm
tasks, which include all three new routes' steps for one contrast: young 2 h 25 min, young 24 h
34 min, old 24 h 18 min, the small contrasts 4-5 min (16 CPUs each; job 50795835, with the DE-kupl
limma part redone in job 50798824 in 1-4 min per contrast).

Limits:
- old 24 h is 2 control and 2 PEC samples from one colony, so it is low power and says nothing
  beyond that colony (ANALYSIS_PLAN.md).
- SPLASH treats every sample as independent, so its anchors in young 2 h and young 24 h, where
  samples come in pairs, are likely too many.
- The paper's lists and our routes come from the same reads. Agreement is corroboration, not
  an independent repeat.
- Reaching a gene means a peptide of 20 or more residues matched its protein. A gene family with
  close paralogs can be reached through a paralog.
- At uncorrected p < 0.05 about 1 tested sequence in 20 passes by chance. The routes that reach
  few genes in the pool have a null near zero, so a list reached by 3 or 4 genes already beats it.
- No route is a full reproduction: most listed genes are not reached, and the lists themselves
  did not pass correction in the paper's own DESeq2 (0 genes at corrected p <= 0.05 in 4 of 6
  contrasts).
