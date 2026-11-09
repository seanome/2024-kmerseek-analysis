# Plan: the Fig. 4C-D score from reads, with no genome (notebook 361)

Written 2026-11-09, before any read was counted for it. Results go in
`exports/fig4_score_from_reads/README.md`; changes to this plan get a dated note at the end.

## Question

Domen et al. 2026 Fig. 4C-D shows one score per sample in 8 exercise datasets. Three of them are
RNA sequencing (RNA-seq): GSE120862 (panel C), GSE99963 and GSE144304 (panel D). PR 212 got the
printed numbers back from the authors' gene tables. Can the same score, on the same samples, be made
from the reads alone, with no genome and no alignment?

## Method

1. **Reads per gene.** `pec-tools gene-counts` on the fastp-trimmed reads, single-end, with its
   default settings: every window of 10 amino acids (a protein 10-mer) of a read's six translations
   is looked up among the 10-mers of the GENCODE v50 proteins; 10-mers found in two genes are left
   out; a read goes to the gene with the most matching windows if that is 3 or more. A gene is its
   GENCODE gene symbol, in upper case.
2. **Expression.** log2(CPM + 1), where CPM is a gene's reads per million reads that went to any
   gene. PR 212 used log2(FPKM + 1) for GSE120862, as the authors did; here all three series get CPM,
   because the score divides each gene by its own spread, so a gene's length matters little.
3. **Score.** The formula of Voskoboynik et al. 2024, as written in `scripts/exercise_score.py` of
   PR 212: for each listed gene, value = (expression - (StepMiner threshold + 0.5)) / 3 / sd, with
   the threshold and sd taken over every sample of the series; a cluster's value is the sum over its
   genes; score = -1 x cluster 13 + 1 x cluster 14 + 2 x cluster 3. The 338 genes are the lists in
   `design/exercise_score/`. A listed gene with no read in any sample is left out.
4. **Groups.** The authors' groups, the ones that gave the printed numbers in PR 212: GSE120862 all
   untrained against all trained samples (42 + 42); GSE99963 before against after, both arms
   (14 + 14); GSE144304 young against fit old adults (26 + 30). AUC is the chance that a sample of
   the second group scores above one of the first; 0.5 means no difference.

## Rules, fixed now

| rule | what is measured | passes when |
|---|---|---|
| 1. Printed AUC comes back | AUC of the score from reads, per series | on the printed side of 0.5 and within 0.15 of the printed AUC (0.08, 0.91, 0.86) |
| 2. Same samples score the same | Spearman rank correlation, per series, between each sample's score from reads and its score from the gene table (PR 212's `exercise_score_samples.tsv`) | 0.8 or higher |
| 3. The listed genes matter | 1,000 random gene sets per series: each listed gene is swapped for a gene from the same tenth of mean expression, clusters and weights kept. Share of sets whose AUC is as far or farther from 0.5 on the printed side | 0.05 or lower |

"Reproduced from reads" needs rule 1 in all three series. Rule 2 says whether it is the same score
sample by sample, and is reported beside it. Rule 3 does not change that verdict. It says whether
the panel depends on these 338 genes or would look the same with any genes, which is a statement
about the figure as much as about this method. The random draws use seed 361. Rule 1 is the rule
PR 213 used for the microarray series, so the two can be read together.

A swapped-label control does not apply: the score never sees the group labels.

## Fewer reads, and cost

The same three rules are scored on the first 1/10, 1/100 and 1/1000 of each sample's trimmed reads
(the first reads of each file, not a random draw). The number of reads comes from fastp's report.
Cost is measured with `/usr/bin/time -v` on each counting task: CPU time (user + system), wall time
and peak memory. The cost reported for "Shazam for Biology" is that of the smallest share of reads
at which rule 1 passes in all three series.

What cost is compared with: no alignment run exists for these human series. The only measured
alignment cost is on the Botryllus samples (`exports/sizes/alignment_vs_kmer_cost.tsv`: trim, STAR,
RSEM and DESeq2 took 70.6 CPU h for 643.5 M read pairs). The result will give CPU seconds per
million reads for both, and say that the two were measured on different reads.

## Order of runs

The 1/1000 and 1/100 shares run first, as a check that the pipeline works and to size the two larger
runs from measured time and memory. All four shares are scored by the rules above whatever the
small ones show.

## Not covered

- The five microarray datasets of Fig. 4C-D (PR 213).
- Panel E, the Botryllus samples.
- Whether the groups are the ones the panel's words describe (PR 212 found that they are not, for
  GSE120862 and GSE144304).
- The gene tables come from the same reads, so agreement shows the two ways of counting agree, not
  that the biology is right.
