# The Fig. 4C-D score from reads, with no genome: result (notebook 361)

Plan and rules, written before any count: `../../design/fig4_score_from_reads/README.md`.
Status 2026-11-10: all four shares of the reads are counted (Sherlock head jobs 51700536 and
51707972, 14 counting tasks, 0 failed).

**Verdict: not reproduced from reads by the plan's rule, which needs all three series. Two of three
come back, and only with all reads.** GSE99963 gives 0.89 (printed 0.91) and GSE144304 0.87 (printed
0.86). GSE120862 gives 0.72 against a printed 0.08, the opposite direction, at every share of reads.
With 1/1000 of the reads no series comes back.

![Score from the gene table, score from reads, and AUC by number of reads](fig4_score_from_reads.png)

**a.** Each sample's score from the authors' gene table (PR 212); black tick, the group's median.
**b.** The same samples' score from reads counted with no genome, all reads. **c.** AUC at each share
of reads. AUC is the chance that a sample of the second group scores above one of the first; 0.5
means no difference. The yellow band is rule 1: within 0.15 of the printed AUC.

## The three rules at each share of reads

From `fig4_reads_comparisons.tsv`. Reads are the first reads of each file, not a random draw.

| series | share of reads | reads placed in a gene, median per sample | AUC from reads | printed AUC | rule 1 | rank correlation with the gene-table score (rule 2 needs 0.8) | random gene sets as far (rule 3 needs 0.05 or less) |
|---|---|---|---|---|---|---|---|
| GSE120862 | all | 26,354,151 | 0.72 | 0.08 | fails | 0.04 | 0.000 |
| GSE120862 | 1/10 | 2,634,480 | 0.69 | 0.08 | fails | 0.09 | 0.000 |
| GSE120862 | 1/100 | 264,247 | 0.62 | 0.08 | fails | 0.20 | 0.004 |
| GSE120862 | 1/1000 | 26,487 | 0.58 | 0.08 | fails | 0.22 | 0.159 |
| GSE99963 | all | 7,088,086 | 0.89 | 0.91 | passes | 0.95 | 0.937 |
| GSE99963 | 1/10 | 709,450 | 0.76 | 0.91 | fails (0.155 away) | 0.73 | 0.781 |
| GSE99963 | 1/100 | 70,986 | 0.51 | 0.91 | fails | 0.21 | 0.262 |
| GSE99963 | 1/1000 | 7,089 | 0.46 | 0.91 | fails | -0.07 | 0.237 |
| GSE144304 | all | 9,086,096 | 0.87 | 0.86 | passes | 0.96 | 0.199 |
| GSE144304 | 1/10 | 904,545 | 0.87 | 0.86 | passes | 0.87 | 0.089 |
| GSE144304 | 1/100 | 90,352 | 0.71 | 0.86 | passes | 0.64 | 0.304 |
| GSE144304 | 1/1000 | 9,186 | 0.46 | 0.86 | fails | 0.09 | 0.846 |

1. **Rule 1, the printed AUC comes back:** 2 of 3 series with all reads; 1 of 3 with 1/10 or 1/100;
   0 of 3 with 1/1000.
2. **Rule 2, same samples score the same:** with all reads, 0.95 and 0.96 in the two series that
   come back, so there it is the same score sample by sample. 0.04 in GSE120862.
3. **Rule 3, the listed genes matter:** fails in the two series that come back. Random gene sets of
   the same expression do as well as the 338 listed genes in 94 % of 1,000 draws for GSE99963 and
   20 % for GSE144304.

232 of the 338 listed genes have a read in each series with all reads (47 of the 48 in cluster 13,
66 of the 137 in cluster 14, 119 of the 153 in cluster 3; columns `listed_genes_found_cluster_*`); the rest are left out, as the plan
says.

## GSE120862 goes the opposite way, and this is not explained

- It is not the number of reads: the AUC moves further from the printed one as reads are added
  (0.58, 0.62, 0.69, 0.72).
- Samples are paired by GEO title, the same pairing that gives 0.95 and 0.96 in the other two series.
- Within each group the two scores agree in part (rank correlation 0.58 among the 42 untrained
  samples and 0.50 among the 42 trained ones). Across the groups they do not (0.04). So the two
  ways of counting differ by a shift between the untrained and the trained samples, not sample by
  sample.
- From reads, most genes shift up in the trained samples: every gene with weight 1 gives 0.87, and
  the median random gene set 0.85. The 338 listed genes give less than every one of the 1,000 random
  sets (0.72), so the listed genes do pull toward the printed direction, but not past 0.5.
- The gene-table score for this series is from the authors' FPKM table (PR 212); this one is from
  CPM. The plan chose CPM for all three before counting. Whether the shift between the groups is in
  the authors' table, in the reads (for example the two groups sequenced apart), or in the change
  from FPKM to CPM has not been tested.

## Is it the counting method or the number of reads? A check on the authors' tables

Added after the first counts, so not part of the plan. `scripts/fig4_gene_table_check.py` puts the
authors' own gene tables for GSE99963 and GSE144304 through the same scoring code, whole and thinned
to the number of reads this method placed in a gene (`gene_table_check/fig4_reads_comparisons.tsv`).
GSE120862 is not in this check: its gene table is FPKM, which cannot be thinned read by read.

| series | reads per sample | AUC, authors' table | AUC, from reads | printed |
|---|---|---|---|---|
| GSE99963 | whole table (7.6 M); all reads (7.1 M) | 0.91 | 0.89 | 0.91 |
| GSE99963 | as the 1/10 share (709,000) | 0.80 | 0.76 | 0.91 |
| GSE99963 | as the 1/100 share (71,000) | 0.43 | 0.51 | 0.91 |
| GSE99963 | as the 1/1000 share (7,100) | 0.45 | 0.46 | 0.91 |
| GSE144304 | whole table (13.4 M); all reads (9.1 M) | 0.84 | 0.87 | 0.86 |
| GSE144304 | as the 1/10 share (905,000) | 0.86 | 0.87 | 0.86 |
| GSE144304 | as the 1/100 share (90,000) | 0.74 | 0.71 | 0.86 |
| GSE144304 | as the 1/1000 share (9,200) | 0.60 | 0.46 | 0.86 |

1. **The scoring code is right.** On the whole tables it gives 0.908 and 0.838 against the printed
   0.91 and 0.86, and ranks samples as PR 212 does (rank correlation 0.995 and 0.985).
2. **Counting with no genome loses nothing that the number of reads does not.** At every depth the
   reads' AUC is within 0.08 of the authors' table thinned to the same depth, except the smallest
   share of GSE144304 (0.46 against 0.60). The score fades below about 1 M reads per sample
   whichever way the reads are counted.
3. **On the authors' whole tables, the 338 genes are not what separates the groups.** Random gene
   sets of the same expression reach the listed genes' AUC in 93 % of 1,000 draws for GSE99963 and
   33 % for GSE144304 (median random AUC 0.95 and 0.82). Every protein-coding gene with weight 1
   gives 0.96 and 0.85 (column `auc_all_genes_weight_1`, also added after the plan). The panels show
   a shift shared by most genes. PR 213 saw the same on the microarray series.

## Cost

Measured with `/usr/bin/time -v`, three series together, 16 threads each, 0.46 GB peak memory:

| share of reads | reads read | CPU time | wall seconds, summed over the three tasks | CPU seconds per million reads |
|---|---|---|---|---|
| 1/1000 | 5.7 M | 77 s | 16 | 13 |
| 1/100 | 57.4 M | 720 s | 115 | 13 |
| 1/10 | 574 M | 1.7 h | 479 | 11 |
| all | 5,743 M | 30.9 h | 8,802 | 19 |

The plan reports the cost of the smallest share of reads at which rule 1 passes in all three
series. There is none. For the two series that come back, that share is all reads: 7.6 CPU h for
1,800 M reads.

The only measured alignment cost is on other reads: 70.6 CPU h for the 643.5 M Botryllus read pairs
(trim, STAR, RSEM, DESeq2), about 395 CPU seconds per million pairs. With all reads, counting with
no genome costs 19 CPU seconds per million single-end reads, about 1/20 of that, not 1/1000. The two
were measured on different reads. A share of 1/100 of the reads would bring the cost near 1/3000,
and at that share only GSE144304 passes rule 1.

## Not done

- Why GSE120862 goes the opposite way. Next test: the score from reads with FPKM in place of CPM
  (gene lengths from GENCODE), and the authors' FPKM table checked for a shift of all genes between
  the untrained and trained samples.
- A random draw of reads in place of the first reads of each file.
- Limits in the plan's "Not covered" still hold: the five microarray panels, panel E, and whether the
  groups are the ones the panel's words describe.

## Files

- `fig4_reads_comparisons.tsv`: one row per series and share of reads, the three rules, cost.
- `fig4_reads_samples.tsv`: each sample's score from reads and from the gene table.
- `fig4_reads_random.tsv.gz`: AUC of every random gene set.
- `counts/`: the count tables the scores come from, copied from
  `/scratch/users/olgabot/nb361-pec/human/fig4_score/` on Sherlock.
- `gene_table_check/`: the same three tables for the authors' gene tables, whole and thinned.
