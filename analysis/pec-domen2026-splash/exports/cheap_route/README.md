# Plan: half the paper's genes, going the paper's way, at 1/1000th of SPLASH's cost (notebook 361)

Asked 2026-11-04. Nothing below has run except the read-fraction table. Both the target and the
rules are written before any test.

## Why the control and human data look better than Botryllus

![Genes found in three data sets](why_control_differs.png)

**a.** Share of genes found by three routes. Control: 400 planted genes. Human: the 12 study lists of
GSE99963 and GSE144304. Botryllus: the paper's 12 lists. Control counts a gene with no direction;
KaMRaT in human and Botryllus must go the list's way; SPLASH has no direction anywhere. **b.** Reads
per sample. **c.** Reads per sample for a list gene: box, middle 50 %; tick, median. The control is
worked out as about 1 M read pairs over 3,000 genes.

The routes work: 94 % in the control and 55 % in human for the KaMRaT t-test at p < 0.05. Botryllus
gets 6 % from the same route. Its list genes have about 11 times fewer reads than human list genes
(median 14.5 against 162.8), and its samples span 1,476-fold. Other differences may add to this:
Botryllus has far fewer samples per group, and the human route was matched to a complete proteome.
The figure does not separate these. The 50 % target is already met in human by one route, at more
than SPLASH's cost; the target below is for Botryllus.

## Target

Gene counted: on one of the paper's 12 lists, and reached by a sequence going the paper's way.

| measure | needed | best today |
|---|---|---|
| paper's genes, right direction | 439 of 878 | 97 of 792 (DE-kupl DESeq2, p < 0.05); no route has more than 12 % |
| CPU, all six contrasts | 44 s | 12.1 h (SPLASH route: SPLASH, compactors, protein search) |
| wall time | 11 s | 3.0 h |
| peak memory | 84 MB | 83.6 GB |

Rules for a pass, all three needed:

1. 439 or more of the 878 genes.
2. 9 or more of the 12 lists beat 1,000 random gene sets matched on baseMean, the test in
   `scripts/paper_recovery.py`. This is needed because a method that gives every gene a direction
   would match half the list by chance.
3. Cost measured with `/usr/bin/time -v`, not estimated.

## Fewer reads alone cannot do it

The list genes are rare: their median baseMean is 14.5, and 42 % are below 10. Best case, from the
paper's own numbers (`thin_read_bound.tsv`, `scripts/thin_read_bound.py`):

| share of reads | list genes seen at all, of 878 | genes with about 10 reads per group |
|---|---|---|
| all | 878 | 824 |
| 1/10 | 859 | 391 |
| 1/100 | 609 | 157 |
| 1/1000 | 296 | 15 |

At 1/1000th of the reads, too few genes are seen at all, so the saving has to come from a cheaper
method per read. Even 1/10th of the reads falls short of 439 genes with about 10 reads per group.

Reading the reads is itself a limit. The trimmed reads are 88 GB of compressed files and 898 M
reads, so 44 CPU s would mean 20 M reads per CPU second, before any matching. I expect that to be out
of reach for any method that opens the reads. Decompression speed here has not been measured.

## Sequencing depth

Samples run from 108,860 to 160,736,718 reads, median 4.4 M. Within young 24 h they run from
301,282 to 93,965,840. So a swap that flips a pair can flip two samples of very different depth,
and a sample with few reads adds almost nothing to either group. Swapped labels are therefore
reported here but are not a pass rule. They show what noise looks like at this depth. They cannot
tell "PEC does nothing" from "too few reads to see it".

## Test, cheapest first

The method: one small `pec-tools` command, not yet written. It translates each read in six frames,
looks up protein k-mers in a small index of the old gene-model proteins, and counts reads per gene
per sample. Counts are summed per group, so shallow samples weigh little. A gene gets a direction
when it has 10 or more reads in each group or 10 or more in one and none in the other, and a 2-fold
change after scaling for each group's total reads. There is no per-sample test, because that is the
costly part and shallow samples have little to give it.

| step | what | reads | stop if |
|---|---|---|---|
| 1 | Write the command, test it on the simulated control | control only | planted direction is wrong for most found genes |
| 2 | All 47 samples, all reads: the most this method can do | all | under 439 genes or under 9 lists: stop, no thinner run can pass |
| 3 | Same with 1/10, then 1/100 of the reads | first reads of each file | stop at the first fraction that fails rules 1 or 2 |
| 4 | Report the cheapest passing run against the target | | |

Step 2 is the deciding run: it costs one Sherlock job, and if it fails, the target is out of reach
for this method. Every run writes genes, directions, and `/usr/bin/time -v` output. CPU and memory are
expected to be well above budget in step 2; that run measures the most genes, not the cost.

If steps 2 and 3 pass, cost is likely to land near 1/100 to 1/300 of SPLASH's, not 1/1000. That is an
estimate from the reading limit above, not a measurement.

## Not covered

- The index is built from the old gene-model proteins, so this does not find genes outside them.
- The index-building cost is left out of the budget, as a one-time cost, like the databases in the
  baseline.
- The paper's lists come from these reads, so agreement is not an independent repeat.
