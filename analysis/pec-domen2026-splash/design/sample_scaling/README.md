# Plan: one way of scaling a sample, used everywhere reads are counted (notebook 361)

Written 2026-11-10, after the GSE120862 test and before anything below was rescored. Results go in
`exports/sample_scaling/README.md`; changes to this plan get a dated note at the end.

## Why

Every count table in this folder was scaled by dividing a sample's counts by its total (counts per
million, CPM). In GSE120862 that choice set the direction of the Fig. 4C-D score
(`exports/fig4_score_from_reads/README.md`): the authors' own table gives AUC 0.08 with their scaling
and 0.64 with every sample scaled to the same total. A total is moved by a few genes with very many
reads. DESeq2 scales by the typical gene instead, and both the paper and the nf-core runs here
(`nextflow-runs/pec-domen2026-nfcore`, `nextflow-runs/botryllus-pec-young`) use DESeq2.

## The scaling rule, fixed now

DESeq2's `poscounts` size factors, the setting the nf-core runs here use because it still works when
no gene has a read in every sample:

1. For each gene with a read in any sample, take the mean of log(count) over the samples where the
   count is above zero, divided by the number of all samples.
2. A sample's factor is the median, over its genes with a read, of log(count) minus that gene's
   value from step 1, taken back from the log.
3. Factors are divided by their geometric mean, so they multiply to 1.

Scaled count = count / factor. Where a script took log2(CPM + 1), it now takes
log2(scaled count x c + 1), with one constant c for the whole table, chosen so that the mean sample
sums to one million. So only the scaling between samples changes, and the + 1 means what it meant
before. The function goes in `scripts/pec_analysis_utils.py` and every script below calls that one
copy. It is checked against PyDESeq2's `poscounts` factors on one table before use.

## Where it is applied, and how each is judged

What was already seen before this plan: GSE120862 from all reads, scaled with DESeq2's default
factors (genes with a read in every sample), gives AUC 0.27. Nothing else below has been computed.

| where | script | CPM result to compare with | judged by |
|---|---|---|---|
| A. Fig. 4C-D score from reads, three human series, four shares of reads | `fig4_score_from_reads.py` | 2 of 3 series pass rule 1 with all reads | the same three rules and cutoffs as `design/fig4_score_from_reads/README.md`; "reproduced from reads" still needs rule 1 in all three series |
| B. The same score from the authors' count tables (GSE99963, GSE144304), whole and thinned | `fig4_gene_table_check.py` then `fig4_score_from_reads.py` | AUC 0.92 and 0.86 on the whole tables | known answer: on the whole tables the AUC must stay within 0.15 of the printed 0.91 and 0.86. If it does not, the new scaling is wrong for that series and A is not read for it |
| C. Botryllus, up or down per gene from reads counted with no genome | `gene_directions.py`, new rule `size_factor_mean`: the `sample_mean` rule with the scaled count in place of CPM, same 10-read and 2-fold cutoffs | `sample_mean`: 609 of 1,135 nf-core list genes and 522 of 878 paper list genes going the list's way; 11 of 12 lists beat random gene sets in each | the same scoring (`nfcore_recovery.py`, `cheap_route_score.py`). Scaling counts as the cause of the gap only if both shares go up and at least 11 of 12 lists still beat random in each. The simulated control is scored too (CPM: 292 of 300 planted genes right, 0 wrong, 25 of 2,600 unchanged genes given a direction) |
| D. Human-to-Botryllus k-mer test | `de2de_test.py` | not changed here | it sums log2 CPM over chosen k-mers of one sample; listed so that it is not forgotten, and left for its own plan because its counts are k-mers per sample, not a gene table |

Not touched: PR 212 (the score from the authors' tables, on its own branch; B covers its two count
tables), PR 213 (microarrays have no read totals), and routes that already use DESeq2 or the tool's
own scaling (anchor counts, KaMRaT, DE-kupl).

## What would count as "the scaling was the problem"

- Human: GSE120862 moves to the printed side of 0.5 and inside 0.15 of 0.08 with all reads, while
  GSE99963 and GSE144304 stay inside. Seen already: 0.27 with the default factors, which is the
  printed side but outside the band, so a pass here would have to come from the difference between
  the default and `poscounts` factors.
- Botryllus: the rule in row C.

Either can fail while the other passes. Both CPM and scaled results are kept side by side; the CPM
files are not overwritten.

## Not covered

- PR 207's own gap between its nf-core refit and the paper's gene lists. Both sides there already use
  DESeq2 size factors; its README lists the set of genes the factors rest on as one untested cause.
  A test that changes only the size factors in that fit belongs with that pipeline.
- Whether `poscounts` or the default factors are the better choice. `poscounts` is fixed here because
  it is defined at every share of reads.
