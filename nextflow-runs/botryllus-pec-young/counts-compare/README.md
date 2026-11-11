# Why do this run's counts differ from the paper's? Plan

Written 2026-11-11, before any of the tables below were compared. The result is added under
"Result" at the end; a change to the plan gets a dated note.

## Question

PR 220 showed that size factors do not explain why this run finds 183 of the paper's 782 listed
genes where the paper's own counts find 626: the counts themselves differ. Three things differ
between the two count tables. Which one matters?

1. **How reads are counted.** The paper's table (Dataset S1, sheet `Pacemaker_Counts_Table`) comes
   from its own pipeline; this run's comes from nf-core/rnaseq (STAR, then Salmon).
2. **The gene models.** The paper counts on 72,115 `g` genes; this run on 35,541 `FUN_` genes of the
   April 2024 genome.
3. **The match between the two gene sets.** `compare_paper_genes.py` pairs each `g` gene with a
   `FUN_` gene through its protein (MMseqs2, 90 % identity over 50 % of the paper's protein).

## The yardstick: one counter run on both gene sets

`pec-tools gene-counts` (PR 201, notebook 361) counts reads per gene with no genome and no
alignment: a read goes to the gene whose protein shares the most 10-amino-acid windows with the
read's translations. It was run on the same trimmed reads against both protein sets, the paper's
(`g` genes) and this run's (`FUN_` genes), for 47 libraries. So it is one counting method on two
gene sets, and it shares no code with either pipeline.

| table | gene set | how counted | file on Sherlock |
|---|---|---|---|
| paper | `g` | the paper's pipeline | Dataset S1, exported by `each-alone-checks/export_paper_counts.py` |
| salmon | `FUN_` | nf-core/rnaseq 3.27.0, STAR + Salmon (PR 207) | `$SCRATCH/botryllus-pec-young/rnaseq/star_salmon/salmon.merged.gene_counts.tsv` |
| rsem | `FUN_` | nf-core/rnaseq, STAR + RSEM (PR 201's run) | `$SCRATCH/nb361-pec/nfcore/rnaseq/star_rsem/rsem.merged.gene_counts.tsv` |
| no-genome, paper's genes | `g` | `pec-tools gene-counts` | `$SCRATCH/nb361-pec/cheap_route/all_reads/all.counts.tsv` |
| no-genome, this run's genes | `FUN_` | `pec-tools gene-counts` | `$SCRATCH/nb361-pec/cheap_route/new_proteome/all.counts.tsv` |

Libraries: those of the paper's 38 that are in every table.

## What is measured

For a pair of tables that name genes the same way:

- **Counts.** Per library, the Spearman rank correlation of the two counts over genes with a read in
  either table. Reported as the median over libraries.
- **Change with treatment.** Per comparison of the paper (2 h, 24 h, 13-49 d, 132 d), each library is
  scaled to one million reads, and a gene's change is log2((mean of treated + 1) / (mean of
  control + 1)). Over genes with 10 or more reads summed over the comparison's libraries in both
  tables, the Spearman rank correlation of the two changes. Reported per comparison and as the
  median of the four. Library totals are used on both sides so that scaling is the same; PR 220 and
  PR 201 showed the choice of scaling matters, so it is held fixed here, not tested.

Pairs:

| pair | what differs | what is the same |
|---|---|---|
| A. paper against no-genome on the paper's genes | how reads are counted | gene set |
| B. salmon against no-genome on this run's genes | how reads are counted | gene set |
| B2. salmon against rsem | two standard counters | gene set; shows how well two accepted counters agree |
| C. no-genome on the paper's genes against no-genome on this run's genes, through the gene match | gene set and match | how reads are counted |
| D. paper against salmon, through the gene match | everything | nothing; this is the gap itself |

For C and D, the counts of all `g` genes matched to one `FUN_` gene are added up. They are also
split by how the genes pair: one `g` gene to one `FUN_` gene, or several `g` genes to one `FUN_`
gene.

## Rule for reading the result, fixed now

"Agree" means a median count correlation of 0.8 or more and a median change correlation of 0.6 or
more. The cutoffs are a judgement made before looking; B2 is reported beside them so that they can
be compared with what two accepted counters reach.

| A | B | reading |
|---|---|---|
| agree | agree | Counting is not the cause. The gap comes from the gene sets or the match; go to C. |
| disagree | agree | The paper's table differs from a plain count on its own genes: the paper's counting is the cause, or part of it. |
| agree | disagree | Salmon's table differs from a plain count on its own genes: this run's counting is the cause, or part of it. |
| disagree | disagree | The no-genome counter is not a usable yardstick here, and the test says nothing. |

If the first row holds, C is read: if the change correlation is 0.6 or more for one-to-one genes and
under 0.6 for several-to-one genes, the cause is genes that the paper splits and this run joins. If
it is under 0.6 for both, the two gene sets differ beyond splitting, or the match pairs the wrong
genes; these two are not separated by this test. If it is 0.6 or more for both, the gene sets carry
the same changes and the test has not found the cause.

## Known limits, before running

- The no-genome counter places about 29 % of read pairs in a gene and cannot count a gene with no
  protein, so it is a coarser count than either pipeline.
- A rank correlation of counts is mostly set by how highly a gene is expressed, which is why the
  change with treatment is measured too.
- The change is a ratio of means with no test, so it is not the same thing as "listed genes found".
