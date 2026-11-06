# The Fig. 4C-D test

`fig4.nf` lands each array series on the signature proteins and scores its samples, with no gene annotation,
probe-to-gene table or genome.

```bash
nextflow run fig4.nf -profile small --probes <folder> --rscript <path> --outdir results/fig4
```

Build the R environment from `environment-r.yml`. `--rscript` is its `Rscript`. Run this on a cluster.

## Inputs

- `signature_proteins.tsv` and `.fasta` (`results/fig4/`), from `signature.py`. The signature is the gene list.
  It comes from Supplementary Table 2 of Ghosh et al. 2023, sheet C13-14-3: 338 list entries, 280 distinct genes,
  since a gene can sit in clusters 14 and 3. 313 entries resolved to a reviewed human UniProt protein, 262
  distinct proteins. Clusters are Ghosh's gene groups. M1, proinflammatory, is cluster 13. M2, cell repair, is
  clusters 14 and 3. An entry whose main gene name is the symbol comes first. 39 entries matched only as a
  synonym, marked in `name_match`, and ZNF228 is one of them, with two possible entries.
- `groups.tsv`: which sample is in which group, from PR 212, the groups in the authors' notebook. PR 212 found
  these differ from what some panels say. `published_fig4.tsv`: the printed AUCs, also from PR 212.
- `pr212_gse111555_scores.tsv`: PR 212's sample scores for GSE111555, at commit f00ee836. `regroup_gse111555.py`
  uses them to recompute the published AUC on the corrected groups. It writes the `--groups` and `--published` files
  for `fig4.nf`. `plot_gse111555_labels.py` draws the figure, outside the pipeline.
- Affymetrix probes: GEO has no sequences for these arrays, so `export_probe_tables.R` takes them from
  Bioconductor packages. GPL13912 (Agilent) has a SEQUENCE column.

## Files in `results/fig4/`

| file | what it holds |
|---|---|
| `platforms.tsv` | each series' platform, type, median probe length, rule, k and `share_of_probes_with_a_kmer_in_proteome` |
| `pairing.tsv` | who gave the samples in each comparison, made by `fig4/pairing.py` outside the pipeline |
| `landing/` | each series' feature-to-protein table and `share_of_probes_with_a_kmer_in_proteome` |
| `auc.tsv` | group sizes, published AUC, this test's AUC, Mann-Whitney p, verdicts |
| `sample_scores.tsv` | each sample's score, as drawn in `fig4_panels.png` |
| `annotation_control.tsv` | how many landings agree with the platform's gene names |
| `chance_test.tsv`, `chance_by_comparison.tsv` | how often chance passes the rule |
| `corrected_labels/` | `fig4.nf` rerun with six GSE111555 samples moved to the post group: `auc.tsv`, the chance tables, `sample_scores.tsv`, the `groups.tsv` and `published_fig4.tsv` it ran on, `gse111555_labels.tsv` with each sample's evidence, and `gse111555_labels.png` |
| `earlier_runs/` | the tables from three earlier runs, before the faults below were fixed |

## Score

For each signature protein, a sample's value is the mean `intensity_weight` of the features landing on it. A
feature is what the series reports a value for. Each protein is standardised across the series with StepMiner's
threshold, which splits the values at the best single step (Sahoo et al. 2007), as (value - (threshold + 0.5)) /
(3 x sd), the formula in the brief and in PR 212's code. The score is -1 x the M1 mean + 1 x the M2 mean, and -1 x
the M1 mean for GSE27536. `score_method` in `auc.tsv` is `stepminer_modified_z`. Mann-Whitney p tests whether post
samples score differently from pre samples. It treats the samples as unrelated, although most comparisons follow
the same people: `pairing.tsv` gives the people in each group, and the AUC compares groups, not each person with
themselves. All five series are already on a log or z scale, so
`intensity_weight` here is that value minus the array's 5th percentile, floored at 0. GSE104079 holds z-scores with
negative values, which `sketch` would refuse; `fig4/score.py` reads the series matrix directly.

`auc_ghosh_weights` is a second score, not in the plan: -1, 1 and 2 on the sums of clusters 13, 14 and 3. The
weights are in Ghosh et al.'s table. Sums are what PR 212 found the published code does; Ghosh's text says
averages. The published method also scores each probe, where this scores each protein.

## Faults found while running

GPL17586 is the Affymetrix Human Transcriptome Array, GSE111555's platform.

- Its probes were first joined to exon-sized groups, so the score used no M1 protein. They now join through the
  package's `core_mps` table to the larger groups the series reports. `earlier_runs/first_run_auc.tsv` has that run.
- Its probes are held as the complement of the strand, so true matches were missed. This was found because the
  annotation control gave 0 agreement, and tracing one gene through the annotation showed why. **So the annotation
  led to a change, against the plan.** The change itself uses no annotation: `land.py` keeps whichever reading has
  more probes with a k-mer, and records both counts (`platforms.tsv`). It took GPL17586 from 1922 landings to
  more than 3000, and turned GSE111555's planned verdict from not reproduced (0.727,
  `earlier_runs/second_run_auc.tsv`) to reproduced. Other platforms keep the reading as given.
- A fresh check found that a synonym search gave APPL2 and MR1 the wrong proteins, that one probe counted twice
  when it served two probe sets, and that short oligos used any stop-free frame. `earlier_runs/third_run_auc.tsv`
  has the run before these fixes. The short oligo fix took GPL13912 from 286 landings to 81 and moved GSE104079's
  second score from 0.356 to 0.322, so that score's count of 6 of 7 rests on a fix made after seeing 5 of 7.
