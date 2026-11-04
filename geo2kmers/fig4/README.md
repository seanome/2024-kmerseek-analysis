# The Fig. 4C-D test

`fig4.nf` lands each array series on the signature proteins and scores its samples, with no gene annotation,
probe-to-gene table or genome.

```bash
nextflow run fig4.nf -profile small --probes <folder> --rscript <path> --outdir results/fig4
```

Build the R environment from `environment-r.yml`. `--rscript` is its `Rscript`. Run this on a cluster.

## Inputs

- `signature_proteins.tsv` and `.fasta` (`results/fig4/`), from `signature.py`. The genes come from Supplementary
  Table 2 of Ghosh et al. 2023, sheet C13-14-3: 338 list entries, 280 distinct genes, since a gene can sit in
  clusters 14 and 3. 313 entries resolved to a reviewed human UniProt protein, 262 distinct proteins. M1 is cluster
  13. M2 is clusters 14 and 3. Symbols that did not resolve, mostly old LOC names, are in the table.
- `groups.tsv`: which sample is in which group, from PR 212, the groups in the authors' notebook. PR 212 found
  these differ from what some panels say. `published_fig4.tsv`: the printed AUCs, also from PR 212.
- Affymetrix probes: GEO has no sequences for these arrays, so `export_probe_tables.R` takes them from
  Bioconductor packages. GPL13912 (Agilent) has a SEQUENCE column.

## Files in `results/fig4/`

| file | what it holds |
|---|---|
| `platforms.tsv` | each series' platform, type, median probe length, rule, k and landing share |
| `landing/` | each series' feature-to-protein table and landing share |
| `auc.tsv` | group sizes, published AUC, this test's AUC, Mann-Whitney p, verdicts |
| `sample_scores.tsv` | each sample's score, as drawn in `fig4_panels.png` |
| `annotation_control.tsv` | how many landings agree with the platform's gene names |
| `earlier_runs/` | the two earlier tables, each with the HTA fault below |

## Score

For each signature protein, a sample's value is the mean `intensity_weight` of the features landing on it. A
feature is what the series reports a value for. Each protein is standardised across the series with StepMiner's
threshold, (value - (threshold + 0.5)) / (3 x sd) (Sahoo et al. 2007). The score is -1 x the M1 mean + 1 x the
M2 mean, and -1 x the M1 mean for GSE27536. `score_method` in `auc.tsv` is `stepminer_modified_z`.

`auc_ghosh_weights` is a second score, not in the plan: -1, 1 and 2 on the sums of clusters 13, 14 and 3, the
weights in Ghosh et al.'s table. Other differences from the published method remain. It used log expression
per probe, where this uses `intensity_weight` per protein. GSE104079 is mouse and landed on human proteins.

## Faults found while running

- GPL17586's probes were first joined at the level of exons, so the score used no M1 protein. They now join
  through `core_mps` to the transcript clusters the series reports. `earlier_runs/first_run_auc.tsv` has that run.
- GPL17586's probes are held as the complement of the strand, so true matches were missed. This was found because
  the annotation control gave 0 agreement, and tracing one gene showed it: that use of the annotation found the
  fault, but it changes no landing. `land.py` now keeps whichever reading has more probes with a k-mer, with no
  annotation, and records both counts (32888 and 90385 in `platforms.tsv`). `earlier_runs/second_run_auc.tsv` has
  the run before it. Other platforms keep the reading as given.
