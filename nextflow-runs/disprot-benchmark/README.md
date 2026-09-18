# DisProt disorder benchmark

Does kmerseek find homologs through intrinsically disordered regions, where a
structure-based tool has nothing to encode?

- Queries: 271 human DisProt proteins that share at least one Pfam domain with a protein
  in at least three of the nine QfO target species.
- Targets: the nine QfO proteomes, mouse (100 Mya) to E. coli (2000 Mya).
- Truth: the Pfam QfO pair truth (`results/pfam_benchmark/pairs`), filtered to the
  DisProt queries. A human-target pair is positive when the two proteins share a Pfam
  domain.
- Strata: each query's mean metapredict disorder, binned at 0.2 and 0.5.
- Arms: kmerseek over every alphabet and ksize (below), MMseqs2 at `-s 7`. Foldseek is
  off by default; it needs the AlphaFold download and a laptop-only binary.

## The kmerseek sweep

The alphabet x ksize matrix is the one table in `../shared/kmerseek_encodings.nf`, shared
with `qfo-pfam-region-benchmark` and `invertebrate-dark-set`: 17 alphabets, each over the
ksize range where its k-mers carry about 18 bits or more, 183 combos. `polarity4` and
`funcgroups8` add 20 more with `--kmerseek_extra_encodings true`. The low-complexity mask
is off; `--low_complexity_toggle false,true` runs both settings and doubles the sweep.

| Flag | What it does |
|---|---|
| (none) | every alphabet over its ksize range |
| `--kmerseek_encodings hp_lehninger2,gbmr4` | those alphabets, each over its full range |
| `--kmerseek_combos hp_thomas_dill2:26` | named alphabet:ksize pairs, replacing the matrix |
| `--target_species mouse,ecoli` | a subset of the nine targets |

A pair's kmerseek score is its best `max_containment` over the regions kmerseek
reported, after the region benchmark's Bonferroni filter on the region's Poisson tail
(`--kmerseek_max_bonferroni_p 0.05`, 0 disables). `--kmerseek_rank_by region_enrichment`
ranks by the region benchmark's column instead. Tool names carry the combo:
`kmerseek_<alphabet>_k<ksize>_lc<false|true>`; `bin/disprot_tool_names.py` parses them.

Indexes are stored under `<db_cache>/kmerseek_index/` with the region benchmark's exact
names, `<species>.<alphabet>.k<ksize>.lc<mask>.kmerseek.rocksdb`. On Sherlock `--db_cache`
points at that pipeline's results directory, so every index is a storeDir hit and the
only kmerseek work is the 271-query searches.

## Running

Checks that need nothing:

```bash
make run-stub
```

A real two-combo, two-target run on the laptop (Docker; needs `make build-python-image`
once):

```bash
make run-mini
```

### The sweep on Sherlock

Once, from the laptop. The Python image carries scikit-learn and matplotlib, which the
kmerseek image does not; rebuild and push it whenever `Dockerfile.python` changes, and
bump `PYTHON_IMAGE_TAG` here and `params.python_image` in `main.nf` together.

```bash
make push-python-image
```

```bash
make sherlock-clone
```

```bash
make sync-data
```

`sync-data` ships the two inputs that are not in git: the Pfam pair truth and the frozen
`disprot_human.tsv` from the laptop run. Sherlock compute nodes have no outbound
internet, so the pipeline cannot fetch DisProt there, and shipping the parse also pins
the DisProt release.

Then on Sherlock, in your own tmux, from the clone's `nextflow-runs/disprot-benchmark`:

```bash
ml load biology nextflow
```

```bash
make prefetch-images
```

```bash
make run-sweep-sbatch
```

The head runs as a batch job on a compute node (`scripts/nf-head`); `make watch` attaches
to its live table read-only (Ctrl-b d detaches), `make stop` sends it Ctrl-C so it
cancels its own jobs. Relaunching with the same target resumes. Scale: 183 combos x 9
targets = 1_647 searches at 271 queries each, plus a format and an evaluate task per
search, all small. Index builds only happen for cells the region benchmark's cache does
not hold.

Back on the laptop:

```bash
make pull-report
```

## Outputs, under `--outdir`

| Path | What |
|---|---|
| `kmerseek/human_vs_<sp>.<alphabet>.k<k>.lc<mask>.regions.csv.zst` | kmerseek's region table (storeDir) |
| `kmerseek_pairs/human_vs_<sp>.<tool>.tsv.gz` | pair, score, corrected p |
| `metrics/<tool>.<sp>.disprot_metrics.parquet` | per stratum: AUC-PR, AUC-ROC, recall at 1% and 5% FDR |
| `all_disprot_metrics.parquet` | every tool x species x stratum, with alphabet/ksize/lowcomp columns |
| `kmerseek_sweep_summary.parquet` | each combo averaged over species, per stratum |
| `figures/sweep_recall_heatmap.pdf` | alphabet x ksize, colour = recall at 5% FDR, one panel per stratum |
| `figures/sweep_best_ksize.pdf` | each alphabet at its best k, against MMseqs2 |
| `multiqc_report.html` | the sweep heatmaps first, then the headline tools |

"Headline tools" in the per-tool figures and tables are the baselines plus the three
kmerseek combos with the best mean recall at 5% FDR on the disordered queries
(`--headline-n`). The sweep figures show every combo.
