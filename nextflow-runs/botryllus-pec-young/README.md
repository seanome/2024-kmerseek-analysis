# Young Botryllus colonies after PEC: treated vs control, per time window

Domen et al. 2026 (PNAS) gave Botryllus colonies pulsatile electrical current (PEC) and
sequenced treated and control colonies from 2 hours to 132 days afterwards (SRA project
PRJNA1219309, 64 libraries). This run compares treated with control colonies at each time,
in young colonies only (370-510 days old), two ways:

- **With the genome**: nf-core/rnaseq 3.27.0 aligns reads to the April 2024 Botryllus
  assembly (STAR) and counts reads per gene (Salmon). nf-core/differentialabundance 2.0.0
  then tests each gene with DESeq2.
- **Without the genome**: SPLASH 2.11.9 finds 31-base sequences (anchors) whose following
  sequence differs between treated and control libraries, and extends each into a longer
  sequence (compactors).

The two are kept apart: separate output folders, never one table.

```mermaid
flowchart LR
  A["52 young-colony libraries<br/>Trim Galore reads, from PR 203"] --> B["nf-core/rnaseq 3.27.0<br/>STAR + Salmon, April 2024 genome"]
  B --> C["gene counts"]
  C --> D["nf-core/differentialabundance 2.0.0<br/>DESeq2, ~ colony + group"]
  D --> E["6 tables: treated vs control<br/>per window, one row per gene"]
  A --> F["splash_prep<br/>read 1 + read 2 in one file"]
  F --> G["SPLASH 2.11.9, one run per window<br/>control -1, treated +1"]
  G --> H["6 tables: anchors and<br/>extended sequences per window"]
```

## Libraries and windows

Every young library is used, whatever its depth (Olga, 2026-10-27: "Don't filter on number
of reads. Keep all samples"). The 52 libraries were collected at 16 different times after
treatment, and 9 of those times have only control or only treated libraries, so times are
grouped into 6 windows that each have both:

![Read pairs per library, by window and treatment](figures/design_libraries_per_window.png)

| Window | Hours after treatment | Control | Treated |
|---|---|---:|---:|
| 2 h | 2 | 6 | 6 |
| 24 h | 24 | 6 | 6 |
| 13-17 d | 312-408 | 6 | 6 |
| 21-28 d | 504-672 | 3 | 2 |
| 49-52 d | 1176-1248 | 2 | 3 |
| 104-132 d | 2496-3168 | 3 | 3 |

Each point in the figure is one library. 40 of the 52 have under 5 million read pairs. In
the 13-17 d window 9 of 12 libraries have fewer than 3,500 pairs, and all 5 libraries in
21-28 d have fewer than 20,000, so those two comparisons can detect only large changes in
highly expressed genes.

Hours after treatment per vial come from the paper's Dataset S1 sample table
(`inputs/published_vial_hours.tsv`); treatment, colony and age from the SRA BioSample
records (`inputs/ena_sample_metadata.tsv`); read files and pair counts from the PR 203
trimming check (`inputs/trimmed-inputs.validated.json`). `make_design.py` turns these into
`samplesheet.csv`, `samples.csv`, `contrasts.csv` and `splash/<window>/`;
`plot_design.py` draws the figure.

## Settings that differ from the pipelines' defaults, and why

| Setting | Value | Why |
|---|---|---|
| rnaseq `skip_trimming` | true | reads were trimmed by Trim Galore in PR 203 |
| rnaseq `strandedness` | unstranded, every library | measured on the 3 deepest libraries (below) and assumed for the other 49; left on `auto`, Salmon cannot work out the strand from SRR32233927's 636 pairs, which stopped the PR 203 run |
| rnaseq `min_mapped_reads` | 0 | keep every library (default drops libraries under 5% aligned) |
| rnaseq `skip_deseq2_qc` | true | its DESeq2 step fails when no gene has a count in every library |
| rnaseq `skip_biotype_qc` | true | the FullNames GTF has no `gene_biotype` |
| differentialabundance `deseq2_sf_type` | poscounts | the default size factors need genes counted in every library |
| differentialabundance design | `~ colony + group`, all 52 libraries in one model | group = window and treatment; colony (4 genotypes) as blocking factor. `make run-alone` fits each comparison alone instead, see below |
| SPLASH `--Cjs_samplesheet` | control -1, treated +1 | tests treated vs control rather than any difference between libraries |
| SPLASH `--pvals_correction_col_name` | `Cjs_0_pval` | corrects the p-value for those weights |

All other settings are the pipelines' defaults.

### Strand of read 1: unstranded

`strand-check/strand_check.sbatch` (Sherlock job 49810662, 2026-10-27) ran Salmon's
library-type detection on the first 1 million read pairs of the three deepest young
libraries, against an index of the FullNames transcripts. In all three, read 1 lies on the
gene's strand for half of the fragments (`strand-check/lib_format_counts.tsv`):

| Library | Window | Fragments placed | Read 1 on the gene's strand | Salmon's call |
|---|---|---:|---:|---|
| SRR32233964 | 2 h | 460,804 | 49.9% | IU (unstranded, paired) |
| SRR32233978 | 24 h | 472,715 | 49.9% | IU |
| SRR32233986 | 24 h | 516,727 | 50.0% | IU |

A stranded library would be near 0% or 100%. The germline libraries from the same lab
were unstranded too (`nextflow-runs/botryllus-rnaseq-preprocess`).

## Run

On Sherlock, one command clones this branch into its own folder (so the shared clone on
`main` is untouched) and submits everything; nothing waits on a person:

```bash
D=/scratch/users/olgabot/2024-kmerseek-analysis-botryllus-pec-young; [ -d $D ] || git clone -b olgabot/botryllus-pec-young-windows git@github.com:seanome/2024-kmerseek-analysis.git $D; cd $D && git pull && cd nextflow-runs/botryllus-pec-young && make run
```

| Job | Waits for | Resources | Expected time |
|---|---|---|---|
| `splash_prep`, 52-task array | nothing | 4 CPUs, 4 GB, 4 h each | under 1 h; largest file 13 GB gzipped |
| SPLASH, one per window | all of `splash_prep` | 16 CPUs, 64 GB, 12 h | under 1 h each; notebook 361's runs on these reads took under 2 min per 1 GB |
| Nextflow head: rnaseq, then differentialabundance | nothing | 2 CPUs, 8 GB, 2 days | about 6-10 h; STAR index about 1 h, then alignment of the 70 million-pair library about 1-2 h |

Task sizes in nf-core/rnaseq are the pipeline's defaults (STAR: 12 CPUs, 72 GB, 16 h); no
earlier STAR run on this genome exists to size from. A task killed by the scheduler is
retried twice (`sherlock.config`).

Watch the Nextflow head with `make watch`; stop it with `make stop`. Job ids are in
`$SCRATCH/botryllus-pec-young/job_ids.txt`. `make run` refuses to submit while any job of
this run (`pec-young-*` or `nf-botryllus-pec-young`) is queued or running.

### Launch record

`make run` was given twice on 2026-10-27, at 13:41 (jobs 49841823-49841922) and 13:49
(49843510-49843518). The second round was cancelled. Before the cancel, its
`splash_prep` task 5 wrote 785 MB to `SRR32233964.fq.gz.part`; the first round had
already finished `SRR32233964.fq.gz` at 13:47:52, so that file is the first round's and
the `.part` file is a leftover. Its Nextflow head stopped at once ("Unable to acquire lock
on session"), so the first run was not touched.

### Result of the 2026-10-27 run

- SPLASH: all 6 windows finished (jobs 49841893-49841918, 16 s to 29 min each).
- nf-core/rnaseq: finished at 17:23, 1,466 tasks, none failed. Its summary says 7 of 52
  libraries failed the strand check (not yet looked at).
- nf-core/differentialabundance: DESeq2 finished for all 6 windows and wrote its tables.
  The last step, the HTML report, failed while drawing a clustering heatmap
  (`hclust`: "NA/NaN/Inf in foreign function call"), so the Nextflow head ended as FAILED
  and `differentialabundance/report/` is empty.
- Report fix (2026-11-01), in `params-differentialabundance.yaml`: two report settings
  changed, each explained there. The heatmap failed because SRR32233947 (236 read pairs)
  has 0 in all 500 genes it uses on raw counts, so the heatmap now uses all genes
  (`exploratory_n_features: -1`), where that library has some counts.
  The per-gene boxplots then failed because their colour list had fewer colours than
  the 12 treatment groups; `exploratory_main_variable: contrasts` fixes that. A copy of
  the failed report step, rerun with both settings on Sherlock, finished in 3.5 minutes
  and wrote all sections for raw, normalised and variance-stabilised counts, including
  the details for all 6 windows.
- The report built on 2026-11-01 (head job 50740329, 19 minutes, 4 steps rerun, 28 reused):
  `differentialabundance/report/rnaseq/botryllus_pec_young_differentialabundance_report.html`.
- A first try left raw counts out of the report instead. The `make run` of 2026-11-01
  19:07 (head job 50725494) failed on it after 7 minutes: the exploratory plots and the
  Shiny app are always given all three count tables and stop when fewer names are set.
  That run did show that resuming works: the 52 read-joining tasks and 6 SPLASH jobs saw
  their done files and exited, rnaseq reused every task, and differentialabundance reused
  28 finished steps, including all 6 DESeq2 fits.

Counts per window are in `significant_per_window.tsv`:

| Window | Genes with adjusted p < 0.05 | Higher in treated | Lower in treated | SPLASH anchors with corrected p < 0.05 |
|---|---:|---:|---:|---:|
| 2 h | 140 | 2 | 138 | 86,283 |
| 24 h | 76 | 27 | 49 | 936,510 |
| 13-17 d | 176 | 23 | 153 | 10,757 |
| 21-28 d | 151 | 101 | 50 | 1 |
| 49-52 d | 88 | 77 | 11 | 704 |
| 104-132 d | 51 | 5 | 46 | 9,049 |

35,541 genes were tested in each window. Gene counts are from `padj` in
`*.deseq2.results.tsv`; anchor counts from `Cjs_0_pval_corrected` in
`splash_out/result.after_correction.scores.tsv`. Treated and control libraries come from
the same colonies in every window except 49-52 d, where colony 6941c has a treated
library (SRR32233984) and no control, so colony blocking cannot absorb it and part of
the 88 genes may be a 6941c effect. In 21-28 d all 5 libraries have under 20,000 read pairs.

### Compared with the paper

The paper's Dataset S1 has its own DESeq2 results for four young-colony comparisons
(sheets Group1, group2, Group4 and group5). By adjusted p < 0.05 those find 0 genes at 2 h,
1 at 24 h, 0 at 13 and 49 days pooled, and 11 at 132 days. The gene lists the paper
reports (its sheet "annotated DE all groups young": 43 up and 237 down at 2 h, 252 and 98
at 24 h, 18 and 48 at 13-49 d, 40 and 46 at 132 d) are therefore not adjusted-p lists.
Their size is near the number of named genes with unadjusted p < 0.05 (37 and 259 at 2 h,
221 and 79 at 24 h), but not equal, so the paper's exact rule is not known. The Methods'
cutoff (p <= 0.05 and absolute log2 fold change > 1) gives 94 and 552 at 2 h in Dataset
S1, more than the 43 and 237 listed, so the lists are cut further in a way not stated.

| Comparison | Paper, adjusted p < 0.05 | Paper, listed up / down | This run, adjusted p < 0.05 up / down | This run, unadjusted p < 0.05 up / down |
|---|---:|---:|---:|---:|
| 2 h | 0 | 43 / 237 | 2 / 138 | 4 / 279 |
| 24 h | 1 | 252 / 98 | 27 / 49 | 82 / 175 |
| 13-49 d (this run: 13-17 d) | 0 | 18 / 48 | 23 / 153 | 39 / 327 |
| 13-49 d (this run: 49-52 d) | 0 | 18 / 48 | 77 / 11 | 358 / 31 |
| 132 d (this run: 104-132 d) | 11 | 40 / 46 | 5 / 46 | 64 / 252 |

The 2 h comparison goes the same way in both (mostly lower in treated). At 24 h the paper
lists mostly higher in treated and this run finds mostly lower. The two analyses differ in
gene models (the paper's 72,115 `g` genes, this run's 35,541 `FUN_` genes), in the model
(the paper fits each comparison alone; this run fits all 52 libraries with colony as a
blocking factor), and in how 13-49 days are grouped. Table: `paper_comparison.tsv`.

### The paper's genes, one by one

![Share of the paper's listed genes found here, and correlation with the paper over all genes](figures/paper_gene_overlap_domen2026_dataset_s1.png)

This run found 30 of the 782 gene entries the paper lists (3.8%) with unadjusted p < 0.05
in the same direction, and 2 of the 782 (0.3%) with adjusted p < 0.05. The two adjusted
hits are g61570 (FUN_035465) and g1353 (FUN_019793), both in the 13-17 d window. g1353
goes the opposite way at 49-52 d (log2 fold change 16.3, 2 against 3 libraries). A listed
gene in the 13-49 d comparison counts as found if either window finds it.

The paper's genes and this run's genes are different gene models, so each paper protein
was aligned to this run's proteins with MMseqs2 and matched to its best hit with at least
90% identity over at least 50% of the paper's protein. These cutoffs were set before any
overlap was counted. 664 of the 782 entries have a match that was tested here. DIAMOND,
the first choice, stalled on the laptop while building its database, on 4 threads and on 1.

The paper's lists are not adjusted-p lists, so a low overlap at adjusted p < 0.05 does not
by itself mean the two disagree. Across all genes tested in both, the two share a signal
that follows the time point (panel b). The Spearman correlation of the signed -log10 p
is 0.16 to 0.44 at the same time point, and between -0.10 and 0.07 at any other pairing.

- Direction: 616 of 723 tested gene-window pairs point the paper's way, where 382 would by
  chance. Chance is the share of genes going that way in that window, leaving out genes
  with a fold change of exactly 0 (20,643 to 34,028 of 35,541 in the four windows from
  13 to 132 d). Every list beats chance except 13-49 d up, which matches it (9 and 13 of
  16, where 9.2 and 12.1 would by chance).
- Overlap at unadjusted p < 0.05, same direction, among genes tested in both: 9 of 433 at
  2 h, 11 of 495 at 24 h, 4 and 10 of 72 at 13-17 d and 49-52 d, 17 of 131 at 132 d. That
  is 5 to 15 times what chance would give, and still a small share of the paper's genes.
- The paper's 12 genes with adjusted p < 0.05: 7 are tested in both. None has adjusted
  p < 0.05 here, and 3 (all at 132 d) have unadjusted p < 0.05 in the same direction.
- A stricter match (95% identity over 80%, or 99% over 90%) keeps the same picture: 517 of
  602 and 344 of 404 pairs point the same way.

This compares the two results. It does not say why they differ, and which of them is
right is not tested here.

Tables: `paper_gene_summary.tsv`, `paper_gene_overlap.tsv`, `paper_gene_overlap_genes.tsv`,
`paper_gene_agreement.tsv`, `paper_gene_agreement_all_windows.tsv`. Code:
`compare_paper_genes.py`, `plot_paper_gene_overlap.py`.

### Each comparison fitted on its own, as the paper did

`make run-alone` refits on the counts of the run above, with each comparison fitted only on
its own libraries and no colony term. It runs nf-core/differentialabundance twice
(`run_each_alone.sh`):

| Output folder | Comparisons | Libraries | Size factors | Variance step (`nsub`) |
|---|---|---|---|---|
| `differentialabundance-paper/` | the paper's four: 2 h, 24 h, 13-49 d pooled, 132 d | the paper's 38 (`samples_paper.csv`) | DESeq2's default (ratio) | 1000 (default) |
| `differentialabundance-each-window/` | this run's six windows | all 52 (`samples.csv`) | poscounts | 10 |

![Rank correlation of refitted p-values with the paper's, by design](figures/paper_design_check_domen2026_dataset_s1.png)

Why these settings:

- **No colony term.** The paper's Methods do not give its design. Refitting the paper's
  own counts (Dataset S1, sheet "Pacemaker_Counts_Table") with `~ treatment` gives back
  its p-values: rank correlation 0.9995 to 0.9999 in all four groups, and 179 of 179
  genes at p < 0.05 at 132 d. With colony added the correlation is 0.58 to 0.80
  (`each-alone-checks/paper_design.out`).
- **The paper's 38 libraries.** Dataset S1's sheet "DEseq group annotation" lists them.
  They are the 52 here less 9 of 12 libraries at 13-17 d and all 5 at 21-28 d. The paper's
  Methods say libraries with fewer than 10,000 counts were removed. Here the 14 left out
  have at most 17,760 read pairs and 2,823 gene counts. The 13-49 d group keeps 3
  libraries at 13-17 d and all 5 at 49-52 d.
- **Size factors.** These are the per-library scaling factors that correct for sequencing
  depth. DESeq2's default uses only genes counted in every library, and that is what the
  paper used. 392 genes are counted in all 38 libraries, so the default works for the
  paper's set. It cannot work for all 52, so the six-window run keeps poscounts (see
  Settings). The choice matters at 2 h: fitted alone, the default gives 686 genes at
  p < 0.05, and poscounts gives 171 (`each-alone-checks/ratio.out`).
- **`nsub` 10 for the six windows.** The variance step stops unless at least `nsub` genes
  have a mean normalised count above 5. Fitted alone, 21-28 d has 14 such genes, 13-17 d
  has 490 and 49-52 d has 992 (`each-alone-checks/fits.out`), so the default of 1000 would
  stop the run. `nsub` changes only the transformed tables and the report's plots, not the
  fold changes or p-values.

Before launch, each comparison was fitted on the laptop in the pipeline's own DESeq2 image
(`each-alone-checks/`). The pipeline's own DESeq2 script was also run, with this run's
settings, on 21-28 d and on the paper's 13-49 d group. Both finished with the design
`~ 0 + group`, on 5 and 8 libraries. Fitted alone, 21-28 d tests 1,513 genes and finds
none at p < 0.05, so expect nothing from it. The pipeline's own checking step
(shinyngs `validate_fom_components.R`) accepts both new sample and comparison files.

On Sherlock:

```bash
D=/scratch/users/olgabot/2024-kmerseek-analysis-botryllus-pec-young; cd $D && git pull && cd nextflow-runs/botryllus-pec-young && make run-alone
```

Watch with `make watch-alone`, stop with `make stop-alone`.

| Task | Resources (pipeline defaults) | Measured in the 2026-11-01 run | Expected |
|---|---|---|---|
| Nextflow head (`nf-botryllus-pec-alone`) | 2 CPUs, 8 GB, 12 h | | about 1-1.5 h for both runs |
| DESeq2, one per comparison (4, then 6) | 1 CPU, 6 GB, 4 h | 1.3 GB, 2 min 15 s each | same or less: fewer libraries per fit |
| Report (Quarto) | 2 CPUs, 24 GB (`sherlock.config`; default 12 GB), 4 h | 10.9 GB, 4 min 34 s | same or less |

## Outputs

In `$SCRATCH/botryllus-pec-young/`:

| Folder | What is in it |
|---|---|
| `rnaseq/multiqc/star_salmon/multiqc_report.html` | per-library read counts, alignment rate, strand, quality |
| `rnaseq/star_salmon/salmon.merged.gene_counts.tsv` | reads per gene, per library |
| `differentialabundance/tables/differential/` | one table per window: gene, log2 fold change, p-value, adjusted p-value |
| `differentialabundance/report/` | HTML report with PCA, volcano plots per window |
| `differentialabundance-paper/`, `differentialabundance-each-window/` | the same, with each comparison fitted on its own libraries (`make run-alone`) |
| `splash/<window>/splash_out/result.after_correction.scores.tsv` | anchors with corrected p-value and effect size |
| `splash/<window>/splash_out/result_compactors.tsv` | anchors extended into longer sequences |

## Related work

- PR 201 (notebook 361) runs SPLASH, an anchor-count model and DE-kupl on these reads for
  the paper's six groups, which include old colonies and pool 13 with 49 days. Its SPLASH
  binary is reused here.
- PR 203 recovered the trimmed reads used here and runs gene-level models on Kallisto
  counts.
