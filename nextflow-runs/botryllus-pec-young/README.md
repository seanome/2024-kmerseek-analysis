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
| rnaseq `strandedness` | see below | Salmon cannot work out the strand from SRR32233927's 636 pairs, which stopped the PR 203 run |
| rnaseq `min_mapped_reads` | 0 | keep every library (default drops libraries under 5% aligned) |
| rnaseq `skip_deseq2_qc` | true | its DESeq2 step fails when no gene has a count in every library |
| rnaseq `skip_biotype_qc` | true | the FullNames GTF has no `gene_biotype` |
| differentialabundance `deseq2_sf_type` | poscounts | the default size factors need genes counted in every library |
| differentialabundance design | `~ colony + group`, all 52 libraries in one model | group = window and treatment; colony (4 genotypes) as blocking factor |
| SPLASH `--Cjs_samplesheet` | control -1, treated +1 | tests treated vs control rather than any difference between libraries |
| SPLASH `--pvals_correction_col_name` | `Cjs_0_pval` | corrects the p-value for those weights |

All other settings are the pipelines' defaults.

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
`$SCRATCH/botryllus-pec-young/job_ids.txt`.

## Outputs

In `$SCRATCH/botryllus-pec-young/`:

| Folder | What is in it |
|---|---|
| `rnaseq/multiqc/star_salmon/multiqc_report.html` | per-library read counts, alignment rate, strand, quality |
| `rnaseq/star_salmon/salmon.merged.gene_counts.tsv` | reads per gene, per library |
| `differentialabundance/tables/differential/` | one table per window: gene, log2 fold change, p-value, adjusted p-value |
| `differentialabundance/report/` | HTML report with PCA, volcano plots per window |
| `splash/<window>/splash_out/result.after_correction.scores.tsv` | anchors with corrected p-value and effect size |
| `splash/<window>/splash_out/result_compactors.tsv` | anchors extended into longer sequences |

## Related work

- PR 201 (notebook 361) runs SPLASH, an anchor-count model and DE-kupl on these reads for
  the paper's six groups, which include old colonies and pool 13 with 49 days. Its SPLASH
  binary is reused here.
- PR 203 recovered the trimmed reads used here and runs gene-level models on Kallisto
  counts.
