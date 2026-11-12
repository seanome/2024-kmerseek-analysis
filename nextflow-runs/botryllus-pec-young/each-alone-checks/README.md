# Checks for `make run-alone`, run on the laptop before launch (2026-11-06)

All R scripts ran in the image the pipeline's DESeq2 step uses,
`community.wave.seqera.io/library/bioconductor-deseq2_bioconductor-limma:b56a0c9ddc3e87e1`,
from a folder holding the inputs below. They are small (12 to 52 libraries, under 3 min
each). Inputs, not committed: `salmon.merged.gene_counts.tsv` from
`$SCRATCH/botryllus-pec-young/rnaseq/star_salmon/`, `samples.csv` and `samples_paper.csv`
from the folder above, and the files `export_paper_counts.py` writes from Dataset S1.

| Script | Question | Output |
|---|---|---|
| `paper_design.R` | Which design gives back the paper's own p-values from the paper's own counts? | `paper_design.out` |
| `fits.R` | Does each window, and the paper's 13-49 d group, fit alone on this run's counts? | `fits.out` |
| `vst_nsub.R` | Does a smaller `nsub` let the variance step run on 21-28 d? | `vst_nsub.out` |
| `ratio.R` | Do the paper's size factors (DESeq2's default) work on the paper's 38 libraries here? | `ratio.out` |
| `vst_ratio.R` | Does the variance step run with those size factors? | all 5 printed "vst ok" |
| `size_factor_genes.py` | On how many genes do the default size factors rest, per comparison, in the paper's counts and here? | `size_factor_genes.out` |
| `size_factors.R` (on Sherlock, `size_factors.sbatch`, 2026-11-11) | Do size factors cause the gap with the paper's gene lists? Only the size factors change. | `size_factors.out`; first job, stopped part-way: `size_factors.first_try.out` |
| `up_skew_49_52d.R` | Which library makes 86% of genes go up at 49-52 d, fitted alone? | `up_skew_49_52d.out` |
