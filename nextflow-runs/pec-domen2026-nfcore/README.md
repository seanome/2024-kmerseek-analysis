# nf-core reference for notebook 361

nf-core/rnaseq 3.27.0 (STAR + RSEM, April 2024 Botryllus genome) and nf-core/differentialabundance
2.0.0 (DESeq2) on the 47 PEC samples, chosen by Olga on 2026-11-04 as the reference to score the
k-mer routes against, instead of the paper's lists.

Differences from `nextflow-runs/botryllus-pec-young` (PR 207): RSEM instead of Salmon; the routes'
six contrasts instead of six windows, old colonies included; the routes' fastp-trimmed reads; each
contrast fitted on its own samples, with the routes' blocking (pair, or genotype for young 132 d).
