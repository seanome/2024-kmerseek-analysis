# nf-core reference for notebook 361

nf-core/rnaseq 3.27.0 (STAR + RSEM, April 2024 Botryllus genome) and nf-core/differentialabundance
2.0.0 (DESeq2) on the 47 PEC samples, chosen by Olga on 2026-11-04 as the reference to score the
k-mer routes against, instead of the paper's lists.

Differences from `nextflow-runs/botryllus-pec-young` (PR 207): RSEM instead of Salmon; the routes'
six contrasts instead of six windows, old colonies included; the routes' fastp-trimmed reads; each
contrast fitted on its own samples, with the routes' blocking (pair, or genotype for young 132 d).

`make run-salmon` runs the same genome, annotation, trimmed reads, contrasts and DESeq2 with Salmon in
place of STAR + RSEM, in `nfcore_salmon/`. Asked by Olga on 2026-11-06, to compare the k-mer routes'
cost with the usual fast route as well as with alignment. It uses `salmon quant` on the reads, with no
alignment and no QC steps, so its trace holds only the steps being timed. The cost row comes from
`analysis/pec-domen2026-splash/scripts/alignment_route_cost.py`, given `nfcore_salmon/` as its last
argument.
