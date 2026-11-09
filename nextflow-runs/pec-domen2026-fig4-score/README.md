# Reads per gene with no genome, for the Fig. 4C-D score (notebook 361)

Counts reads per gene in the three RNA-seq series of Domen et al. 2026 Fig. 4C-D with
`pec-tools gene-counts` (protein 10-mers against GENCODE v50 proteins, no alignment), on all reads
and on the first 1/10, 1/100 and 1/1000 of each run's reads. Plan and rules:
`analysis/pec-domen2026-splash/design/fig4_score_from_reads/README.md`. Results:
`analysis/pec-domen2026-splash/exports/fig4_score_from_reads/README.md`.

It reuses the trimmed reads and proteins that `../pec-domen2026-human` left under
`$SCRATCH/nb361-pec/human/`, and writes to `$SCRATCH/nb361-pec/human/fig4_score/`.

```
ssh sherlock
cd /scratch/users/olgabot/2024-kmerseek-analysis-nb361/nextflow-runs/pec-domen2026-fig4-score
git pull
make run      # all four shares; finished ones are reused
make watch    # the live table
```

## Resources asked, from the pilot

Pilot of 2026-11-09 (head job 51700536): the 1/100 and 1/1000 shares, 16 threads, peak memory
0.46 GB in every task (the protein index, the same whatever the number of reads).

| series | reads | 1/100 share: CPU s, wall | all reads, predicted (x 100) | asked for all reads |
|---|---|---|---|---|
| GSE99963 | 409 M | 100 s, 14 s | 2.8 CPU h, 24 min | 16 CPUs, 4 GB, 6 h |
| GSE144304 | 1,432 M | 183 s, 33 s | 5.1 CPU h, 54 min | 16 CPUs, 4 GB, 6 h |
| GSE120862 | 3,971 M | 438 s, 68 s | 12.2 CPU h, 1 h 53 min | 16 CPUs, 4 GB, 6 h |

The 1/10 share asks 1 h and the two small shares 30 min. A task killed for memory or time is retried
twice with double the ask; any other failure stops the run with an error.
