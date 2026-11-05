# Data size at each step (notebook 361)

![Size at each step](sizes_by_step_botryllus.png)

Measured 2026-11-05 by `scripts/route_sizes.py` (Sherlock job 51207173). Figure:
`scripts/plot_route_sizes.py`. Numbers: `raw_fastq.tsv`, `route_sizes.tsv`.

Botryllus, 47 samples, six contrasts added up. A contig kept in two contrasts counts twice.

| step | reads | SPLASH | KaMRaT (t-test, p < 0.05) |
|---|---|---|---|
| FASTQ, gzip | 76.7 GB | | |
| bases | 193.1 GB (643.5 M read pairs) | 99.6 MB | 6.1 MB |
| contigs | | 1,116,488 | 167,860 |
| peptides of 20+ residues | | 4,285,316 | 28,333 |
| DNA sketch, k=31 | | 917 MB | 119 MB |
| protein sketch, k=7 | | 1.69 GB | 10.1 MB |
| Dayhoff sketch, k=12 | | 1.64 GB | 9.7 MB |
| H/P sketch, k=30 | | 1.47 GB | 8.7 MB |

All sketches use scaled=5, which keeps about one k-mer in five.

- Keeping contigs cuts the bases about 1,900 times for SPLASH and 31,500 times for KaMRaT.
- Sketches are larger than the contigs, because each contig or peptide has its own small file in the
  zip: about 820 bytes per contig (DNA) and 390 bytes per peptide (protein). The Dayhoff sketches
  hold 13.4 M k-mers in all, which at 8 bytes each is 107 MB, against 1.64 GB on disk.
- KaMRaT contigs average 37 bases, shorter than the 60 bases a peptide of 20 residues needs, so few
  give a peptide. That is why its protein sketches are smaller than its DNA sketches.
- The DNA sketches were made only to measure this. The search uses protein, Dayhoff and H/P.

Human: FASTQ gzip is 175.3 GB for GSE120862, 59.8 GB for GSE144304 and 19.8 GB for GSE99963, all
samples of each series. Contigs and peptides per contrast are in `route_sizes.tsv`. Peptide counts
are 0 there where the peptide files were not kept, and the human sketch zips were not kept, so human
sketch sizes are not measured.
