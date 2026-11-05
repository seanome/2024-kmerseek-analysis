# Data size at each step (notebook 361)

![Size at each step](sizes_by_step_botryllus.png)

Measured 2026-11-05 by `scripts/route_sizes.py` (job 51207173) and `scripts/sketch_settings.py` (job
51225670). Figure: `scripts/plot_route_sizes.py`. Numbers: `raw_fastq.tsv`, `route_sizes.tsv`,
`sketch_settings.tsv`, `fastp_time.tsv`, `sizes_by_step_botryllus.tsv`.

Botryllus, 47 samples, six contrasts added up. A contig kept in two contrasts counts twice. % is of the
gzip FASTQ. Total time runs from the FASTQ to that step: fastp trimming, then the route's steps to
contigs, then that one sketch, one after the other.

| step | SPLASH | % | total wall, CPU | KaMRaT (t-test, p < 0.05) | % | total wall, CPU |
|---|---|---|---|---|---|---|
| kept contigs, bases | 99.6 MB | 0.13 | 2.11 h, 10.5 h | 6.1 MB | 0.008 | 5.76 h, 23.1 h |
| DNA sketch, k=31, scaled=5 | 917 MB | 1.2 | 2.13 h, 10.5 h | 119 MB | 0.15 | 5.77 h, 23.1 h |
| protein sketch, k=7, scaled=5 | 3.31 GB | 4.3 | 2.17 h, 10.6 h | 21.8 MB | 0.028 | 5.76 h, 23.1 h |
| protein sketch, k=7, scaled=20 | 3.18 GB | 4.1 | 2.17 h, 10.6 h | 20.8 MB | 0.027 | 5.76 h, 23.1 h |
| Dayhoff sketch, k=12, scaled=5 | 3.27 GB | 4.3 | 2.17 h, 10.6 h | 21.4 MB | 0.028 | 5.76 h, 23.1 h |
| Dayhoff sketch, k=12, scaled=10 | 3.21 GB | 4.2 | 2.17 h, 10.6 h | 21.0 MB | 0.027 | 5.76 h, 23.1 h |
| H/P sketch, k=30, scaled=5 | 3.14 GB | 4.1 | 2.17 h, 10.6 h | 20.6 MB | 0.027 | 5.76 h, 23.1 h |

For reference: FASTQ gzip 76.7 GB (100 %), bases 193.1 GB (252 %), 643.5 M read pairs. SPLASH keeps
1,116,488 contigs and 4,285,316 peptides of 20+ residues; KaMRaT keeps 167,860 and 28,333.

- **Correction:** the first version of this table gave 1.69 GB, 1.64 GB and 1.47 GB for SPLASH sketches.
  That summed the sketch members of one zip holding all three types and left out the zip's own index and
  manifest. The numbers here are each setting's whole zip on disk, which is what a user stores.
- A higher scaled cuts k-mers, not the file. Protein k-mers fall from 17.4 M to 4.3 M at scaled=20, and
  Dayhoff from 13.4 M to 6.8 M at scaled=10, but the SPLASH zips shrink by 4 % and 2 %. Each peptide
  has its own sketch, about 740 bytes, and nearly all of that is the sketch's wrapping. The k-mers
  alone at 8 bytes each would be 35 MB at protein scaled=20. One sketch per contrast should be far
  smaller. That was worked out, not tested, and such a sketch cannot say which peptide matched.
- Sketching adds little time. All sketches take under 4 min in total per setting; trimming takes 0.61 h
  and getting to contigs 1.50 h (SPLASH) or 5.15 h (KaMRaT), and that KaMRaT time holds DE-kupl's
  limma test, so it is an upper limit. Peptide translation is left out.
- KaMRaT contigs average 37 bases, shorter than the 60 bases a peptide of 20 residues needs, so few
  give a peptide. That is why its protein sketches are smaller than its DNA sketches.
- Contigs are uncompressed bases set against a gzip file, so their % is not like for like.
- The DNA sketches were made only to measure this. The search uses protein, Dayhoff and H/P.

Human: FASTQ gzip is 175.3 GB for GSE120862, 59.8 GB for GSE144304 and 19.8 GB for GSE99963, all
samples of each series. Contigs and peptides per contrast are in `route_sizes.tsv`. Peptide counts
are 0 there where the peptide files were not kept, and the human sketch zips were not kept, so human
sketch sizes are not measured.
