# DE-kupl and KaMRaT: significant k-mers per contrast

Run: `nextflow-runs/pec-domen2026`, Sherlock head job 50675765, finished 2026-11-01 11:41
(15 tasks, 0 failed). Counts read from `$SCRATCH/nb361-pec/contrasts/<contrast>/` on
2026-11-01.

- DE-kupl: rows in `dekupl_limma/DEkupl_result/control_vs_PEC_kmer_counts/diff-counts.tsv.gz`
  (k-mers kept by DE-kupl's limma test) and `merged-diff-counts.tsv.gz` (those k-mers joined
  into contigs).
- KaMRaT: from `kamrat_ttest/summary.txt` and `contigs_all.tsv.gz`. A k-mer is kept when its
  t-test p-value, corrected for the number of k-mers tested (Benjamini-Hochberg), is 0.05 or
  less. "Smallest corrected p" is the smallest `rep-value` over all contigs.

| contrast | DE-kupl k-mers kept | DE-kupl contigs | KaMRaT k-mers tested | KaMRaT k-mers kept | KaMRaT smallest corrected p |
|---|---:|---:|---:|---:|---:|
| control (simulated, known answer) | 10419 | 405 | 2761527 | 196784 | not read |
| young_2h | 0 | 0 | 48121412 | 0 | 1 |
| young_24h | 0 | 0 | 52904962 | 0 | not read |
| old_24h | 0 | 0 | 4227531 | 0 | 0.999995 |
| young_13_49d | 0 | 0 | 2018979 | 0 | not read |
| young_132d | 0 | 0 | 1397827 | 0 | 0.212515 |
| old_41_93d | 0 | 0 | 2190081 | 0 | not read |

Both k-mer methods keep k-mers in the control, where 400 genes were changed 4x or 9x, and
keep none in any of the six real contrasts. So the KaMRaT searches on the real contrasts
had no contigs to search, and their outputs are empty.

Not yet checked: whether the real contrasts have too few samples for a per-k-mer test across
millions of k-mers, or whether the changes the paper reports are too small to pass at any
k-mer. Comparing with the SPLASH and anchor-count results is the next step.
