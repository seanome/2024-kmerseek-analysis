# Pilot tune half, rescored with both 2026-10-14 changes

`make score-partial MMSEQS2_DIR=mmseqs2-one-round SCORE_OUT=scores-partial-projected`
(Slurm job 47775598 at 2067fa1, 5 min, peak 9.5 GiB). A call is now the target domain placed
on the query through the hit, and MMseqs2 is the one-round search. 186 kmerseek searches had
finished when the pilot stopped (head cancelled 2026-10-13 14:00). Test half unread.

`compare_scorings.py` compares this with the two earlier scorings on the same 186 settings and
62 tune-half P4 instances (on 52 queries):

| | whole hit, 3 rounds | whole hit, 1 round | placed domain, 1 round |
|---|---|---|---|
| MMseqs2 cutoff | E 1.5e-22 | E 0.095 | E 0.095 |
| P4 instances found: MMseqs2 / phmmer / hmmscan | 7 / 24 / 37 | 17 / 24 / 37 | 31 / 41 / 37 |
| P4 instances found by any kmerseek setting (best single setting) | 19 (9) | 19 (9) | 21 (13) |
| P4 instances found by any comparator | 46 | 45 | 52 |
| controls found at any score: MMseqs2 / phmmer (of 23) | 18 / 19 | 17 / 19 | 22 / 23 |
| kmerseek settings with a unique find | 35 | 35 | 0 |

With the domain placed through the hit, every P4 instance a kmerseek setting finds is also
found by MMseqs2, phmmer or hmmscan. P4's pick (`p4.json`) is then a tie at 0 broken by
decoy rate and name, so it means nothing.

`kmerseek_only_any_score.csv` (`kmerseek_only_any_score.py`): tune-half instances that some
kmerseek setting places, at any score, where MMseqs2, phmmer and hmmscan have no landing call
of that family at any score. There are 16 of 115 tune-half instances; none passes kmerseek's
own cutoff. 5 are held back by a score ceiling: `region_poisson_score` stops at 307.65
(-log10 of the smallest normal double, so the p-value underflows), and in 102 of 107 exact
settings two or more tune decoy pairs reach it, which puts the matched cutoff above the
ceiling. Those exact settings then find nothing. 3,561 of 4,920 exact best calls and 561,027
of 5.0 million exact tune decoy pairs sit at the ceiling.
