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
