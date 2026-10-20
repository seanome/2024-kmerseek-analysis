# settings.tsv: the six kmerseek settings and their Sherlock asks

The six settings are the dark-set run's six H/P settings that notebooks 279 (sections 7
and 9) and 290 read (`SIX_SETTINGS` in `scripts/290_shuffle_control/tasks.py` on
olgabot/dark-set-shuffle-control). Every one is scaled 1, low-complexity mask off. The
extension penalty and X-drop are the values in the dark-set run's own commands, read on
2026-10-20 from `/scratch/users/olgabot/290-shuffle-control/plan/tasks.tsv` (which
notebook 290 checked text-for-text against the run's `.command.sh` files). The search
filters are the dark-set run's: threshold 0.0, min shared k-mers 2, max query p-value
0.05, min region score 1.3, chain max gap 30, chain max shift 10.

Asks, sized from measured runs, not guessed:

* Dark set, Swiss-Prot minus Mammalia whole proteins (175,144,459 residues), 2,000-query
  chunks (`290-shuffle-control/plan/resources.tsv`, human row): peak memory and the
  longest chunk, per setting. hp_lehninger2 k17: 176 GB, 2.7 h (4.9 s per query);
  hp_pbotc_1st_ed2 k19: 82 GB, 1.4 h; hp_lehninger_c_nonpolar2 k19: 79 GB, 0.9 h;
  hp_pbotc_1st_ed2 k21: 52 GB, 0.5 h; k23: 44 GB, 0.3 h; hp_lehninger2 k25: 40 GB, 0.2 h.
* Regions-index pilot, the same regions index with its decoys (91,473,924 residues, twice
  the one searched here), 100-query chunks (`runs/full/*.trace.txt`): H/P k19-k22 index
  build and fit at most 40 GB and 6 min; a search at most 45 GB and 11 min, mostly
  loading the index (CPU use 0.8-3 of 8 cores).

The regions index searched here has 45,736,962 residues, 3.8 times fewer than the whole
protein index, so `sec_per_query` is the dark-set time per query divided by 3.8 and
doubled. `mem_gb` is the larger of the pilot's measured peak and the dark-set peak divided
by 3.8, times 1.5, rounded up to 64 or 96 GB. `index_hours` covers the build and the
Karlin-Altschul fit (2,500 queries, as the pilot's regions index), 6 min measured at k19
to k22; k17 gets 1 h because it was never built on this index.
