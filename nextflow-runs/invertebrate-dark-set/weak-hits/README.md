# Can a tool's weak hits be trusted?

When a tool's best hit on a protein is weak (E between 1 and 10), how often is it right?
[PR 133](https://github.com/seanome/2024-kmerseek-analysis/pull/133) answered this for
phmmer, jackhmmer, MMseqs2 and one kmerseek setting per species. This folder extends it to
every kmerseek setting of the dark-set midi run (`runs/ladder-0.4-midi`), splits every best
hit into E-value bands, and adds a stricter check that looks at positions on the query.

## Rules, written 2026-10-16 before any new result was read

What had been seen before these rules were written: PR 133's table (dark proteins at
E <= 1: human 1,071 with a known Pfam family, phmmer 24 right / 716 wrong at E 1-10,
kmerseek hp_lehninger2 k=17 0 right / 270 wrong at region E <= 1; worm, yeast and
*Borrelia* rows; placed proteins 69-99% right), its perilipin-4 spot check, and the file
counts and sizes of the region files. No count from the stricter check, no count for any
other kmerseek setting, and no count split by band had been seen.

**Species.** Human, *C. elegans* (worm), *S. cerevisiae* (yeast), *B. burgdorferi*
(*Borrelia*). *Botryllus* is left out: it has no Pfam annotation.

**Sources.** phmmer, jackhmmer, MMseqs2 (the run's own hits files), and every kmerseek
setting the run fitted (listed in `by_combo` of `<species>_kmerseek_dark_gain.json`).
A setting the run could not fit has no E-values and is counted as not run.

**Best hit.** For each protein and source, the hit with the smallest E-value (for
kmerseek, the region with the smallest `region_evalue`). Ties keep the first row, as in
PR 133.

**Bands** of the best hit's E: at most 0.001; above 0.001 up to 1; above 1 up to 10.
A protein whose best hit is above 10 (or that has no hit) is in no band.

**Dark and placed.** Dark at T: no phmmer, jackhmmer or MMseqs2 hit with E <= T. Placed at
T: the rest. T = 1 is the headline; 0.001 and 10 are also written. Only proteins with a
known Pfam family count.

**Two calls of a best hit.** Both use the target's Pfam families (DR lines of its
Swiss-Prot entry) and the query's (Pfam's annotation of UniProt; for *Borrelia*, UniProt's
Pfam cross-references, with positions from the InterPro API).

* Family or clan (PR 133's call): right when the target shares a Pfam family or a Pfam
  clan with the query.
* Same family and same place (the strict call, the headline on the cards): right when
  the target shares a Pfam family with the query AND the stretch of the query that the
  hit covers overlaps, by at least one residue, a Pfam domain of that shared family on the
  query.
* Both have one denominator: best hits whose target has any Pfam family ("judged").
  Targets without Pfam families are counted apart.

**Where the query stretch comes from.** kmerseek writes it (`region_start`, `region_end`,
0-based; converted to 1-based). The sequence tools' hits files keep only target positions,
so each sequence tool's best target is aligned to its query again with phmmer (the run's
own HMMER 3.4 image, `--max`, all domains reported), and the domain whose target stretch
overlaps the tool's target stretch most gives the query stretch. For phmmer this repeats
the same alignment. For jackhmmer and MMseqs2, which align with a profile built over
iterations, it is the stretch that a single-sequence alignment of the same two proteins
gives. When no realigned domain overlaps, the strict call counts the hit as not right,
and the number of such hits is in the column `strict_query_stretch_unknown`.

**Precision** = right / judged, with a 95% Wilson interval, and n (judged) beside it.
A value with fewer than 30 judged hits is shown grey.

**Choosing the kmerseek setting for a card.** Proteins are split once into tune and test
halves by the md5 of the UniProt accession (even = tune). Per species, the setting with
the highest strict precision at E 1-10 on dark tune proteins, among settings with at
least 30 judged hits there, is the card's setting; ties go to more judged hits, then to
the setting name. Its precision is reported on the test half. The sequence tools are not
chosen from anything, but the leaderboard column also uses the test half for them so that
every tool is scored on the same proteins.

**Findable.** A protein is findable when some protein in the clade-excluded reference
carries one of its Pfam families. A protein that is not findable cannot be right. The
count of findable dark proteins is reported per species.

## Predictions, written with the rules

1. The strict call is right less often than the family-or-clan call for every source.
2. On dark proteins at E 1-10, phmmer, jackhmmer and MMseqs2 are right (strict) in under
   5% of judged best hits in every species.
3. The chosen kmerseek setting is right (strict) in under 10% of judged dark best hits at
   E 1-10 on the test half, in every species.
4. On placed proteins at E <= 0.001, MMseqs2 is right (strict) in over 90% of judged
   best hits in every species.

## Limits found after the results were read (2026-10-16, independent review)

* Prediction 1 cannot fail: the strict call only marks a hit right inside the shared-family
  branch of the loose call, so strict is never above loose by construction.
* The kmerseek pick is close to arbitrary. On the tune half the winner led by one right hit
  of 268 (human) and two of 37 (worm); in yeast and *Borrelia* every eligible setting had 0
  right, so the most judged hits decided it.
* phmmer and jackhmmer agree on dark proteins for human, yeast and *Borrelia*, not fully for
  worm (1,431 judged for jackhmmer, 1,430 for phmmer): the best target differs for 1 of
  1,322 human and 14 of 4,028 worm dark proteins. Why was not traced.
* 243 human Pfam domains (worm 13, yeast 1) end past the end of the protein in the
  annotation table, so Pfam's positions for those proteins seem to come from an older
  sequence version. The strict call can be off for them; nothing flags it.
* The phmmer re-alignment matches on the target envelope (env from/to) but takes the query
  stretch from the alignment (hmm from/to), a slightly narrower span.

## Steps (run 2026-10-16)

Tools ran from `git archive` of this branch out of `$SCRATCH/2024-kmerseek-analysis`
(commit 13749b8 for steps 1-3, 8292a0f for step 4; Sherlock's git 1.8 has no worktrees).
`A=$SCRATCH/invertebrate-dark-set/results-0.4/accuracy`.

1. Plan, on a login node: `python3 tools/dark-set-accuracy plan-regions
   /scratch/users/olgabot/invertebrate-dark-set/results-0.4 60 $A/weak_plan_60.tsv`.
   4,361 region files of fitted settings, 5.22 TB (human 1,837 files of 167 settings,
   worm 1,690 of 169, yeast 660 of 165, *Borrelia* 174 of 174). Each species also has
   56-65 settings the run could not fit; they wrote no E-values and are skipped.
2. Best region per query and file: `sbatch --array=1-60 --time=02:00:00 --mem=2G
   tools/dark-set-accuracy-sbatch best-regions $A/weak_plan_60.tsv`, job 48132790. First
   ask from PR 133's job 47672974 (26 files, about 2 min each, 260 MB). 54 tasks finished in
   43 min to 1 h 40 min. Six tasks held files up to 59.5 GB (hp_lehninger_hpc3 k=17) and
   were cancelled at 13-38% done; their 151 unread files (377 GB) were re-planned over 24
   tasks (`weak_plan_rest.tsv`) and read by job 48142858 (`--time=04:00:00`), 2-26 min each.
   Peak memory of the Python process 15 MB (`/usr/bin/time`); Slurm's MaxRSS of 2 GB in the
   first array counts the file cache. Read speed 16 MB/s median per file, one CPU.
3. Query stretch of the sequence tools' best hits: `sbatch --cpus-per-task=8
   tools/dark-set-accuracy-sbatch realign <species>`, jobs 48133064 (yeast, 4 min, 0.8 GB),
   48134152 (human, 11 min, 3.6 GB), 48134154 (worm, 13 min, 4.2 GB of a 4 GB ask, finished),
   48134157 (*Borrelia*, 1 min). *Borrelia* domain positions: `query-domains-interpro`
   (1,032 proteins, 1,553 domain rows; a second fetch from the laptop gave an identical file).
4. Scoring: `sbatch --mem=8G tools/dark-set-accuracy-sbatch score-bands`, job 48146170,
   1 min 36 s, 1.9 GB. Before it, job 48143874 scored only PR 133's 26 region files
   (`BEST_ROOT`, `OUT`) for the recount. Outputs copied to `weak-hits/data/`.
5. On the laptop: `python weak-hits/summarize.py` writes `summary.txt`, the figures, and
   `results/leaderboard/weak_hits/` for the site.
