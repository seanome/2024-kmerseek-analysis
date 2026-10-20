# Notebook 353 report: does reduced-alphabet search find existing measurements faster?

Written 2026-10-20.

## Verdict: INSUFFICIENT_EVIDENCE

The pass rule needs 100 consecutive real queries from an analyst's queue, the analyst's
own workflow, blind usefulness judgments and timed sessions. None of these exist yet, so
the rule cannot be applied. Exact reasons, each enough on its own:

1. No analyst queue. No queue of real 15-60 residue queries was found in the repository,
   in `/Users/olga/data`, `~/Documents` or `~/Downloads`. 0 of the 100 evaluation queries
   and 0 of the 40 development queries exist.
2. No description of workflow C (the analyst's current workflow).
3. No usefulness judgments, so no setting could be chosen and nothing could be frozen on
   real data.
4. No timed sessions, so the 50% handling-time rule cannot be tested. Machine run time
   is not a substitute (plan, `pass_rule.machine_time`).
5. Workflow A (kmerseek) has not been run: it runs on Sherlock only, and the compute
   budget below has not been agreed.

No query, judgment or timing was invented. The software was tested end to end on
synthetic fixtures, and on those it returns INSUFFICIENT_EVIDENCE as it must
(`results/synthetic_fixture_run/verdict.json`).

## What exists and what was built

| piece | state |
|---|---|
| kmerseek search with region coordinates on both proteins | exists (v0.4.0); used as is, not reimplemented |
| region score and E-value | exist with `--extend-mismatch-penalty` (`region_ka_bits`, `region_evalue`); see the E-value section |
| ELM data | downloaded again 2026-10-20 with ELM's own sequences |
| Sanborn 2021 activation tiles | downloaded 2026-10-20 from eLife |
| evidence collection with sources, coordinates, sequence versions | built: `mr353.py import-evidence` |
| query import, split, copy check | built: `import-queries` |
| shuffled controls with low-variety flags | built: `make-controls` |
| workflow B (exact lookup + BLAST set for query length) | built, runs on the laptop: `search-b` |
| workflow A on Sherlock | job scripts written, not submitted: `sherlock/` |
| evidence cards, self-record removal, shuffle-based expected count | built: `cards` |
| tuning on development queries, freeze with checksums | built: `tune`, `freeze` |
| balanced session order, blind review sheet and sealed key | built: `session-pack`, `review-pack` |
| scoring with uncertainty and carry-over | built: `score` |
| local HTML viewer | built: `viewer`; example on synthetic data in `results/synthetic_fixture_run/viewer.html` |

## The evidence collection

| source | records | details |
|---|---|---|
| ELM (download 2026-10-20, 4,277 instances in the file) | 4,275 | true positive 4,045, true negative 33, false positive 73, unknown 124; 2 pairs of instances were the same stretch, class and verdict on the same sequence and were merged |
| Sanborn et al. 2021, Figure 1 source data 1 | 7,457 | active 448, weak 358, inactive 6,564, not measured 87 |
| total | 11,732 records on 2,883 distinct protein sequences | |

Checks made on the sources before use:

- ELM Start and End are 1-based and inclusive on ELM's own copy of each sequence
  (`instances.fasta`). Every instance lay inside its sequence; none was dropped.
- Sanborn's `position` column is 0-based: every one of the 7,457 tiles was found at that
  offset in the paper's own TF sequence, and at no other offset.
- The paper says 7,460 tiles. The file has 7,457 rows named `A_tiles_*`. The three-tile
  difference is not explained in the paper or the file.
- The paper counts 451 tiles significant at p < 0.0001 (Z-test). Z >= 3.719 (one-sided
  p < 0.0001) gives 448 here. The difference may be the same three tiles; it was not
  traced further.
- Activation is the fold increase in GFP over background in a yeast reporter assay. Each
  card shows the value, its units, the Z score and the assay.
- Sequence versions: all 7,457 tile sequences match current UniProt. Of the ELM
  records, 4,006 match current UniProt, 126 sit on a sequence that has changed since, and
  143 are on isoforms, which were not looked up. Cards say which.
- A tile's ends come from the library design (53 aa, step 12-13 aa), not from the
  residues the activation needs. Every tile card says so.

## Are the E-values E-values?

| statistic | counts database size | counts choosing the best region or alignment | counts choosing the alphabet, k or threshold | shown as |
|---|---|---|---|---|
| kmerseek `region_evalue` | yes (query length x k-mers in the database) | yes, within one query-target pair | no | "kmerseek region_evalue (not calibrated for this use)" |
| BLAST E-value | yes | yes | no | "BLAST E-value", with the task |
| expected cards this good from one shuffle (this tool) | yes: the shuffles search the same database | yes: shuffles go through the same search, region choice, evidence mapping and self-record removal | the choice is made on development queries only, so it does not touch the evaluation | the number the cut-off uses |

Notebook 344 found kmerseek's `region_evalue` 10 to 113 times too small for proteins of
300 aa or more with at least 15% low-complexity sequence. This is why it is shown but
not used for the cut-off. The shuffle-based number has a floor: with 20 shuffles it
moves in steps of 0.05, so it cannot tell a card expected once in 100 shuffles from one
expected once in 1,000.

## Problems found while building, which the real run will face

- **Overlapping tiles.** Sanborn tiles overlap by about 40 residues, so one match to a
  transcription factor often returns 3 or 4 tile cards from the same protein. Cards are
  one per measured record, so the reviewer reads all of them. This adds reading time to
  every workflow that finds the protein, A and B alike.
- **Short ELM stretches.** The median ELM instance is 6 residues. A match counts if it
  covers min(5, stretch length) residues of the stretch, so a 15-60 residue match near a
  motif may cover it only in its last few residues.
- **Low-variety queries.** On the 140 synthetic queries, 14 of 280 query-and-shuffle-kind
  pairs could not give 20 distinct shuffles (flagged in `control_diversity.csv`). Real
  queries rich in repeats will be flagged more often, and their shuffle-based numbers are
  then less reliable.
- **Two-letter alphabets at short length.** At hp_lehninger2 k=14, a 15-residue query has
  2 k-mers. The grid keeps k at most 14, and the shuffles test whether such short seeds
  find anything beyond chance.
- **The control check is close to the cut-off by design.** A shuffle compared against
  the other 19 keeps about as many cards as the cut-off allows, and at most 0.05 more
  (on the synthetic run: 0.19 and 0.23 at cut-off 0.25). Much more than that on real
  development queries would mean the shuffles do not behave alike, and would need a
  look before freezing.
- **A changed rule.** The first draft of the tuning rule asked for at most 0.05 kept
  control cards per shuffle. That can never be met at any cut-off above 0, because a
  shuffle's best card always beats the other 19. Seen on the synthetic run, where
  workflow B got no setting at all. Replaced, before any real data, by a cap on the
  cut-off (at most 0.25 chance cards per query), with the control count kept as a check.

## Compute

Measured on the laptop (Apple M4 Max, 16 cores, 128 GB), with 4 threads at most.

| step | data | elapsed | CPU time | peak memory | disk |
|---|---|---|---|---|---|
| import-evidence, first run with downloads | real collection | 46.9 s | 2.1 s | 256 MB | raw files 9.9 MB, evidence 12.7 MB |
| import-evidence, downloads cached | real collection | 1.4 s | 4.4 s | 252 MB | same |
| BLAST database build (index) | 2,883 proteins | 0.04 s | | | 3.1 MB |
| search-b (exact + BLAST) | 140 synthetic queries + 5,408 shuffles vs the real collection | 30.8 s | 79.6 s | 1.53 GB | 2,222,465 hits; folder 63 MB |
| cards | the same | 21.6 s | 21.9 s | 1.71 GB | folder 70 MB |
| all tests and the synthetic end-to-end run | synthetic | under 30 s | | under 0.5 GB | |

Everything above used about 0.04 CPU-hours. Machine time says nothing about the
analyst's handling time, which is what the pass rule measures.

Workflow A was not run. The proposed Sherlock budget (in the plan) is one kmerseek build
(8 CPUs, 1 h) plus one index-and-search task per grid setting (12 settings, 4 CPUs,
16 GB, 2 h each): at most 104 CPU-hours, and under 5 GB of disk. The database is
2,883 proteins, about 1.7 million residues, so the real cost is expected to be far
below the ceiling. These asks are not measured: no kmerseek run on this database
exists yet. Record the measured numbers here after the first run
(`sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State`).

## What would let this reach PASS or FAIL

1. An analyst's queue of at least 140 consecutive real queries (40 development, then
   100 evaluation), saved in `templates/analyst_queue.csv` form before any result is seen.
2. The analyst's description of workflow C, committed before the evaluation.
3. Agreement on the Sherlock budget; then `make sherlock-inputs` and `submit.sh`.
4. Blind labels on development cards, then `tune` and `freeze`.
5. The timed sessions and the blind review, then `score`.
