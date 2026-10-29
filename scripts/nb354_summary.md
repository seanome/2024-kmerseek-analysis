## Summary: FAIL under the fixed rule

kmerseek with the mass-spec alphabet (massspec18, k = 8, chosen on the development split)
found the accepted source of 140 of 200 test de novo calls in its top five, and gave a
confident source to 4 of the 50 calls whose source had been removed. The rule asked for at
least 95% and at most 1%, so the test fails. The comparison (exact full-alphabet index plus
blastp-short, with the same full-alphabet check) did worse on both counts: 128 of 200 and 6
of 50. Neither workflow reaches the bar. About 40% of these de novo calls differ from the
accepted database-search peptide (rule 1), and some removed sources have a close paralog
left in the database (rule 2). These observations do not isolate the cause of every failure.

| Rule | Needed | kmerseek massspec18, k = 8 | Holds | Comparison C | Section |
|---|---|---|---|---|---|
| 1. accepted source in the top five, source-present calls | >= 95% | 140 of 200, 70.0% | no | 128 of 200, 64.0% | 4, 10 |
| 2. confident source, source-absent calls | <= 1% | 4 of 50, 8% | no | 6 of 50, 12% | 4, 10 |
| 3. median candidate-list length | <= 5 | 1 | yes | 1 | 5, 10 |
| 4a. recovery gain over C at the same false-assignment rate | >= 10 points | 70.0% against C's 53.5% at <= 8%: +16.5 points | yes | | 4, 10 |
| 4b. end-to-end speed-up at matched recovery | >= 5x | 125 s / 27 s = 4.6x (recovery matched) | no | | 6, 10 |

Compute for the whole notebook, every SLURM job included (section 9): 8.2 CPU-hours
allocated (cores x elapsed), 3.1 CPU-hours used; 1.2 hours elapsed summed over jobs, and
1 hour 29 minutes of wall clock from the first job's start (14:07) to the last job's end
(15:36) on 20 Oct 2026. The budget was two compute-days.

What the numbers say, in plain words:

- **kmerseek's lead over C is a threshold effect, not a better alphabet.** Both workflows
  give the accepted source the same check score whenever both send it to the check
  (section 4, scatter). kmerseek sends a median of 1 candidate per call, C a median of 221,
  so shuffled calls reach high scores less often under kmerseek and its 1%-of-shuffles
  threshold sits lower (62 against 66). Of the 15 calls only kmerseek recovers, 14 have the
  same score, 62 to 65, under both; the 15th (scan13458) was not among C's blastp-short hits.
  The 3 calls only C recovers share no exact 8-mer with their source.
- **Merging residues adds nothing here.** An exact 8-mer in massspec18 is the same as an
  exact 8-mer in the full alphabet once I is read as L and Q as K, which C also does. The two
  coarser alphabets, hsdm17 and sdm12, gave the same development recovery as massspec18 at
  k = 8 (70.0%) and were not sent to test (section 3).
- **The adjacent-pair shuffle is not a usable control for calls this short.** 566 of 1,250
  such shuffles of the test calls are the call itself, which is why both workflows assign
  34-36% of them (section 7). The composition shuffles behave: 1.0% (kmerseek) and 1.5% (C).
- **The fail clause is not what failed.** Reduction did not lengthen candidate lists (median
  1 sent to the check against C's 221), and the check costs kmerseek almost nothing (0.007 s).
  The test fails on rules 1 and 2. Rule 4 passes through the recovery-gain alternative
  (4a); the speed alternative (4b) fails.
- **Part of rule 1 is out of reach at k = 8.** 22 of the 200 source-present test calls are
  shorter than 8 residues and count as abstentions; 81 differ from the database-search peptide; this can reflect a de novo error or an
  incorrect accepted match.
- **A second reader recomputed every number** and found the FAIL holds (last section).

Decision: kmerseek is not pitched for source lookup of de novo peptide calls on the strength
of its alphabet. If a seed-and-check route is wanted, the same gain is available from an exact
full-alphabet 8-mer seed with I read as L and Q as K; that is untested here.

### Deviation from the pre-registration

One development setting, sdm12 at k = 4, was stopped after 40 minutes with 43 GB of search
output (about 5,000 candidate proteins per call), before any development score was read. It
is recorded in `STOPPED` in `nextflow-runs/fragment-source/bin/nb354.py` and in
`354_data/dev_choice.json`, and was not run or chosen. Nothing else changed.

### Inputs and runs

- Frozen data: `data/kill_tests/fragment_source/` (sha256 in `manifest.tsv`; downloads in
  `download_manifest.tsv`).
- Pipeline: `nextflow-runs/fragment-source/` (`make all` on Sherlock), kmerseek 4d7e27b
  (seanome/kmerseek PR 156), BLAST+ 2.16.0, parasail 1.3.4.
- Sherlock jobs: download 48678939, kmerseek build 48679601, freeze 48681774,
  development 48682102 (stopped) and 48689384, test 48690317, summary 48690319.
- Builder: `scripts/make_nb354.py`; pre-registration text `scripts/make_nb354_prereg.py`,
  committed in 3743f18a before any method ran on the frozen calls (a smoke test at 14:27
  searched 10 calls of an earlier freeze, none of the final 500, and computed no score).
