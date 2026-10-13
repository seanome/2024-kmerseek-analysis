# Pilot scores from the 185 searches finished by 2026-10-13

Partial scoring of the nb 280 time-split pilot, tune half only (the test half is
dropped before any call is read). Made with `make score-partial` (Slurm job 47672527,
code at ff2b055, 6 min, peak 8.2 GiB) from `$SCRATCH/280-time-split/results-pilot/`.
The pilot was still running: 185 of its 256 kmerseek searches had finished (106 exact,
79 extend), plus MMseqs2, phmmer and hmmscan. The 71 missing searches are listed in
`setting_coverage.csv` (chunks = 0).

| File | What it holds |
|---|---|
| setting_summary.csv | per setting: calls, queries with a call, controls found at any score, tune-half cutoff (decoys get at most 1 family transfer per 100 decoy queries) |
| unique_finds_tune.csv | per kmerseek setting: P4 instances found that no other tool finds, minus the number expected from its decoy rate |
| p4.json | the setting the P4 rule picks on the tune half |
| controls.csv | each positive control's best call per setting |
| parse_checks.csv, kmerseek_files.csv, setting_coverage.csv | checks: no call outside its query, files read, settings with output |

Larger outputs (calls/, decoy_pairs/, instance_calls.parquet) stay on Sherlock in
`results-pilot/scores-partial/`.

`unique_finds_tune_instances.csv` (one row per setting and instance) and
`unique_finds_tune_by_instance.csv` (one row per instance) list the instances behind
`unique_finds_tune.csv`; made by `unique_finds_tune_instances.py` with the scorer's own
functions, and their counts match `unique_finds_tune.csv` for every setting. There are 3:
LysM 79-119 in A0A2K6L9H6 (35 settings; MMseqs2 calls it at E 1.1e-4, below its cutoff of
E 1.5e-22) and LRR_1 64-86 and 88-110 in A0A078HBQ0 (4 and 11 settings; the mmseqs12 k9 extend
calls are E 1.6 and 17, against that setting's cutoff of E 18). MMseqs2 and phmmer do
transfer LRR_1 to these repeats (50-201 calls each, best hit P47735 at E 2e-59), but every
one of their calls is 47 aa or longer, so under half of it lies inside a 23-aa repeat
(best landing 0.22-0.49) and the landing rule (>= 0.5) does not count it.
`lrr1_calls_residues.html` draws the two kmerseek calls residue by residue.
