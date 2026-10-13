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
