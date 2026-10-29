# Pipeline location and production restart, 2026-10-27

The actual DSL2 implementation now lives in `nextflow-runs/botryllus-shazam`,
following the directory and Makefile pattern in PR 197. This is a directory in
2024-kmerseek-analysis, not a separate repository. It provides run, resume, test,
watch and stop targets. Helpers remain under `scripts/botryllus_shazam`.

The old pipeline path is a compatibility symlink. No historical report, trimmed
read, output or Sherlock cache was removed. The two unfinished helper edits and
untracked exercise-sources files remain uncommitted and were not deployed.
Runtime code is an immutable archive of commit d20ae2b74ad8d1f83a5634d705801eed99e5f115.
The concurrently pushed sample-ID whitespace review fix was merged and preserved.

## Actual blocker

Controller 49525592 ran for 23 seconds and failed 1:0 with
`Frozen sample metadata changed`. The Nextflow log reports zero submitted tasks;
its empty failure count is not a successful run. The original directory remains
`$SCRATCH/botryllus-shazam-PEC-nextflow-20261025`.

The frozen design records SHA-256
`c97b989a5c8a7777a0cbe54a4c8fab53b4486783f467488bb6af3fadfe9021df`.
Git's LF table hashes to
`da3bc356346cd63f08beebd40cf3da1a70009c04055d3d556a89d630ef392ad7`.
Restoring CRLF exactly reproduces the frozen hash. Preparation writes those exact
bytes to a new launch input; it never changes the expected hash or sample labels.
A changed treatment label is rejected, and existing destinations cannot be overwritten.

## Validation and submission

Local checks passed: two frozen-table tests, ten workflow helper tests, one pooling
test, Black, shell syntax, Makefile dry runs and git whitespace checks. The native
Jellyfish test cannot run locally because it needs Sherlock's GROUP_HOME tools;
the Sherlock helper-test phase has now passed, including the native counter and sketch tests. The full real-tool fixture is still running. The relocation kept process
scripts unchanged; the merged review changes only sample-ID parsing.

- Fixture: 49811516; last observed running at 1m08s; relocated main.nf loaded and tasks submitted.
- Production: 49811553; last observed pending Dependency on successful fixture.
- Production directory: `$SCRATCH/botryllus-shazam-PEC-nextflow-20261027`.
- Source: `$HOME/botryllus-shazam-code/production-d20ae2b7`.
- Reused successful preflight: 49525580; reference checks: 49521127.

Production checks the fixture verification marker and remeasures resources before
starting. A fresh resource report is retained for each controller, including on
resume. The head requests 2 CPUs, 16 GB and seven days on hns. Largest-library count
requests 91 GB/12 h from the 60.39 GiB observed allocation peak; other libraries
scale to 32–91 GB. Gene models get 32 GB/4 h; k-mer models get 32 GB/96 h.
Full-scale model timing and other unmeasured stages remain provisional.

## First Figure 4 comparisons

Published-count redraws already exist under `reports/botryllus-shazam/figure4`.
Independent read-based comparisons require all gene counts and the three young
cohort gene models. They can finish before k-mer pooling and sequence searches.
The previous 2–6 hour estimate begins only after production starts and excludes
child-job queue delays; it is not a measured full-run duration or a promised time.
No new read-based Figure 4 result is claimed by this submission.
