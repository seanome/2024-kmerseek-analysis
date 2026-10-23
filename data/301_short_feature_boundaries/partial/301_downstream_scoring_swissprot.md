# Downstream scoring, 2026-10-23

Scoring job 49182992 runs with Slurm afterany on recovery jobs 49182599–49182614.
Submission uses scripts/301_score_after_recovery.sbatch at commit cba19f50.
No search jobs were added by this step.

The scorer was rerun on the committed three-setting snapshot; its summary matches the
committed table byte for byte. Fifteen notebook-301 tests pass.

Current outputs: 301_partial_summary_swissprot.tsv (per-setting and per-type scores),
301_partial_features_swissprot.tsv (per-feature calls), and raw/ (input hits and manifest).

The dependent job writes to:
$SCRATCH/301-short-feature-boundaries/scored-recovery-49182992/

Expected files:
- 301_partial_summary_swissprot.tsv
- 301_partial_features_swissprot.tsv
- 301_settings_coverage_swissprot.tsv

All 16 recovery jobs were pending at the latest check in this turn. The dependent job
has not run. Failed or missing settings remain incomplete. Comparators, full timing,
remaining figures and the independent second-agent check remain unfinished.

Latest check: 2026-10-23 13:06:06 UTC. All sixteen recovery jobs and scorer 49182992 are still pending. No new completed settings or scoring files exist. Current coverage is in 301_current_coverage_swissprot.tsv.
