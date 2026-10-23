# Whole-library oyster analysis

The primary analysis uses every original paired record from all six experimental libraries. The first-million-pair results are a deterministic prefix pilot, not a full-depth paper comparison. These pilot results remain separate.

On 2026-10-23, all six full-library trimming completion records report 153,671,701 input pairs and 153,628,178 retained pairs. Each retained count independently agrees with the completed full-depth baseline aggregation record. The compact evidence is `full-depth-status.json`; original records are under `$SCRATCH/oyster-mvp-20261023/trimmed/full` and `baseline/full/aggregation.json`. No new full-file recount was performed for this check.

Full-depth DE-kupl job 49203559 was running when checked. Full-depth SPLASH 49203558 failed with explicit OOM events: six KMC processes each received a 96 GB limit inside a 128 GB job. A killed process recorded about 24 GB RSS; aggregate concurrent memory caused the failure. Replacement 49214375 requests 32 CPUs, 256 GB and 12 hours, with only two KMC processes at once and 32 GB per process. Twelve hours is a conservative allowance for the subsequent stages, which have no successful full-depth timing measurement. The failed stage lasted 72 seconds. Output uses `splash/full-memory-corrected`, preserving the failed run.

Every full-depth SPLASH anchor with adjusted P strictly below 0.05 must be extended, with no top-N cap. Full-depth selection, completion records and searches must be recorded separately from the 213,658-anchor prefix selection. All recovered full-depth sequences must enter downstream public searches when available. No full-depth SPLASH or extension completion is claimed.
