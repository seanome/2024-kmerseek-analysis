# Whole-library oyster analysis

The primary analysis uses every original paired record from all six experimental libraries. The first-million-pair results are a deterministic prefix pilot, not a full-depth paper comparison. These pilot results remain separate.

On 2026-10-23, all six full-library trimming completion records report 153,671,701 input pairs and 153,628,178 retained pairs. Each retained count independently agrees with the completed full-depth baseline aggregation record. The compact evidence is `full-depth-status.json`; original records are under `$SCRATCH/oyster-mvp-20261023/trimmed/full` and `baseline/full/aggregation.json`. No new full-file recount was performed for this check.

Full-depth DE-kupl job 49203559 was running when checked. Full-depth SPLASH 49203558 failed with explicit OOM events: six KMC processes each received a 96 GB limit inside a 128 GB job. A killed process recorded about 24 GB RSS; aggregate concurrent memory caused the failure. Replacement 49214375 requests 32 CPUs, 256 GB and 12 hours, with only two KMC processes at once and 32 GB per process. Twelve hours is a conservative allowance for the subsequent stages, which have no successful full-depth timing measurement. The failed stage lasted 72 seconds. Output uses `splash/full-memory-corrected`, preserving the failed run.

Every full-depth SPLASH anchor with adjusted P strictly below 0.05 must be extended, with no top-N cap. Full-depth selection, completion records and searches must be recorded separately from the 213,658-anchor prefix selection. All recovered full-depth sequences must enter downstream public searches when available. No full-depth SPLASH or extension completion is claimed.

## Completion and all-significant extension, 2026-10-23

SPLASH job 49214375 completed with exit 0 after 16:09; batch peak RSS was 71,641,740KiB. Its input manifest uses the complete trimmed read 1 of each library, rather than pooled mates as in the earlier prefix experiment. Comparisons between those runs therefore differ in read-end usage as well as depth. The paired-file counts above describe the input libraries, not the number of mates tested by SPLASH.

Two independent streaming passes counted 5,538,699 unique anchors with corrected P strictly below 0.05; the complete corrected-score file SHA256 is 4312b75c216d9f62767c2457b531e80d0bbb1343a013634c44f850751e8e5f9e. All are selected for Compactors in 554 disjoint batches of 10,000 (final batch smaller), array 49221554, concurrency 4. Extension reads include both complete trimmed mates for all six libraries. No top-N cap is applied. `full-depth-extension-selection.json` records the selection, source hash, read-end distinction and submitted job.

Each extension task requests 8CPUs,64GB and 4hours. The original prefix tasks used 8CPUs/32GB/2hours; full-depth read input is about 25times larger. The larger allowance is a conservative model, not a measured full-batch runtime. Selection now streams score rows instead of retaining every wide score row in memory. Strict 0.05 exclusion, inclusion below 0.05, deterministic order and NaN rejection were tested locally. Compactors completion and resulting sequence counts are not yet claimed.

## Verified partial extensions and automatic searches

Nine completed batches cover 90,000 selected anchors. Independent audit 49241735 checked output hashes, valid sequence/anchor membership, recomputed counts and each batch's exact slice of the frozen sorted selection:87,870 anchors have extended DNA, producing 192,124 records. The other batches remain pending or running. `results/full-depth-extension-progress.json` records this verified snapshot.

Full-depth DE-kupl 49203559 completed exit 0 in 3:05:46 with 71,361,004 KiB peak RSS; biological comparison remains pending. Full-depth public-search array 49241935 covers batches 0-31 with per-task `aftercorr:49221554` dependencies, so each search starts after its own extension succeeds. It uses all four encodings, scaled 10/100 and explicit threshold 0 against the eight untrimmed public-pilot indexes. Query IDs use `splash:full`, keeping them distinct from prefix queries. Remaining batches 32-553 must be submitted as scheduler slots free: the attempt to submit all 554 search tasks exceeded the normal QOS limit of 2000 submitted jobs per user. No healthy work was cancelled to make room. These full-depth results, and future searches against newly indexed archive runs, remain separate from the completed prefix-query comparison.
