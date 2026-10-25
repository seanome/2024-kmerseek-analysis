# Oyster migration evidence, 2026-10-25

Job 49520265 ran source 49d75dcb on Sherlock and failed after 6 min 6 sec in the
Nextflow AUDIT process. Its upstream checks completed. Peak batch RSS was
1,426,444 KiB. Existing outputs and workers were not modified.

| Check | Result | Scope |
|---|---|---|
| Corrected translation fixtures | Passed | Lengths 29/30/31, stop-only boundaries, internal X, all six frames, coordinates and original IDs |
| Pinned Branchwater 0.9.15 X handling | Passed | In the 121-aa fixture, all 7 protein-k7, 12 Dayhoff-k12 and 30 HP-k30 X-spanning hashes were retained at scaled1 |
| Empty and duplicate-hash IDs | Passed | Explicit empty original and two nonempty originals with identical hashes remained separate |
| Corrected real-query sample | Passed hash checks | 2,000 queries, 4,659 segment/alphabet comparisons; no X-bearing retained segment in this real sample |
| Initial DNA shard | Passed independent audit | 380,867 original queries, 38 collections, eight public runs, 36 query–run pairs; original DNA hashes and exact pair/intersection equality |
| Historical preflight saved results | Passed independent audit | All four alphabets at scales10/100, complete query identities and exact pair/loss sets; retains historical stop/X protocol |
| First Nextflow execution | Failed in AUDIT | `log` resolved to `Logger[nextflow.Nextflow]` instead of the input search log; adoption, indexing and search completed |
| Production adoption | Blocked | Migration gates pending; original build-code and Sourmash-version provenance unrecorded for historical inputs |

The reserved-name collision is fixed in source by using `search_log`. Recovery
will reuse the saved audit manifests and corrected sketches in a new output
folder. It also independently reconstructs historical hashes one word at a time
and measures corrected matches against the same small public run used previously.
These recovery results are pending at this commit.

Evidence files are under `job49520265/summaries/`. The complete work/cache and
partial-output archive remains at
`$GROUP_SCRATCH/oyster-migration-tests-20261025-49d75dcb/evidence.tar.gz`.
This archive preserves evidence; paths to node-local files do not provide a
portable cross-job Nextflow resume cache. Within-allocation resume tests remain
required, and a real scheduler-interruption test is still pending.
