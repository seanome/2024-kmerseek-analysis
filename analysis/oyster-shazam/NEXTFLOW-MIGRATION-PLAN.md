# Preserve oyster results while moving public searches to Nextflow

## Latest tested status, after access restoration on 2026-10-25

Olga confirmed renewed Sherlock access. This section supersedes older access and
pending-test statements below; those remain a historical record. Recovery job
49521583 (source 8d7efccb) completed with exit 0:0. Independent historical audits,
the corrected >=30-aa stop-only X-preserving fixtures and 2,000-query pilot,
and all seven bounded Nextflow migration scenarios passed. See
[the evidence and remaining limits](results/migration-20261025/STATUS.md) and
[the implementation README](nextflow/README.md).

The controlled application-stop test does not test scheduler interruption or Slurm
child-task dispatch. Production rollout, full coverage and report integration
remain pending. Existing worker outputs and dirty recovery/report files were
preserved. Historical stop/X results retain their historical labels.


Written before implementation, 2026-10-25, at Olga's request.

## Objective

Replace the sequential public-archive workers with a Nextflow DSL2 workflow using Slurm tasks for download, trimming, Branchwater sketching, indexing, searches, and validation. Preserve completed results and useful running work. The primary queries are full-depth SPLASH extensions and all 363,635 significant DE-kupl contigs. Salmon remains the gene-expression reference path; it is not part of public-reference preparation.

## Invariants

- Never overwrite or delete an existing result during migration.
- File existence is not validation. Reuse requires recorded source checksums, exact settings, expected output structure, and successful validation.
- Keep historical stop-inclusive results separate from the primary stop-free protocol. Keep stretches >=30 amino acids from all six frames; split ONLY at stop codons (*), preserving internal X. Do not filter queries by reference-proteome support.
- DNA reuse requires identical input preprocessing, k, scale, seed, and abundance settings. Translated historical sketches cannot stand in for stop-free sketches.
- Query membership, empty sketches, and all unmatched query IDs remain explicit. No caps or silently dropped collections.
- Only one writer may create a given artifact. Old workers and Nextflow never own the same run concurrently.
- Heavy validation, hashing, sketching, and tests run on Sherlock compute nodes.

## Phase 1: Freeze ownership and inventory

1. Record live array tasks, current public run, completed checkpoints, source script hashes, resource allocations, and logs. Record the three specifically discussed tasks: 49241661_778, _275, and _228.
2. Inventory existing per-run validated.json records and files in a bounded Slurm audit. Also inventory the completed linear DNA replay and stop-free preflight; validate their outputs before counting matches.
3. Write an immutable adoption manifest with artifact paths, checksums, protocol IDs, run IDs, query-set IDs, settings, validation scope, and ownership. Mark incomplete or incompatible outputs explicitly.
4. Preserve historical directories. Do not move live node-local files or cancel healthy current work to populate the manifest.

Acceptance: every adopted output passes a content check; corrupted, missing, and wrong-protocol examples fail. No production jobs change in this phase.

## Phase 2: Define explicit reusable artifacts

Use explicit workflow input channels populated from the validated manifest. Use Nextflow -resume for tasks created by this workflow. Keep .nextflow/cache and the work directory while they are needed. Do not depend on publishDir as an input or cache. Do not use storeDir for the initial migration: existence-only checks cannot establish protocol compatibility, and concurrent writers can race.

Artifact keys include stage, run or query-collection ID, source checksum, preprocessing settings, alphabet, k, scale, seed, abundance policy, stop policy, and relevant tool/code versions. Search keys also include query and reference checksums. Resource requests are recorded separately from scientific identity.

| Process | Input | Transformation | Required output and reuse test |
|---|---|---|---|
| DOWNLOAD | ENA URLs and expected MD5s | Resumable download, checksum verification | FASTQ files and source manifest; MD5 matches |
| TRIM | Verified FASTQs | Pinned Trim Galore/Cutadapt Q20, minimum length20; preserve existing independent-file trimming semantics | Trimmed FASTQs, reports, settings/checksums |
| TRANSLATE | Trimmed FASTQs or query DNA | Stream >=30aa stretches from six frames; split only at *, preserve X and origin IDs | Peptide file plus ID/membership manifest; independent exact-word test |
| SKETCH | Trimmed DNA or translated peptides | Branchwater manysketch, exact encoding/k/scale; preserve per-query identity | Sketch collection, settings, source hash, count and empty-ID manifest |
| INDEX | Validated reference sketches | Branchwater index with allocated threads | Complete searchable index plus source/settings manifest |
| SEARCH | Validated query collection and compatible index | Branchwater manysearch, explicit threshold0 and all-overlap output | Compressed matches plus expected-query/DONE-count validation |
| COMPARE | Matched scaled10 and100 results | Exact pair-set difference | Lost pairs and reasons; no unexpected gained pairs |
| AUDIT | Outputs and manifests | Check hashes, IDs, counts, settings, scope | Atomic completion record; only then report coverage |

The first migration milestone uses an existing compatible DNA100 sketch. New translated production builds remain gated on Branchwater peptide aggregation, exact hash validation, and tests that internal X is preserved and only * splits stretches.

## Phase 3: Minimal real-input workflow and tests

1. Pin and record the installed Sherlock Nextflow and Branchwater versions; resolve the actual Slurm profile before launch.
2. Build one-run DNA100 search path that adopts an existing reference sketch and frozen full-depth query collection. Do not redownload that run.
3. Verify each adoption hash on compute. Index and search via Branchwater; validate every result ID and expected query count.
4. Recompute match pairs independently and compare with the existing first-shard replay for the same run and queries.
5. Run again with -resume: eligible tasks must be cached, and outputs must retain the same checksums.
6. Interrupt a bounded test after a completed upstream task, then resume. Completed upstream tasks must not rerun; missing downstream tasks must run.
7. Change only the query collection: reference preparation must remain reusable; search identity must change.
8. Inject a truncated sketch and wrong stop-policy manifest: adoption must fail before search.
9. Test real Branchwater manysketch on empty and nonempty inputs, duplicate sketch hashes, and split peptide records sharing one query ID. Compare exact hashes and query counts against the independent correctness implementation.

Acceptance: saved test evidence includes source SHA, commands, input hashes, trace/cache status, outputs, independent pair comparisons, and failures deliberately tested. A successful Nextflow driver alone is insufficient.

## Phase 4: Drain the old workers safely

Only after Phase 3 passes:

1. Add a per-worker drain request checked between public runs. Test on an isolated worker fixture; do not assume changing a Python file changes an already-running interpreter.
2. For already-running workers, determine whether a safe checkpoint signal is supported. If not, let them finish their allocation rather than kill healthy work mid-run. Held or cancelled pending tasks must be explicitly recorded with their assigned runs.
3. Reserve active old-worker runs in the ownership manifest. Nextflow receives only unowned runs. Reconcile each final worker checkpoint before releasing its remaining assignments.
4. Transfer responsibility in a small bounded batch first. No blanket cancellation, duplicate submissions, or concurrent writes to legacy paths.

## Phase 5: Full workflow and controlled rollout

Add DOWNLOAD, TRIM, TRANSLATE, and SKETCH paths for missing artifacts. Retain trimmed reads through protocol validation and agreed retention, so changing search settings does not trigger another download. Estimate storage from measured bytes before retaining the whole archive. Adopt completed DNA and validated stop-free artifacts wherever compatible.

Choose CPU, memory, runtime, and concurrency from measured stage histories and pilot traces. The existing 8CPU/128GB workers are not a resource template. Thread Branchwater explicitly; bound downloads, translations, and dense sketches separately. Run task I/O in node-local scratch when appropriate, with completed artifacts copied to stable project paths. No long-term retention claim for purged Scratch.

Enable Nextflow trace, report, timeline, and DAG; retain scheduler IDs, tool versions, exit status and command logs. Record wall time, user/system CPU and peak RSS for major commands; missing measurements are unavailable, not zero. Avoid adding parent and child CPU measurements together. Imported historical outputs have historical timing gaps, not fabricated Nextflow traces.

Search the full validated SPLASH snapshot and all DE-kupl collections against public runs; extend to later audited SPLASH batches and newly completed references through new immutable snapshots. Add Swiss-Prot scaled5 through a separate compatible protein-reference channel. Publish verified partial results with exact run/query/protocol denominators.

## Rollback

Stop only new Nextflow submissions if validation fails. Preserve work/cache, logs, old results, and new partial outputs. Diagnose the failure before resubmission. Historical results remain readable throughout; rollback does not restart old workers or mix protocols automatically.

## Completion criteria

- Validated existing artifacts are reused without recomputing compatible upstream stages.
- First-run, resume, interrupted-run, query-change, corruption, and protocol-mismatch tests pass on Sherlock.
- New sketches use Branchwater manysketch; retrieval uses Branchwater index/manysearch.
- Old-worker ownership is reconciled without losing completed runs or submitting duplicate work.
- Full-depth SPLASH and DE-kupl public/Swiss-Prot coverage is recorded separately from historical prefix coverage.
- Sources, tests, resource evidence, and migration status are committed on PR202; the report describes measured progress only.

## Initial status

Plan committed before implementation. Production ownership transfer and pending-task holds have not occurred. The earlier linear replay and stop-free preflight completed according to Slurm, but independent result audits and report integration remain outstanding. Their completion must not be mistaken for archive-wide coverage.

## Protocol correction, 2026-10-25

Olga corrected an unrequested agent assumption: X is not a split point. The rule is >=30aa stretches from all six frames, split only at *, preserve internal X. Earlier stop/X and min20 pilots are historical and must not be called measurements of this corrected rule. Test lengths29/30/31 and internalX before production changes.

## Local implementation continuation, 2026-10-25

The first one-run DNA adoption/index/search/audit workflow is now implemented in
[nextflow/](nextflow/README.md). Its Python validation helpers pass local fixture
tests; the Nextflow driver has not been parsed or executed. Corrected translation
is implemented separately and passes the 29/30/31-aa, stop-only, X-preserving,
six-frame, coordinate and original-membership tests. The corrected Branchwater
pilot builder is written, but pinned-tool execution remains pending.

Historical recovery scripts and outputs are preserved. No Sherlock connection,
submission, cancellation, ownership transfer or production protocol migration
was attempted. Access confirmation, independent historical audits, pinned X
hashing, all Phase 3 runtime gates and the remaining workflow phases are pending.
