# Fixture passed; production resumed after Compactors OOM

Checked 2026-10-27 at 13:38 PDT. Fixture 49811516 completed 0:0 in 7m18s,
with 128 task rows and the independent verification marker. It checked 29,459
canonical k-mers, six model corrections and 100 retained gene-evidence rows.

Production 49811553 started at 10:12 PDT and failed after 2h17m. Its trace contains
244 completed tasks, one failed task and one aborted task. Completed work includes
64 pair validations, 64 counts, 64 quantifications, 38 target preparations, the
combined k-mer matrix, SPLASH, and all three gene models. Gene models completed
around 11:16 PDT. Their first comparisons are now saved in
[the read-based comparison report](../../figure4/read-based-20261027/README.md).

## Diagnosis and recovery

Compactors task 246, Slurm job 49830468, exceeded its 96 GB request. Slurm records
OUT_OF_MEMORY, batch MaxRSS 100662776K and 7m53s runtime. The command log records
`Detected 1 oom_kill event in StepId=49830468.batch`. Python surfaced SIGKILL as exit
1; this was not a generic tool error. Nextflow then cancelled k-mer pooling job
49827024, which used about 1.04 GiB; that task did not independently run out of memory.

The reviewed failure is tied to the exact trace SHA-256 and task ID. The resource
preparer still rejects unreviewed failures, stale reviews, and reviews without the
original OOM log. It now permits this diagnosed failure with a 192 GB/48 h Compactors
request. Regression tests verify that unrelated exit-1 failures remain blocked.
The controller peak was 16776552K, close to its 16 GB request; the next request is
24 GB with a 16 GB Java heap. Other resource requests incorporate completed traces.

Replacement controller 49841010 is RUNNING. It uses launcher revision e90c540c,
but `WORKFLOW_CODE` pins the original immutable d20ae2b7 pipeline and helpers.
It resumes the same launch and work directory with `-resume`. Its log already
reports cached validation, counting, quantification, SPLASH and all three gene models; k-mer pooling has been resubmitted. No output/cache
cleanup, helper changes or workflow-process edits were used for the repair.

The 192 GB request is twice the killed allocation; a successful full-scale
Compactors peak is not yet measured. The recovery is running, not complete.

## First comparisons

They are ready now: the preliminary 2-hour and 24-hour gene-effect comparisons,
three gene-model tables, independent multiple-testing checks and a figure are saved
under `reports/botryllus-shazam/figure4/read-based-20261027`. Remaining sequence
analysis does not block these tables. Full Figure 4 reproduction still needs the
unresolved reference, model, later-time grouping and panel-score differences settled;
a full-panel completion time cannot yet be supported by the traces.
