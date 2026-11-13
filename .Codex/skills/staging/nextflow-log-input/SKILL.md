---
name: nextflow-log-input
description: Diagnoses Nextflow input names that resolve to built-in objects. Use when an emitted command contains Logger[nextflow.Nextflow] instead of a file path, especially for a process input named log.
---

# Nextflow input named log resolves to the logger

Observed on Nextflow 25.04.7 in oyster migration job 49520265, 2026-10-25.
The Python audit failed because its command received `Logger[nextflow.Nextflow]`
in place of `search.log`. The declared process input was `path(log)` and the
script interpolated `${log}`. Upstream indexing and searching had succeeded.

1. Read the failed task's `.command.sh` and `.command.err`.
2. Confirm the file exists among staged inputs and that the command contains the
   logger string. Do not infer a missing search result from this error.
3. Rename the process input and script interpolation together, for example
   `path(search_log)` and `${search_log}`.
4. Keep the failed work directory and logs. Test the actual generated command
   under the pinned Nextflow version, then test resume and query changes.
5. Python unit tests do not exercise Nextflow's name resolution. Require the
   real workflow test before calling this fixed.

Project evidence: `analysis/oyster-shazam/results/migration-20261025/STATUS.md`.
This is a first-use staged skill. Recovery job 49521583 verified the repair with
Nextflow 25.04.7, including cached resume, query changes and controlled stop/resume.
