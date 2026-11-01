---
name: nb357-alignment-memory
description: Repair or resume notebook 357 protein SPLASH alignment preparation after a memory kill, while retaining completed Nextflow stages.
---

Added 2026-10-22 after the full run's alignInput was killed at 16, 32 and 48 GB.

- Inspect the failed task's staged inputs, not only their file count. This run staged
  512 occurrence files (162 GB) and 512 protein FASTAs (1.29 TB); the completed
  dedupProteins output was only 8.2 GB. Reuse that channel for sequence lookup.
- Read one occurrence hash bucket at a time, select columns before collecting,
  and retain only eligible anchors. Each (setting, gap, anchor) belongs to one
  bucket. Reject an anchor repeated across buckets instead of overwriting its FASTA.
- Pfam domains, protein sequences and joined rows can also exceed RAM. The alignment
  preparer uses a temporary SQLite database on node-local disk and checks free space.
  It discards joined rows after each bucket. Keep unsigned anchor IDs as text in SQLite.
- The completed pfamOverlap task calls clade_splash_pfam.py. Changing that file can
  invalidate its Nextflow cache. Alignment preparation has its own executable,
  clade_splash_align_input.py, so its fix need not repeat Pfam mapping.
- Before resuming, compare manifests, parts and FASTA contents against the original
  on a small fixture. Cover empty inputs, repeated domains and occurrences, unsigned
  IDs, duplicate protein FASTAs, and targets outside domain envelopes.
- Run make measure-resources before the resume; make run includes it. Generated
  sizing files live in run/measured, not tracked configuration. Keep the same launch
  directory and -resume session. Confirm upstream tasks appear as CACHED.

- Nextflow's Apptainer wrapper clears most host environment variables. Selecting a
  directory from L_SCRATCH_JOB inside Python alone is insufficient. The Sherlock
  alignInput selector uses scratch '$L_SCRATCH_JOB' and binds that directory into
  Apptainer. A Nextflow probe inside allocation 48999084 confirmed the resulting
  working directory was node-local. Inspect the generated .command.run after launch.

- Added 2026-10-23: CANCELLED is not evidence of a pipeline crash. Jobs 49000133
  and 49000241 were stopped by an explicit request in another chat; its transcript
  contained the exact scancel command and Slurm named the same UID. Check both
  accounting and chat history before diagnosing a cancellation. A new explicit
  resume request permits restarting; a status message alone does not.
- Size from the full run, not only its one-file check. That full task's Slurm peak
  was 93.28 GiB and completed 11.12% of occurrence bytes in 2h25m37s. The relaunch
  floor is 128 GiB/36 h. The earlier one-file 28.07 GiB check was not a full-run peak.

- Added 2026-10-23 after the same experiment was stopped twice: when Olga cancels
  because the experiment asks the wrong question, establish the intended biological
  question before treating a generic “fix and resume” request as approval of this
  design. Showing the workflow after relaunch was too late. This design is stopped.
