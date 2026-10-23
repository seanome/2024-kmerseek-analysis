# Restart checkpoint: 2026-10-23

This checkpoint does not claim full-depth completion or unattended monitoring.
Read current Sherlock instructions in `/etc/agents/AGENTS.md` and relevant topic
files before continuing. Use the existing `sherlock` SSH alias. Authentication
worked in this session. The `sherlock-dtn` connection rejected authentication;
transfers through the authenticated login alias succeeded.

## Locations

- Local worktree: `/Users/olga/.codex/worktrees/botryllus-shazam-full-depth/2024-kmerseek-analysis`.
- Local preserved pilot: `/Users/olga/code/botryllus-shazam-inputs/pilot`.
- Original inputs remain in `/Users/olga/Downloads`.
- Sherlock project: `$SCRATCH/botryllus-shazam-PEC-20261023`.
- Durable code: `$HOME/botryllus-shazam-code`.
- Preserved pilot: project `checkpoints/pilot`.
- Original proteome: project `references/botryllus_new_genome_protein.fasta`.
- Completed pilot audit: project `audit/pilot-audit.json`.
- Raw runs: project `raw/<SRR accession>`; `validated.json` gates reuse.
- Logs: project `logs`.
- Existing read-only environment: `$GROUP_HOME/opt/oyster-shazam-20261023`.
  It contains sourmash 4.9.4 and Biopython 1.88. Do not modify it in place.
- Existing Pfam database: `$SCRATCH/data/pfam/Pfam-A.hmm`.
- Existing eggNOG 5.0.2 resources: `$SCRATCH/data/eggnog/emapperdb-5.0.2`.
  The database's presence does not establish a working mapper environment.

## Jobs and recovery

| Job | Work | Last inspected state |
|---|---|---|
| 49188713 tasks 1–7 | Preserved full-depth download batches | Tasks 1–3 running; 4–7 queued |
| 49188713 task 0 | Initial download batch | Failed on one corrupted downloaded mate |
| 49192201 | Batch-0 recovery, reuse validated runs | Running |
| 49189184 | Initial audit and annotation attempt | Audit and 500-protein pilot passed; failed on dot in input |
| 49191037 | Separate masked search copy and annotation | Running |
| 49189785 | Initial archive inventory | Tunicate complete; chordate parser failed |
| 49191136 | Chordate-only inventory after parser fix | Running |
| 49192465 | Four-encoding representative Sourmash pilot | Running |
| 49191258 | Full-depth nf-core preprocessing | Pending successful downloads |

Use current scheduler state before acting. Do not cancel healthy or unrelated jobs.
The array master ID is not an active allocation. Task 0's actual job ID was
49189113; it has ended. Task 1's actual job ID was 49189114 at the later audit.
Bounded audit and small indexing checks used an unused CPU within that allocation.

Queued download tasks 4–7 were extended from 8 to 24 hours after measuring slower
transfers. Task 0 failed on SRR32233934 mate 2: correct byte count, wrong MD5,
and gzip CRC/length errors. Its bytes remain in
`raw/SRR32233934/SRR32233934_2.failed-md5-1dcb77331c5d52cc6623379ae9f3ea0d.fastq.gz`.
The recovery code is isolated under `$HOME/botryllus-shazam-code/recovery-20261023`.
Do not remove validated files or reuse the quarantined mate.

Preprocessing depends on tasks 1–7 and 49192201, replacing the failed master-array
dependency. No failed download can become an accepted discovery input.
The initial downloader copy remains beneath the active original tasks; the recovery
copy handles a completed partial download without issuing an invalid resume request.

The first annotation attempt's successful database build and 500-protein pilot were
reused. Search-copy masking is explicit in
`annotation-maskX-20261023/masked-positions.tsv`; it is not a reference correction.
Original annotation outputs and logs remain in `annotation`.

Tunicate inventory: `archive-inventory-20261023/tunicate-rna-seq.tsv.gz` and summary.
Chordate retry: `chordate-inventory-attempt2-20261023`.
The parser treats quotation marks as literal TSV content. The failing raw metadata
and partial database remain in the initial inventory directory.

The dev partition rejected a batch partition change because it permits interactive
sessions only. No jobs were moved there. The successful earlier oyster preprocessing
used hns; new bounded recovery jobs use the authorized hns/normal partitions.

## Completed checks

The independent pilot audit reconciled the discovery population, selected subset,
parent/child records, translations, all 3,984 real Botryllus sketch hits, and every
proteome sequence. It did not rerun discovery or reconstruct absent pilot FASTQs.
The public workbook and uploaded proteome match recorded pilot checksums.
SRR32233927 has a completed raw-read marker: 692 paired records and 207,600 bases.
This very shallow library is a real input, not a representative performance pilot.
SRR32233926 is also validated: 4,418,415 pairs and 1,325,524,500 bases.
The small SRR32233927 indexing check validated 12 signatures and 12 searchable indexes.

## Next steps

1. Read current logs and completion markers. Wait at least 60 seconds between scheduler
   checks. Account for actual array tasks, not just compressed queue rows.
2. Finish metadata design review using Dataset S1 and paper text. Resolve collection
   relationships, late time labels, stage differences, and the single old genotype.
   No condition-only DE-kupl treatment test is approved by this checkpoint.
3. Recover complete validated reads, then run documented nf-core/rnaseq 3.27.0
   preprocessing under Slurm. Preserve pipeline trace and tool versions.
   Use identical completed trimmed files for SPLASH and DE-kupl.
4. Run pinned upstream SPLASH/Compactors on every anchor with corrected P < 0.05
   (no top-300 cap), supported DE-kupl counting, and only valid
   inferential contrasts. Implement SPLASH+ only with consistent genome resources.
5. Inspect the new standard annotation results. Verify exact reference snapshots and
   retain no-hit proteins. Add orthology only after checking its environment.
6. Freeze specific deep-review IDs under PROTOCOL.md. Produce separate observations,
   predictions, hypotheses, controls, blinded cards, and proposed model corrections.
7. Finish tunicate metadata inventory. Measure a representative four-encoding indexing
   pilot before a bounded archive launch. Chordates are inventory only.
   Do not stage the whole archive or run SPLASH/DE-kupl across it.
8. Build the interactive evidence explorer with the original report preserved.
   Pending stages must remain pending. Export JSON/TSV/FASTA and resource records.

Scratch is purged after 90 days without content modification. Preserve important
derived results on durable authorized storage; never refresh timestamps to evade purge.
