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
| 49191136 | Chordate-only inventory after parser fix | Completed |
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

The chordate inventory completed at 6,311,627 runs and 13.15 PB of listed FASTQs.
Its compressed metadata and SQLite identities are on Sherlock in the retry folder;
only the compact summary is committed. Chordate indexing remains unlaunched.
The representative Sourmash pilot had completed its DNA scaled-5 sketch in
5:18.83 at 2,522,944 KB peak RSS when checked; other encodings were still running.
The optional DIAMOND full search finished; Pfam was still running. These incomplete
annotation outputs have not been promoted to biological candidate claims.

## Continuation checkpoint: 2026-10-23, 16:24 UTC

The private explorer is published at
https://botryllus-shazam-evidence.brawnyspoon2048.chatgpt.site .
Site ID: `appgprj_6adb8461f2a08191a6bcce0594b60661`.
The Site source checkout is outside the analysis worktree at
`/Users/olga/.codex/visualizations/2026/10/23/01a168f0-7460-76a3-bb7b-e25675bb1a63/botryllus-site`.
Tracked page source is `sites/botryllus-shazam`; generated data are ignored there.
The published Site is owner-private and the hosted browser requires sign-in.
Local browser tests passed peptide/protein lookup, zero-match handling, and file tables.
Original report and protein FASTA bytes remain unchanged in the private deployment.

The 16:19 UTC snapshot had 36/64 validated raw runs. Current status is in `status.json`.
Batch 0 recovered successfully (49192201); transfer-interrupted batches 3, 6 and 7
were resumed as array 49205641 using a separate code folder. Task 3 completed.
Tasks 1, 4 and 5 of the original download array were kept running.
The retry wrapper handles curl 18/28/52/55/56, with bounded attempts. Native curl
is old and lacks `--retry-all-errors`; it was not given that unsupported flag.

The old preprocessing job was cancelled by the failed dependency.
Replacement 49205669 waits for surviving original tasks 1/4/5 and array 49205641.
Its own 64-run marker gate also checks completed inputs; completed older job IDs
were removed from the dependency because Slurm had purged their live records.
Full-depth SPLASH/Compactors job 49206899 waits for 49205669. It selects from the
complete all-anchors table with strict adjusted P < 0.05, no top-N cap, and exports
DNA FASTA, peptide FASTA and membership TSV. Resources: 16 CPUs, 96 GB, 48 h.
Its all-anchor Compactors memory/runtime are model estimates, not measured success.
Discovery script and selector are isolated in `full-discovery-20261023` under HOME code.

Representative four-encoding indexing job 49192465 completed in 1:53:13,
peak RSS 11,448,104 KB. All 12 signatures, 12 SBT indexes and self-searches passed.
Its complete marker is committed as `indexing-pilot-validated.json`.
DIAMOND/Pfam annotation job 49191037 completed; candidate interpretations are unaudited.

Primary search pilot 49206896 uses actual Sourmash 4.9.4 / Branchwater 0.9.15 Rust
manysketch and multisearch, all 987 historical pilot peptides, and full Botryllus
and Swiss-Prot references, in three encodings and scales 5/10/100. It has 8 CPUs,
32 GB, 4 h. Separate canonical record maps and masked-position tables preserve original
IDs and sequences. `execution.complete.json` means execution only: independent
comparison to `sourmash search` and hash arithmetic is required before promotion.
Do not queue full-depth protein retrieval until this native bulk-search pilot passes.
No full-depth protein-result or biological candidate claim is complete.

Olga authorized messaging the oyster chat to make a matching metro SVG and label
where the projects differ. Request delivered to chat
`01a168d3-ac40-7c43-a22c-22fc05b21d8b`, title
`#202 Oyster public bivalve searches and report`. Do not edit that chat's worktree.
Both diagrams use blue solid read-preparation tracks and orange dashed discovery tracks,
separate file icons, legends before tracks, and exact program/file labels.
The oyster chat now plans scaled-100 archive sketches. That does not silently change
the Botryllus 5/10/100 protocol. Tunicate archive expansion remains unlaunched pending
resource/storage planning; chordates remain inventory only.

### Updated searches and map, 23 October 2026

Primary Sourmash pilot 49206896 completed in 11:59 with 4,553,276 KB peak RSS.
It searched all 987 pilot peptides against 45,339 Botryllus and 575,748 Swiss-Prot
records. All 18 raw CSVs exist; `primary-search-execution.json` is the execution
record. Raw rows include single-hash matches and are NOT validated candidate claims.
Apply the fixed two-shared-hash filter only after independent hash/CLI checks.
The public archive is explicitly named SRA/ENA on the first metro panel and has a
separate route. Its 5,776-run inventory is complete; external collection indexing
and occurrence searching have not been claimed complete.

Updated source-specific scales were relayed from Olga via the oyster chat:
public SRA DNA/protein/Dayhoff/HP uses only scales 10 and 100, comparing identical
runs/preprocessing/queries. Swiss-Prot protein/Dayhoff/HP production searches use 5.
Preserve the completed historical 5/10/100 pilots. The source-specific amendment is
in PROTOCOL.md; do not apply the archive reduction to Swiss-Prot.

Requested Swiss-Prot size comparison job 49210194 measures the original compressed
and uncompressed FASTA, retains existing scaled-5 reference ZIPs, and builds
scaled 10 and 100 for each protein encoding from identical canonical reference
inputs. It has 8 CPUs, 8 GB, 30 min, based on the completed 12-minute primary pilot.
Outputs: `$SCRATCH/botryllus-shazam-PEC-20261023/swissprot-size-comparison-20261023/`.
The scale-10/100 reference sketches are a requested storage measurement, not a change
to the production Swiss-Prot scale-5 rule.

SVG renderer correction: its exported canvas height must use the complete nf-metro
canvas height, not the legend's vertical coordinate. The renderer may place its
legend in an upper gap; using legend Y clipped downstream stations. Legends are
moved above the full geometry. All six source files render with nf-metro --strict.

## Latest execution and UI checkpoint, 17:14 UTC

This section supersedes job status above. 41/64 raw runs have completion markers.
Original batch 4 failed curl 18; recovery 49211401 preserves the partial transfer.
Preprocessing 49211406 replaces dependency-cancelled 49205669. Full discovery
49211416 replaces 49206899 and waits for preprocessing. Healthy downloads continue.
Storage benchmark 49212645 runs merged Swiss-Prot storage measurements and eight
Branchwater RocksDB/manysearch self-checks using the historical raw focal run.
No full external tunicate search has run. SBT outputs remain historical only.
The default map now includes SPLASH in an nf-metro Seqera overview. Detailed maps
and file tables show Branchwater RocksDB directories; streaming ZIP search is also
authorized. Sequence extensions show support, translated peptides, and retained match
counts instead of raw anchor descriptions. New native primary matches remain unvalidated.

## Shared search command, 23 October 2026

Use Branchwater manysearch for both Swiss-Prot protein references and public SRA
collections. Both manysearch and the historical multisearch are multithreaded.
Manysearch supports signature ZIP collections and RocksDB targets; it can load
target sketches on demand instead of holding the whole collection in memory.
The completed multisearch pilot is preserved. Comparison job 49215765 reuses the
same sketches, encodings, scales, and query-containment threshold before its new
match tables can replace historical results. No new biological validation is
claimed from engine agreement alone. Swiss-Prot production remains scaled 5.
