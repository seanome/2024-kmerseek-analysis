# Trimmed-read recovery and Nextflow migration

Written by Codex at Olga's request. Checked 2026-10-25 Pacific.

## Recovered experimental inputs

The preserved workflow log records 64 distinct Trim Galore tasks completing with exit
zero. The nf-core head failed later because Salmon could not infer strandedness for
SRR32233927. The six dependent discovery/abundance/search jobs were cancelled before
execution. No full-depth discoveries or expression results were imported from them.

Recovery job 49512151 completed with Slurm exit 0:0. It validated all 128 gzip files,
synchronized identifiers, sequence/quality lengths and pair counts, then computed
SHA-256 and created the shared trimmed-read links. It retained 405,240,058 pairs from
654,672,879 raw pairs across 64 libraries. These are read-validation counts, not
biological results. Every library remains in the denominator, including tiny outputs.

![Raw and retained paired reads](trimmed-pairs-PRJNA1219309.png)

Each point is one library. The grey line shows equal raw and retained counts. All
64 libraries passed validation; trimming retained fewer reads in every library.

`trimmed-inputs.validated.json` records each file and checksum. The validation command
ran for 5:31.39; the Slurm job elapsed 5:44. The command's maximum RSS was 36,760 KB,
whereas Slurm reported 8,388,036 KB for the batch step. Use the larger scheduler
measurement for future allocation sizing; these are different accounting scopes.

Original FASTQ files, nf-core logs, failed-run work directories, historical query
protocols and old results were preserved. The modified Figure 4 plotting script,
exercise-source directory and historical search-driver edits remain unfinished.

## Migration implementation and gates

The workflow is `pipelines/botryllus-shazam/main.nf`. It calls pinned counting tools
directly, keeps separate per-library tasks, and reuses the established pooling and
blocked-model code. Both SPLASH and DE-kupl sequences reach translation, target
preparation, per-setting searches and the gene-evidence join.

The protocol is six frames, stop-only splitting, internal X retained, minimum 30 aa.
Target sketches are rebuilt into new versioned outputs. Search candidates remain
candidates; exact seed-to-gene containment does not prove full-context alignment,
orthology, function or splicing. Optional genome alignment and query Pfam stages
require explicit reference assets and executables.

Twelve Python helper and pinned-executable tests passed on Sherlock. They cover
selection boundaries, gene mapping, cache-copy isolation, both-source joins,
translation, X retention, empty searches and native counter boundaries.

The corrected workflow completed all 125 tasks in fixture 49516351. Its count and
adjusted-P checks passed, but its first acceptance check incorrectly required DE
seeds to be nonempty. The corrected check requires query presence to follow the
number of significant seeds. Job 49516796 completed with exit 0:0. All 125 tasks
resumed from cache and the independent checks passed. Corrected native counts
matched all 18 libraries of the previously verified DE-positive fixture.

| Synthetic fixture | SPLASH DNA / peptides | DE-kupl DNA / peptides | Independent count rows |
|---|---:|---:|---:|
| 49514880 | 0 / 0 | 1,104 / 1,151 | 29,483 |
| 49516351, verified by 49516796 | 72 / 96 | 0 / 0 | 29,459 |

Both fixtures have 18 libraries, six independently checked complete-population
BH corrections and 100 retained gene rows. The second selected 21 of 21 tested
SPLASH anchors and zero DE-kupl seeds. These are software checks, not experimental
findings. Nonempty recovery and valid empty outputs are covered across the fixtures.

Pipeline source: `8f659e2c`; verifier and cross-fixture comparison: `92555f15`.
The fresh corrected workflow used 7:23 and 1,122,568 KB peak RSS, including its
failed overly strict postcondition. Successful resume plus verification used 0:50
and 328,308 KB. Trace and verifier JSON files are stored beside this report.

Production resources, experimental-data equivalence, reference-backed genome/Pfam
execution and archive-wide target preparation remain unvalidated. The resource
preparation script records measured requests and stops when supplied traces contain
failed tasks.

The app remains the historical snapshot. No experimental output from this migration
has been published to it. Archive-wide tunicate preparation still needs explicit
input batches; a manifest containing four prepared runs is a four-run search.

## Software-test failures retained during development

All test inputs are synthetic. No failed test output was imported as experimental
results. Each test directory remains separate on Sherlock.

- Job 49513321 exposed a malformed smoke-profile selector; the profile syntax was fixed.
- Job 49513387 exposed identical synthetic feature profiles that made the dispersion
  fit singular. The fixture now varies counts and read starts. The model was not changed.
  Inspection also exposed directory copies following staged cache links; explicit
  dereferencing and a source-unchanged regression test now protect those inputs.
- Job 49513563 exposed a duplicate Groovy closure variable; it was renamed.
- Job 49513576 completed 124 of 125 tasks, including all six models and all eight
  searches. The final join failed because native manysearch CSVs already contain a
  `scaled` column. The join now merges those fields explicitly; its regression fixture
  includes the native column. This run is not a successful end-to-end test.

Pinned manysearch help also showed that `-A` includes zero-overlap comparisons for ZIP
collections. The migration removes that flag, keeps raw CSVs and explicitly retains
only positive `intersect_hashes`. A successful disjoint search writes a zero-byte CSV
in the pinned executable; job 49514822 exposed that case. The migration keeps the
raw file and writes a known empty-table header. The corrected identical/disjoint
and X-retention tests passed
in job 49514880 before the workflow started. Historical search outputs and drivers
are unchanged.

## Completed empty-SPLASH fixture

Job 49514880 used code commit `6f7a7afc`. Its committed trace contains 125 completed
tasks and no failed tasks. The independent verifier reproduced 29,483 canonical
31-mer count rows across 18 libraries and all six complete-population BH corrections.
The join retained 100 gene rows. DE-kupl produced 1,104 DNA sequences and 1,151
peptides; SPLASH had zero tested anchors. These are synthetic software-test counts.
The fixture does not establish biological recovery or model equivalence on the
experimental libraries.

The allocation requested one CPU, 4 GB and one hour. Subsequent workflow tests used
the same resources; the observed peak was about 1.1 GB and elapsed time was 7:29. Production
requests must be measured separately.

## Independent recount caught a native counter error

The nonempty-SPLASH fixture, job 49515476, completed the workflow but failed its
independent count comparison. In S11T, `AATACATGAAGAACCACAATTGGCAGATTCG` occurred
134 times in the reads; pinned Jellyfish 2.2.10 reported six. Diagnostic job 49515995
reproduced six without count-time filtering, and returned 134 with a 32-bit counter.
The internal cause is not established beyond this dependence on counter width.

Regression jobs 49516138 and 49516219 returned zeros with eight-byte output.
Job 49516266 tested 12 combinations: eight-byte output returned zeros at every
tested memory width; seven-byte output with 32, 63 or 64 memory bits returned 134.
The workflow now requests explicit 63-bit in-memory counters and seven-byte output
counters. A regression test runs the actual counting script at counts around 128
and 256, up to 2,048. The migrated workflow has not counted or modeled the
experimental reads. The new counter configuration needs fresh production resource
measurements.
