# Benchmark outputs found on 2026-10-15

Every finished or partly finished benchmark output, where it is, when it was written, the
commit that wrote it, and whether `results.parquet` uses it. Paths are under
`/scratch/users/olgabot/` on Sherlock. "Commit" is HEAD of the Sherlock clone when the run
was launched, read from that clone's `git log -g` (dark-set Fmax records its own commit in
`code/COMMIT`). Dates are the output file's modification time.

## Used in results.parquet

| experiment | output | written | commit | complete? |
|---|---|---|---|---|
| qfo_region_mini | `2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/mini/results/all_domain_metrics.parquet` | 2026-09-09 | dd17259 | yes: test set, 2 targets (E. coli, yeast) |
| qfo_region_midi | `.../qfo-pfam-region-benchmark/data/midi/results/all_domain_metrics.parquet` | 2026-08-31 | 4cd14cc | yes: 9 QfO targets |
| qfo_region_midi_plus | `.../qfo-pfam-region-benchmark/data/midi-plus/results/all_domain_metrics.parquet` | 2026-09-13 | a47ea25 | yes for 15 targets. The all-QfO launch (`run-midi-plus-all-qfo`) searched 77 targets, but the aggregated metrics hold 15; the others were never aggregated |
| elm_cover (nb 253, PR 72) | `250-elm-cover/cover/<species>/*.summary.json` (463 files, 8 species) | chicken 2026-09-28, others 2026-09-30 | b26328b (chicken), ca051a6 (seven species) | yes. Foldseek and Reseek were scored on orthologs with no AlphaFold model too, so their shares are low for a reason that is not the search (memory note `elm_motif_benchmark_state`) |
| dark_set_pfam_fmax (nb 291) | `invertebrate-dark-set/results-0.4/dark-fmax-2026-10-14/dark_fmax_metrics.csv` | 2026-10-14 | 0cb74bb | yes: 4 species, 230 kmerseek settings + 3 sequence tools |
| dark_set_gain (ladder-0.4-midi) | `invertebrate-dark-set/results-0.4/<species>/<species>_kmerseek_dark_gain.parquet` (5 species) | 2026-10-12 | 0dad375 | yes. Its trace holds only CACHED rows, so compute is null |
| dark_set_accuracy (PR 133) | `invertebrate-dark-set/results-0.4/accuracy/<species>_accuracy.json` (4 species) | 2026-10-13 | 645e771 | partly: only the best kmerseek setting per species was judged; 494 other region files not read |
| alphabet_ranking_three_cases (nb 241) | `241-alphabet-ranking/ranks.csv` | 2026-10-13 | bd9cc3b | yes: 3 queries x 119 settings x 11 scores |
| deep_twilight (nb 252) | `2024-kmerseek-analysis-deep-twilight/nextflow-runs/deep-twilight-controls/run/results/outcomes.parquet` | 2026-09-25 | c8cd2dd | yes: 16 proteins, 150 kmerseek settings + 3 tools |
| scope40_v040 (nb 277) | `scope/results-scope40-alphabet-ksize-v040/scope40_v040_auc.tsv` | 2026-10-12 | 702de2c | yes. No identity-bin file was written for this run |
| scope40_v040_below_kmin | `scope/results-scope40-alphabet-ksize-v040-below-kmin/scope40_v040_{auc,identity_bands}.tsv` | 2026-10-13 | 46dfecb | yes |
| scope40_v040_shuffled_k2_seed1 | `scope/results-scope40-alphabet-ksize-v040-shuffled-k2-seed1/scope40_v040_{auc,identity_bands}.tsv` | 2026-10-13 | 80ec410 | yes: a null control (alphabet letters shuffled), not a tool |
| pfam_pair_roc (`pfam-benchmark-tools`, added 2026-10-16) | **laptop** `/Users/olga/data/pfam-benchmark-tools/results/metrics/<tool>.<species>.metrics.parquet` (26 of 27 files) | 2026-04-03 to 2026-04-24 | not recorded: a laptop run, and the laptop clone's reflog starts later | partly. MMseqs2 (sequence and 3-iteration profile) on all 9 QfO targets; phmmer on 7 (chicken and fly searched, never scored); DIAMOND on E. coli only (6 of 7 hit files are empty 20-byte gzips, and zebrafish's metrics file is all NaN, so it is left out); HHblits on none (9 empty hit files). A different question from the region runs: see "How values were read" |

## Found, not used

| output | written | state | why not in the table |
|---|---|---|---|
| nb 230 (`notebooks/230_hp_class_conservation_in_aligned_homologs.ipynb`, main ba20e8b) | 2026-09-13 | done | No search: H/P agreement inside Pfam seed and SCOPe alignments. Measures alphabets, not a tool |
| nb 231 (main ba20e8b) | 2026-09-13 | done | Re-reads the midi-plus metrics already in the table. Its Kyte-Doolittle scan baseline is computed in the notebook and written to no file |
| nb 232 (main ba20e8b) | 2026-09-13 | done | No search: seed-and-extend recall on alignment pairs against shuffled pairs |
| nb 242, 243 (PR 48) | 2026-09-23/24 | done | Combinations of nb 241's searches (ensemble ranks, logistic regression); no table of per-tool results in a file |
| nb 244 (PR 49) | 2026-09-25 | done | Picks hero cases from the midi-plus calls; a selection, not a measurement |
| nb 250 (PR 51) | 2026-09-25 | done | No search: H/P run lengths on aligned orthologs around ELM motifs |
| nb 251, DisProt region transfer (`2024-kmerseek-disprot-region-transfer/.../run-disprot`) | 2026-09-29 | **stopped**: 716 tasks done, 636 aborted, 3,460 never started; no metrics written | No metrics file. The yeast dry run's numbers live only in the notebook |
| nb 279 / protein E-values (`invertebrate-dark-set/results-0.4/protein-evalues-repaired/`) | 2026-10-13 | done | Per-protein E-values, the input to the dark-set reach that `dark_set_gain` already holds |
| Time-split test, nb 280 (`280-time-split/results-pilot/scores-partial/setting_summary.csv`) | 2026-10-13 | **pilot, tune half only** | The test half is held out on purpose until the setting is chosen; the full run has not been launched |
| midi-plus 0.4 (`.../data/midi-plus-0.4`) | 2026-09-24 | **stopped after 33 min** | One region file, no metrics (memory note `midi_plus_0_4_never_ran`) |
| SCOPe40 0.4 alphabet combinations (`scope/results-scope40-alphabet-ksize-v040-combine*`) | 2026-10-15 05:43 | being written today | Rewritten this morning by another chat; read once it settles |
| nb 263 shuffled queries (`263-shuffled-queries`) | 2026-10-02 | done | Decoy calls for 4 zebrafish searches; a control for nb 260, no per-tool metric file |

## Other pipelines in `nextflow-runs/` (not read in this pass)

These have Makefile targets and some have finished outputs, but none was in the request
list and their output was not opened here, so whether each is complete is not stated:
`scope40-benchmark-aws` (kmerseek 0.3 SCOPe40 sweep, Figure 1), `pfam-benchmark-tools` and
`qfo-pfam-benchmark*` (protein-pair homology ROC, not domain finding),
`human-mouse-gencode-orthologs*` (family AUC and RBH-F1, PR 64), `kmer-spectra`,
`qfo-hp-alphabet-sweep`, `scope40-lowcomplexity-null` (nb 086), `scope40-foldseek-pdb`,
`disprot-benchmark`, `throughput-benchmark`, `vs-other-orthology-tools-resource-benchmark`,
`human-mouse-dnds-omega`, `qfo-dnds-omega`, `proteome-interproscan`, `proteome-unifire`.
The first three are the most likely to add rows next.

## Second pass, 2026-10-16: outputs looked for and not read

The pipelines below were listed above as "not read in this pass". Every count and
"absent" here is printed by `scripts/check_leaderboard_unread_outputs.py`; its output is
`unread_outputs_check.txt` in this folder.

| experiment | where the output should be | what is there | why no rows |
|---|---|---|---|
| human_mouse_orthologs (`human-mouse-gencode-orthologs` and its `-hp-v040` copy, PR 64): family AUC and RBH-F1 per tool | laptop `~/data/gencode/results-human-mouse-orthologs{,-hp-v040}` (the pipelines' `OUTDIR`) | absent on the laptop and on Sherlock. Memory note `mac_disk_cleanup_2026_09_13` says the 2026-09-13 sweep kept them; they were removed later. 7 Nextflow traces remain in the laptop clone | The aggregated tables (`200_rbh_f1_all_combos.csv`, `206_family_auc_all_combos.csv`) are gone. PR 64's Sherlock version of the sweep was never run (no output on Sherlock). Rebuilding needs the kmerseek searches again |
| throughput (`throughput-benchmark`): wall time, CPU and peak RSS per tool and query count | laptop `~/data/scope/results-throughput-benchmark/throughput_benchmark.tsv` (`params.outdir`) | absent. Notebook 078's committed output prints "Throughput benchmark not yet run" and uses a placeholder queries-per-second for kmerseek | Never run, so there is no compute row to read and no hardware to name |
| resource benchmark (`vs-other-orthology-tools-resource-benchmark`) | laptop clone `nextflow-runs/vs-other-orthology-tools-resource-benchmark/results/pipeline_trace.txt` | one trace row: `ORTHOFINDER (human_vs_mouse)`, ABORTED, submitted 2026-03-19 06:21 | Stopped before any task finished; nothing measured |
| scope40_lowcomplexity_null (`scope40-lowcomplexity-null`, notebook 086): pairs above the first false positive, a control beside `scope40_v040_shuffled_k2_seed1` | laptop `~/data/scope/results-lowcomplexity-null` (`params.outdir`) | absent on the laptop and on Sherlock. Memory note `lowcomplexity_null_pipeline` records a finished run there on 2026-07-29 (828 tasks) | The hits are gone. That null shuffles the sequences (the memory note says dipeptide-preserving), while `scope40_v040_shuffled_k2_seed1` shuffles the alphabet's letters; the two are different controls and would need different names even if the hits came back |

| kmerseek on the same Pfam pairs (`qfo-pfam-benchmark`, notebook 123): ROC AUC per species, k and score | laptop clone `results/pfam_benchmark/merged/auc_by_species_ksize_score.parquet` (1,188 rows); the raw searches in `~/data/qfo-pfam-benchmark` are absent | The AUC file is there, but it was computed on a merged table with repeated pairs: mouse at k=24 has 72,649 rows for 59,915 distinct pairs, and 32,208 positive rows where the pair file has 20,000 positive pairs. All 540 rows at k=10 to 18 have ROC AUC exactly 0.5, because the merged tables hold no kmerseek hit there | An AUC over repeated pairs is not the number the other tools' rows give (each pair once). Left out rather than shown beside them; recomputing needs the merge fixed first |

## How values were read

- Pfam pair benchmark (`pfam_pair_roc`): a different question from every other experiment
  here. It asks whether a tool ranks a protein pair that shares a Pfam domain above a pair
  that shares none (pairs from `notebooks/construct_subdomain_pairs.py`: positives share
  a domain but differ in domain architecture, up to 20,000 positive and 40,000 negative
  pairs per species). It does not ask where the domain is. Its metric names start with
  `pair_` (`pair_roc_auc`, `pair_average_precision`, `pair_recall_at_99pct_precision`,
  `pair_precision_at_50pct_recall`, `pair_n_with_hit`, `pair_n_negatives`) so they never
  share a name with a region metric. `n_instances` is the number of positive pairs.
  `pair_n_with_hit` counts pairs of either label that the tool scored above 0.

- QfO region and dark-set Fmax tables: `split == "all"`, stratum axes all, identity, pLDDT,
  feature type, feature length bin; metrics fmax, fmax_precision, fmax_recall,
  fmax_threshold (the tool-score cut where fmax is reached), auprc,
  recall_reachable, median_iou_tp, family_fmax; `n_instances` = n_truth_instances.
- ELM: the summary's counts, and each count over its denominator in the same file.
- Deep twilight: per (tool, setting, family) the number of correct calls, wrong carries and
  the lowest needle identity with a correct call (the nb 252 headline numbers).
- SCOPe40: `status == "ok"` rows; AUC per SCOPe level; identity bins only where the run
  wrote `identity_bands.tsv`.
- Dark-set accuracy: counts per judged class at each best-hit E-value cut.
- `n_instances` is the number of truth instances in that stratum. It is not the
  denominator of `fmax_recall` or `fmax_precision`: those are averaged over proteins at the
  best threshold (CAFA style), so `value x n_instances` is not a count of hits.
- Left out on purpose: the region runs' Swiss-Prot identity bins. Their only bin,
  "no_homolog", repeats the whole-set numbers (57,148 of 57,148 rows), so the identity
  step did not run for Swiss-Prot features.

## Adversarial check, 2026-10-15

An agent that had not seen the tables or the builder recomputed ten rows drawn at random
(seed 20261015) from their `source_path`, on the laptop copies, whose sha256 matched
Sherlock. Uniform sampling drew nine rows from the 9-species midi run, which holds two
thirds of the table.

| # | row | verdict | source row (0-based) |
|---|---|---|---|
| 1 | midi, kmerseek uniprot18_k6_lcFalse, worm, Pfam-N, pLDDT 90-100, fmax_recall 0.0302 | HOLDS | all_domain_metrics.parquet 411531 |
| 2 | midi, hp_pbotc_1st_ed2_k29_lcTrue, yeast, Swiss-Prot, DNA_BIND, fmax_precision 0 | HOLDS | 1316927 |
| 3 | midi, hp_kyte_doolittle2_k26_lcTrue, mouse, Swiss-Prot, ACT_SITE, median_iou_tp 0.0106 | HOLDS | 80917 |
| 4 | midi, gbmr7_k10_lcTrue, zebrafish, Swiss-Prot, pLDDT 70-90, fmax_recall 0 (no calls) | HOLDS | 1294782 |
| 5 | midi, hsdm17_k11_lcFalse, arabidopsis, Swiss-Prot, INTRAMEM, fmax_recall 0 (9 instances) | HOLDS | 1324697 |
| 6 | midi, wass14_k12_lcFalse, chicken, Swiss-Prot, identity no_homolog, fmax_recall 0.0337 | WRONG: value matches, label does not (the whole set). Rows like it are now dropped | 396984 |
| 7 | midi, gbmr7_k16_lcFalse, fly, Swiss-Prot, length 2-15, fmax_precision 0.0625 | HOLDS | 706713 |
| 8 | midi, mmseqs12_k14_lcFalse, fly, Swiss-Prot, length 61-120, median_iou_tp 0.653 | HOLDS | 587728 |
| 9 | midi, wass14_k7_lcTrue, worm, Swiss-Prot, length 2-15, median_iou_tp 0.583 | HOLDS | 356071 |
| 10 | SCOPe40 below k_min, hp_thomas_dill_no_c2.k22.c_min, 30-40%, same fold, pairs above first FP 0 of 2 | HOLDS | scope40_v040_identity_bands.tsv 230804 |

Every row's key matched exactly one source row, and both commits exist and predate their
files. Other findings, now fixed: the region query-set text said "964 queries" for every
truth set (removed). Not fixed: rows 4, 5 and 10 are zeros on strata with no calls or very
few instances; the site should show `n_instances` beside every value.
