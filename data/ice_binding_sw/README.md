# Seedless alignment benchmark on ice-binding proteins (notebook 349)

Made by `scripts/ice_binding_sw_benchmark.py all` with `scripts/ice_binding_sw_config.toml`.
Read and drawn by `notebooks/349_ice_binding_seedless_alignment_swissprot.ipynb`.

| File | What it holds |
|---|---|
| `origins.tsv` | Hand-written origin rules (UniProt family, taxonomic family, name), with the paper behind each |
| `alphabet_classes.tsv` | `hp_lehninger2` and `dayhoff6` classes, from kmerseek 3c353d4 via PR 141 |
| `truth_manifest.tsv` | The 59 truth-set IBPs: mature boundaries, origin, evidence on the UniProt FUNCTION comment, assays it names |
| `manifest.tsv` | Every sequence aligned: IBPs, controls, shuffles; split, status, category, mature sequence |
| `split_dev.tsv`, `split_eval.tsv` | Sequence IDs per split |
| `dedupe_removed.tsv` | Sequences collapsed as near-duplicates, and what was kept instead |
| `leakage.tsv`, `leakage_composition_only.tsv` | Evaluation sequences close to a development sequence; the second file lists pairs close by identity but not beyond shuffles |
| `control_candidates.tsv` | Every control picked, and whether it was kept, collapsed or removed as a leak |
| `background_frequencies.tsv` | Residue frequencies of Swiss-Prot 2026_03 |
| `matrices.tsv` | Every substitution matrix used, in half bits, with its expected score |
| `pairs_dev.tsv.gz`, `pairs_eval.tsv.gz` | Every query-target alignment: score, coordinates (0-based, end-exclusive), identity |
| `calibration.tsv` | Lambda and K per setting, fitted on development control pairs |
| `dev_grid_metrics.tsv`, `selection.tsv` | The grid on the development split, and the setting chosen per alphabet |
| `eval_guard.json` | MD5 of `selection.tsv` and of the config when the evaluation split was aligned |
| `ranked_results.tsv.gz` | Per query, every target ranked, for the chosen settings and the baselines |
| `metrics_per_query.tsv`, `metrics_per_origin.tsv`, `metrics_pooled.tsv`, `pr_curves.tsv` | Average precision, recall at false-hit budgets, pooled recall at false-positive rates |
| `control_comparison.tsv` | AUC of other-origin IBPs against each kind of control and against shuffles |
| `calibration_check.tsv` | Observed vs promised chance-hit rate on controls and shuffles |
| `subgroup_metrics.tsv`, `decision.tsv` | The predeclared subgroups and the decision rule's result |
| `representative_alignments.txt`, `.tsv` | Best other-origin IBP and best control per query origin, residue by residue |
| `runtime.tsv` | Wall time and peak memory per step |
| `raw/` | UniProt REST responses, each with its release; `fetch_log.tsv` has the URLs |
