# parse_disprot.py: what the June 2026 table got wrong, and what used it

Checked 2026-10-06 against the DisProt API (`release=current`, which today returns the same
3_337 entries as the `2026_06` release that notebook 251 cached).

## The three bugs

1. The script asked for `page=1` first. DisProt pages count from 0, so page 0 was skipped.
2. It read the entry count from `payload["count"]` or `payload["total"]`. The API calls it
   `payload["size"]`, so the fallback (the number already downloaded) ended the loop after
   one page.
3. It read `entry["regions"]`, which holds every DisProt annotation (disorder, structural
   transitions, functions, binding partners). `entry["disprot_consensus"]["regions"]` does not
   exist. Consensus disorder is `entry["disprot_consensus"]["Structural state"]` with
   `type == "D"`.

With `page_size=2000`, bugs 1 and 2 together kept only entries 2001 onwards: DP02991 and up.

## The table on disk

`~/data/disprot-benchmark/{disprot,disprot_null,null}/results/disprot/disprot_human.tsv` are
the same file: three runs downloaded it on 2026-06-05, 06-06 and 06-12 and got identical
tables.

| | June table | fixed script, current release |
|---|---|---|
| human DisProt proteins | 504 | 1_339 |
| lowest DisProt ID | DP03004 | DP00004 |
| proteins kept as benchmark queries (`map_disprot_pfam.py`, Pfam homologs in >= 3 species) | 271 | 710 |

The old script, run on the second page of the `2026_06` release, gives the June regions for
502 of 503 shared proteins (DP03755 is in the June table but not in `2026_06`). Running
`map_disprot_pfam.py` on the June table gives the June `disprot_pfam_mapping.tsv` byte for
byte, so the 710 above is computed the same way as the 271.

Bug 3 changed the regions of 16 of the 503 proteins (46_371 disordered residues in the June
table, 46_299 from consensus disorder). No reported number depends on it:
`n_disordered_regions`, `total_disordered_residues` and `regions_json` are carried into
`disprot_pfam_mapping.tsv` and the ground-truth parquets but nothing downstream reads them.
`disorder_location` is `no_pfam` for every protein because `map_disprot_pfam.py` has no
domain coordinates (`domains = []`), whatever the regions are. The ordered / partial /
disordered split in notebook 081 comes from metapredict (`predict_disorder.py`), not from
DisProt.

## Numbers that came from the truncated query set

Every number from the `--database disprot` runs. They are computed on 271 human proteins
drawn from the second page of DisProt, not the 710 the full release gives.

- `~/data/disprot-benchmark/disprot/results/`: `benchmark_stats.txt` (271 queries, per-species
  pair counts such as mouse 1_561 pairs / 662 positives), `all_disprot_metrics.parquet`,
  `all_disprot_pr_curves.parquet`, `metrics/`, `figures/`, `disprot_benchmark_report.md`,
  `multiqc_report.html`, and the kmerseek k20/k24/k25/k26, MMseqs2 and FoldSeek search
  outputs (searches of the 271 queries).
- `~/data/disprot-benchmark/disprot_null/results/` and `null/results/`: the same 271 queries.
- Notebook 081 (`notebooks/081_disprot_benchmark_analysis.ipynb`), every DisProt column:
  - 271 DisProt queries, split 88 ordered / 135 partial / 48 disordered;
  - AUC-PR, mean over 9 species: MMseqs2 0.715 / kmerseek k26 0.570 (all), 0.698 / 0.467
    (ordered), 0.729 / 0.614 (partial), 0.666 / 0.535 (disordered), and the gaps 0.231,
    0.115, 0.131;
  - recall at 5% FDR, 5-9% kmerseek vs 40-48% MMseqs2;
  - PESK fraction 24.4% / 27.4% / 33.2%; kmerseek k26 false-positive rate 0.089 / 0.107 /
    0.133; Pearson r 0.004 (kmerseek, n=149) and -0.084 (MMseqs2, n=243);
  - score statistics: kmerseek TP n=413, FP n=65, AUROC 0.536; MMseqs2 TP n=2_345;
  - figures, DisProt only: `081_auc_heatmap_disprot.pdf`, `081_kmerseek_vs_mmseqs2_disprot.pdf`,
    `081_pr_curves_disprot.pdf`, `081_compositional_inflation_check.pdf`,
    `081_score_calibration.pdf`; and the DisProt half of `081_auc_by_tool_disorder_db.pdf`,
    `081_mmseqs2_disprot_vs_mobidb.pdf`, `081_pr_curves_kmerseek_vs_mmseqs2.pdf`,
    `081_per_protein_recall_vs_disorder.pdf`, `081_protein_characteristics.pdf`,
    `081_recall_vs_divergence.pdf` and `081_recall_vs_protein_length.pdf`.

## Not affected

- The MobiDB columns of notebook 081 and `~/data/disprot-benchmark/mobidb/`:
  `parse_mobidb.py` streams one request and does not page.
- Notebook 251 (DisProt functional-region transfer, branch `olgabot/disprot-region-transfer`):
  `scripts/fetch_disprot.py` downloads the `2026_06` release in one page of 100_000 (3_337 of
  3_337 entries in `~/data/disprot-region-transfer/raw/disprot_2026_06.json`) and picks
  regions by namespace itself ("Disorder function" overlapping experimental "Structural
  state" disorder). It does not call `parse_disprot.py`.
- `nextflow-runs/regions-index-pilot/tools/fetch_inputs.py` has its own corrected fetcher.

## What needs rerunning

1. The `--database disprot` run, without `-resume` for the download step. `downloadDisprot`
   calls `${projectDir}/bin/parse_disprot.py` by path, so the task hash does not change when
   the script does and `-resume` would reuse the June table. Do not use
   `rerun-disprot-missing-arms` (`--reuse_results true`): it reads the old ground truth from
   the output directory. To pin the release, pass
   `--disprot_json ~/data/disprot-region-transfer/raw/disprot_2026_06.json`, the release
   notebook 251 used; the fixed script checks that a local file holds all `size` entries.
2. The runs whose output went to `disprot_null/` and `null/`, if they are still wanted.
3. Notebook 081, after the rerun, then its summary table and text.
