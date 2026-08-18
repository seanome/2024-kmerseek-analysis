#!/usr/bin/env python3
"""Renumber notebooks to 3-digit format and move/rename figures into figures/."""

import os
import shutil
from pathlib import Path

NOTEBOOKS_DIR = Path("/Users/olga/code/2024-kmerseek-analysis/notebooks")
FIGURES_DIR = Path("/Users/olga/code/2024-kmerseek-analysis/figures")

# --- Notebook renames: (old_name, new_name) ---
NOTEBOOK_RENAMES = [
    # 2-digit → 3-digit (straightforward padding)
    ("00-explore-protein-benchmark.ipynb", "000-explore-protein-benchmark.ipynb"),
    ("01-make-test-datasets.ipynb", "001-make-test-datasets.ipynb"),
    ("02-make-query-metadata.ipynb", "002-make-query-metadata.ipynb"),
    ("03-process-protein-benchmarks.ipynb", "003-process-protein-benchmarks.ipynb"),
    ("04-rewrite-multisearch-parser-to-polars.ipynb", "004-rewrite-multisearch-parser-to-polars.ipynb"),
    ("05-scan-s3-csv-to-s3-parquet.ipynb", "005-scan-s3-csv-to-s3-parquet.ipynb"),
    ("06-process-protein-benchmarks-polars.ipynb", "006-process-protein-benchmarks-polars.ipynb"),
    ("07-mem-opt-process-protein-benchmarks-polars.ipynb", "007-mem-opt-process-protein-benchmarks-polars.ipynb"),
    ("11-pandas-explore-compute-sensitivity-to-first-false-positive.ipynb", "011-pandas-explore-compute-sensitivity-to-first-false-positive.ipynb"),
    ("12-polars-explore-compute-sensitivity-to-first-false-positive.ipynb", "012-polars-explore-compute-sensitivity-to-first-false-positive.ipynb"),
    ("13-compute-all-sensitivity-to-first-fp.ipynb", "013-compute-all-sensitivity-to-first-fp.ipynb"),
    ("14-plot-sensitivity-to-first-fp.ipynb", "014-plot-sensitivity-to-first-fp.ipynb"),
    ("15-which-kmers-match.ipynb", "015-which-kmers-match.ipynb"),
    ("16-ced9-bcl2-and-p66-cd47.ipynb", "016-ced9-bcl2-and-p66-cd47.ipynb"),
    ("17-p66-cd47.ipynb", "017-p66-cd47.ipynb"),
    ("18-explore-hp-k10.ipynb", "018-explore-hp-k10.ipynb"),
    ("20-explore-protein-benchmark-polars.ipynb", "020-explore-protein-benchmark-polars.ipynb"),
    ("21-subsample-1000-queries-per-moltype-ksize-to-compare-values.ipynb", "021-subsample-1000-queries-per-moltype-ksize-to-compare-values.ipynb"),
    ("22-plot-subsampled-sourmash-scores-per-moltype-ksize.ipynb", "022-plot-subsampled-sourmash-scores-per-moltype-ksize.ipynb"),
    ("23-compute-sklearn-benchmarks-plot-protein-benchmark.ipynb", "023-compute-sklearn-benchmarks-plot-protein-benchmark.ipynb"),
    ("30-test-polars-read-fasta.ipynb", "030-test-polars-read-fasta.ipynb"),
    ("40-cargo-bench-string-with-capacity.ipynb", "040-cargo-bench-string-with-capacity.ipynb"),
    ("50-scope-benchmark-diff.ipynb", "050-scope-benchmark-diff.ipynb"),
    ("51-scope40-hp-k15.ipynb", "051-scope40-hp-k15.ipynb"),
    ("52-scope40-hp-k15-bcl2-ced9.ipynb", "052-scope40-hp-k15-bcl2-ced9.ipynb"),
    ("53-scope40-hp-k15-plots.ipynb", "053-scope40-hp-k15-plots.ipynb"),
    ("54-scope40-hp-dayhoff-protein-plots.ipynb", "054-scope40-hp-dayhoff-protein-plots.ipynb"),
    ("55-scope40-hp-dayhoff-protein-sensitivity.ipynb", "055-scope40-hp-dayhoff-protein-sensitivity.ipynb"),
    # 56 duplicate: original → 056, clean (newer) → 057, then push 57-64 forward
    ("56-scope40-hp-k15-k20-vs-tea.ipynb", "056-scope40-hp-k15-k20-vs-tea.ipynb"),
    ("56-scope40-hp-k15-k20-vs-tea-clean.ipynb", "057-scope40-hp-k15-k20-vs-tea-clean.ipynb"),
    ("57-hp-k20-vs-foldseek-tea.ipynb", "058-hp-k20-vs-foldseek-tea.ipynb"),
    ("58-hp-k20-30-comparison.ipynb", "059-hp-k20-30-comparison.ipynb"),
    ("59-hp-k41-unique-vs-foldseek-analysis.ipynb", "060-hp-k41-unique-vs-foldseek-analysis.ipynb"),
    ("60-hp-k35-k36-family-analysis.ipynb", "061-hp-k35-k36-family-analysis.ipynb"),
    ("61-hp-k24-threshold-optimization.ipynb", "062-hp-k24-threshold-optimization.ipynb"),
    ("62-hp-unique-families-deep-analysis.ipynb", "063-hp-unique-families-deep-analysis.ipynb"),
    ("63-hp-k24-ml-classification.ipynb", "064-hp-k24-ml-classification.ipynb"),
    ("64-hp-scope-pvalue-benchmark.ipynb", "065-hp-scope-pvalue-benchmark.ipynb"),
    # 015/016/017/018 → 115/116/117/118
    ("015-human-mouse-ortholog-double-peaks.ipynb", "115-human-mouse-ortholog-double-peaks.ipynb"),
    ("016-kmerseek-vs-orthofinder.ipynb", "116-kmerseek-vs-orthofinder.ipynb"),
    ("017-human-mouse-ortholog-bimodality-investigation.ipynb", "117-human-mouse-ortholog-bimodality-investigation.ipynb"),
    ("018-kmerseek-vs-orthofinder-mgi-ground-truth.ipynb", "118-kmerseek-vs-orthofinder-mgi-ground-truth.ipynb"),
]

# --- Figure renames: (old_name, new_name_in_figures_dir) ---
# Prefix = new notebook number that generated the figure
FIGURE_RENAMES = [
    # From 053 (was 53-scope40-hp-k15-plots)
    ("sensitivity_curves.pdf", "053-sensitivity_curves.pdf"),
    ("query_coverage.pdf", "053-query_coverage.pdf"),
    ("precision_recall_overlap_filter.pdf", "053-precision_recall_overlap_filter.pdf"),
    ("combined_benchmark_figure.pdf", "053-combined_benchmark_figure.pdf"),
    ("tea_comparison_sensitivity.pdf", "053-tea_comparison_sensitivity.pdf"),
    ("tea_comparison_auc.pdf", "053-tea_comparison_auc.pdf"),
    ("precision_recall_overlap_corrected.pdf", "053-precision_recall_overlap_corrected.pdf"),
    ("tea_comparison_optimized.pdf", "053-tea_comparison_optimized.pdf"),
    ("performance_by_threshold.pdf", "053-performance_by_threshold.pdf"),
    # From 055 (was 55-scope40-hp-dayhoff-protein-sensitivity)
    ("sensitivity_boxplots_top_metrics.pdf", "055-sensitivity_boxplots_top_metrics.pdf"),
    ("metric_ranking_overall.pdf", "055-metric_ranking_overall.pdf"),
    ("sensitivity_heatmaps_best_metrics.pdf", "055-sensitivity_heatmaps_best_metrics.pdf"),
    # From 058 (was 57-hp-k20-vs-foldseek-tea)
    ("hp_k20_tfidf_distribution.pdf", "058-hp_k20_tfidf_distribution.pdf"),
    ("hp_k20_all_metrics_by_match_type.pdf", "058-hp_k20_all_metrics_by_match_type.pdf"),
    ("hp_k20_all_metrics_by_family_match.pdf", "058-hp_k20_all_metrics_by_family_match.pdf"),
    ("hp_k20_all_metrics_by_superfamily_match.pdf", "058-hp_k20_all_metrics_by_superfamily_match.pdf"),
    ("hp_k20_vs_foldseek_tea_sensitivity.pdf", "058-hp_k20_vs_foldseek_tea_sensitivity.pdf"),
    ("hp_k20_comparison_detailed.pdf", "058-hp_k20_comparison_detailed.pdf"),
    # From 059 (was 58-hp-k20-30-comparison)
    ("hp_k20_30_performance_summary.pdf", "059-hp_k20_30_performance_summary.pdf"),
    ("hp_k20_30_intersecting_hashes.pdf", "059-hp_k20_30_intersecting_hashes.pdf"),
    ("hp_k20_30_jaccard_distribution.pdf", "059-hp_k20_30_jaccard_distribution.pdf"),
    ("hp_k20_30_max_containment_distribution.pdf", "059-hp_k20_30_max_containment_distribution.pdf"),
    ("hp_k20_30_tfidf_distribution.pdf", "059-hp_k20_30_tfidf_distribution.pdf"),
    ("hp_k20_30_average_kmer_frequency_distribution.pdf", "059-hp_k20_30_average_kmer_frequency_distribution.pdf"),
    ("hp_k20_30_observed_over_expected_distribution.pdf", "059-hp_k20_30_observed_over_expected_distribution.pdf"),
    ("hp_k20_30_sensitivity_tfidf_0.0.pdf", "059-hp_k20_30_sensitivity_tfidf_0.0.pdf"),
    ("hp_k20_30_sensitivity_tfidf_1.pdf", "059-hp_k20_30_sensitivity_tfidf_1.pdf"),
    ("hp_k20_30_sensitivity_tfidf_10.pdf", "059-hp_k20_30_sensitivity_tfidf_10.pdf"),
    ("hp_k20_30_sensitivity_tfidf_50.pdf", "059-hp_k20_30_sensitivity_tfidf_50.pdf"),
    ("hp_k20_30_sensitivity_tfidf_100.pdf", "059-hp_k20_30_sensitivity_tfidf_100.pdf"),
    ("hp_k20_30_sensitivity_tfidf_500.pdf", "059-hp_k20_30_sensitivity_tfidf_500.pdf"),
    ("hp_k20_30_sensitivity_tfidf_1000.pdf", "059-hp_k20_30_sensitivity_tfidf_1000.pdf"),
    ("hp_k20_30_sensitivity_tfidf_5000.pdf", "059-hp_k20_30_sensitivity_tfidf_5000.pdf"),
    ("hp_k20_30_sensitivity_tfidf_10000.pdf", "059-hp_k20_30_sensitivity_tfidf_10000.pdf"),
    ("hp_k20_30_comprehensive_comparison.pdf", "059-hp_k20_30_comprehensive_comparison.pdf"),
    # From 060 (was 59-hp-k41-unique-vs-foldseek-analysis)
    ("hp_k41_unique_scop_class_distribution.pdf", "060-hp_k41_unique_scop_class_distribution.pdf"),
    ("hp_k41_unique_metrics_distribution.pdf", "060-hp_k41_unique_metrics_distribution.pdf"),
    ("hp_k41_unique_sfamcnt_comparison.pdf", "060-hp_k41_unique_sfamcnt_comparison.pdf"),
    # From 061 (was 60-hp-k35-k36-family-analysis)
    ("hp_k35_k36_fam_vs_sfam_comparison.pdf", "061-hp_k35_k36_fam_vs_sfam_comparison.pdf"),
    ("hp_k35_k36_superfamily_comparison.pdf", "061-hp_k35_k36_superfamily_comparison.pdf"),
    ("hp_k35_k36_class_distribution.pdf", "061-hp_k35_k36_class_distribution.pdf"),
    ("hp_k35_k36_scope_characteristics.pdf", "061-hp_k35_k36_scope_characteristics.pdf"),
    ("hp_k35_k36_scope_threshold_analysis.pdf", "061-hp_k35_k36_scope_threshold_analysis.pdf"),
    ("hp_k35_k36_scope_threshold_analysis_families.pdf", "061-hp_k35_k36_scope_threshold_analysis_families.pdf"),
    ("hp_vs_foldseek_tea_by_threshold.pdf", "061-hp_vs_foldseek_tea_by_threshold.pdf"),
    ("hp_vs_foldseek_tea_threshold_comparison_families.pdf", "061-hp_vs_foldseek_tea_threshold_comparison_families.pdf"),
    ("hp_family_venn_by_threshold.pdf", "061-hp_family_venn_by_threshold.pdf"),
    ("hp_family_upset_th10.pdf", "061-hp_family_upset_th10.pdf"),
    ("hp_family_upset_th20.pdf", "061-hp_family_upset_th20.pdf"),
    ("hp_family_upset_th25.pdf", "061-hp_family_upset_th25.pdf"),
    ("hp_family_upset_th30.pdf", "061-hp_family_upset_th30.pdf"),
    ("hp_family_upset_th35.pdf", "061-hp_family_upset_th35.pdf"),
    ("hp_family_upset_th40.pdf", "061-hp_family_upset_th40.pdf"),
    ("hp_k35_k36_disorder_analysis.pdf", "061-hp_k35_k36_disorder_analysis.pdf"),
    # From 062 (was 61-hp-k24-threshold-optimization)
    ("hp_k24_single_metric_sweeps.pdf", "062-hp_k24_single_metric_sweeps.pdf"),
    ("hp_k24_grid_search_heatmap.pdf", "062-hp_k24_grid_search_heatmap.pdf"),
    ("hp_k24_grid_search_query_counts.pdf", "062-hp_k24_grid_search_query_counts.pdf"),
    ("hp_k24_fine_grid_n_hashes_10_15.pdf", "062-hp_k24_fine_grid_n_hashes_10_15.pdf"),
    ("hp_k24_combined_threshold_heatmaps.pdf", "062-hp_k24_combined_threshold_heatmaps.pdf"),
    # From 063 (was 62-hp-unique-families-deep-analysis)
    ("hp_unique_families_scop_class_distribution.pdf", "063-hp_unique_families_scop_class_distribution.pdf"),
    ("hp_unique_sequence_length_comparison.pdf", "063-hp_unique_sequence_length_comparison.pdf"),
    ("per_class_venn_diagrams.pdf", "063-per_class_venn_diagrams.pdf"),
    ("hp_contribution_by_class.pdf", "063-hp_contribution_by_class.pdf"),
    # From 064 (was 63-hp-k24-ml-classification)
    ("hp_k24_feature_importance.pdf", "064-hp_k24_feature_importance.pdf"),
    ("ml_high_conf_feature_distributions.pdf", "064-ml_high_conf_feature_distributions.pdf"),
    ("ml_high_conf_tp_fp_distributions.pdf", "064-ml_high_conf_tp_fp_distributions.pdf"),
    # From 065 (was 64-hp-scope-pvalue-benchmark)
    ("scope_pvalue_recall_vs_ksize.pdf", "065-scope_pvalue_recall_vs_ksize.pdf"),
    ("scope_pvalue_precision_vs_ksize.pdf", "065-scope_pvalue_precision_vs_ksize.pdf"),
    ("scope_pvalue_pr_curves.pdf", "065-scope_pvalue_pr_curves.pdf"),
    ("scope_pvalue_sensitivity_vs_tea_foldseek.pdf", "065-scope_pvalue_sensitivity_vs_tea_foldseek.pdf"),
    ("scope_pvalue_correction_comparison_k24.pdf", "065-scope_pvalue_correction_comparison_k24.pdf"),
    ("scope_pvalue_low_seqid_analysis.pdf", "065-scope_pvalue_low_seqid_analysis.pdf"),
    ("scope_pvalue_sensitivity_low_seqid_th30.pdf", "065-scope_pvalue_sensitivity_low_seqid_th30.pdf"),
    ("scope_pvalue_k15_strict_alpha.pdf", "065-scope_pvalue_k15_strict_alpha.pdf"),
    ("scope_pvalue_k16_feature_comparison.pdf", "065-scope_pvalue_k16_feature_comparison.pdf"),
    ("scope_pvalue_k15_metric_comparison.pdf", "065-scope_pvalue_k15_metric_comparison.pdf"),
]


def main():
    # 1. Create figures/ directory
    FIGURES_DIR.mkdir(exist_ok=True)
    print(f"Created {FIGURES_DIR}")

    # 2. Rename notebooks
    print("\n--- Renaming notebooks ---")
    for old_name, new_name in NOTEBOOK_RENAMES:
        old_path = NOTEBOOKS_DIR / old_name
        new_path = NOTEBOOKS_DIR / new_name
        if old_path.exists():
            if new_path.exists() and old_path != new_path:
                print(f"  SKIP (target exists): {old_name} → {new_name}")
            else:
                old_path.rename(new_path)
                print(f"  {old_name} → {new_name}")
        else:
            print(f"  MISSING: {old_name}")

    # 3. Move and rename figures
    print("\n--- Moving figures to figures/ ---")
    for old_name, new_name in FIGURE_RENAMES:
        old_path = NOTEBOOKS_DIR / old_name
        new_path = FIGURES_DIR / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"  {old_name} → figures/{new_name}")
        else:
            print(f"  MISSING: {old_name}")

    # 4. Report any remaining PDFs in notebooks/ not in our mapping
    remaining = sorted(NOTEBOOKS_DIR.glob("*.pdf"))
    if remaining:
        print(f"\n--- {len(remaining)} PDFs remaining in notebooks/ (not in mapping) ---")
        for f in remaining:
            print(f"  {f.name}")
    else:
        print("\nAll PDFs moved successfully.")


if __name__ == "__main__":
    main()
