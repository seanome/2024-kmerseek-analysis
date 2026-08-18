# SCOPE KmerSeek Utilities Function Reference

This document lists all functions available in `scope_kmerseek_utils.py`.

## File I/O Utilities

### `read_csv_with_size_limit(file_path, max_rows=15_000_000, size_threshold_gb=8.0)`
Read a CSV file with automatic row limiting for large files.

### `load_kmerseek_results(results_dir, pattern="*.csv", add_ksize_from_filename=True, max_rows=15_000_000, size_threshold_gb=8.0)`
Load and concatenate multiple KmerSeek result files.

### `extract_ksize_from_filename(filename)`
Extract ksize value from filename (e.g., 'hp.k15' -> 15).

## SCOPe Hierarchical Level Extraction

### `extract_scope_levels(name)`
Extract SCOPe hierarchical levels from protein name.
- Returns dict with keys: 'family', 'superfamily', 'fold', 'class'
- Example: "d1a0a_ a.1.1.1" -> {'family': 'a.1.1.1', 'superfamily': 'a.1.1', 'fold': 'a.1', 'class': 'a'}

### `add_scope_hierarchical_levels(df)`
Add SCOPe hierarchical level columns to dataframe.
- Adds columns for query and target: family, superfamily, fold, class
- Adds match columns indicating if query/target share same level

## Sensitivity Calculation

### `calculate_sensitivity_to_first_fp(df, metric_col, level_match_col, group_by_cols=None)`
Calculate sensitivity until first false positive for each query.
- For each query, targets are ranked by metric_col
- Counts how many true positives are found before the first false positive
- Returns DataFrame with: query_name, sensitivity_to_first_fp, tps_to_first_fp, total_tps

## Data Saving Utilities

### `save_processed_data(df, output_path, compression='snappy')`
Save processed dataframe as Parquet for efficient reuse.

## TEA Benchmark Comparison Utilities

These functions were extracted from notebook 53 and productionized for reuse.

### `parse_scop_simple(df)`
Parse SCOP lineages for query and target from protein names (simplified version).
- Extracts full lineage as family level
- Adds columns: query_lineage, target_lineage, query_family, target_family, same_family

### `compute_sensitivity_curve(df, score_col, scop_level="family")`
Compute sensitivity curve for homology detection.
- Implements "sensitivity until first FP" metric used in TEA, FoldSeek benchmarks
- Returns: (fractions, sensitivities) tuple
- scop_level: "family", "superfamily", or "fold"

### `compute_query_coverage(df, score_col, scop_level="family")`
Compute query coverage metrics for homology detection.
- Returns: (tp_hits, coverages) tuple
- tp_hits: Number of TP hits before first FP for each query
- coverages: Approximate coverage fraction for each query

### `compute_auc(fractions, sensitivities)`
Compute area under sensitivity curve using trapezoidal rule.

### `load_rocx_file(filepath)`
Load a .rocx file from TEA paper benchmark.
- .rocx format contains: NAME, SCOP, FAM, SFAM, FOLD, FAMCNT, SFAMCNT, FOLDCNT, FP

### `kmerseek_to_rocx(df, score_col="jaccard", overlap_threshold=0.0)`
Convert KmerSeek results to TEA .rocx format for benchmarking.
- Computes sensitivity at family, superfamily, and fold levels
- score_col: Column to use for ranking (default: "jaccard")
- overlap_threshold: Minimum overlap_probability to include hits

### `plot_sensitivity_from_rocx(rocx_data, level_col='SFAM')`
Extract sensitivity curve data from .rocx format DataFrame.
- Returns: (fractions, sensitivities) tuple
- level_col: 'FAM', 'SFAM', or 'FOLD'

### `compute_sensitivity_at_threshold(rocx_df, threshold_df, scop_level='SFAM')`
Compute sensitivity AUC for queries in a specific SCOPE threshold dataset.
- Used to analyze performance across different sequence identity thresholds
- Returns: (auc_val, n_queries) tuple

## Usage Example

```python
import polars as pl
from scope_kmerseek_utils import (
    load_kmerseek_results,
    add_scope_hierarchical_levels,
    compute_sensitivity_curve,
    kmerseek_to_rocx,
    compute_auc
)

# Load data
df = load_kmerseek_results('../data/results', pattern='*.hp.k15.csv')

# Add hierarchical levels
df = add_scope_hierarchical_levels(df)

# Compute sensitivity curve
fractions, sensitivities = compute_sensitivity_curve(
    df,
    score_col='jaccard',
    scop_level='family'
)

# Compute AUC
auc_value = compute_auc(fractions, sensitivities)
print(f"AUC: {auc_value:.4f}")

# Convert to TEA format for benchmarking
rocx = kmerseek_to_rocx(df, score_col='jaccard')
```

## Notes

- All functions use Polars DataFrames for efficient data processing
- TEA benchmark functions match the format from the TEA paper for fair comparison
- The default score_col for `kmerseek_to_rocx` is "jaccard" for simple Jaccard similarity
- Functions handle self-hit removal automatically when query_md5 and target_md5 columns exist
