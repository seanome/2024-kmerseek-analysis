"""
Example script showing how to use scope_kmerseek_utils module.

This demonstrates:
1. Loading CSV files with size-aware reading
2. Extracting SCOPe hierarchical levels
3. Calculating sensitivity until first false positive
4. Saving results as Parquet for later use
"""

from scope_kmerseek_utils import (
    load_kmerseek_results,
    add_scope_hierarchical_levels,
    calculate_sensitivity_to_first_fp,
    save_processed_data,
)
from pathlib import Path


def process_kmerseek_results(results_dir, output_dir, pattern="*hp*.csv", metrics=None):
    """
    Complete workflow for processing KmerSeek results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing CSV result files
    output_dir : str or Path
        Directory to save processed Parquet files
    pattern : str
        Glob pattern for input files
    metrics : dict, optional
        Dictionary mapping metric names to column names
        Default: {'TF-IDF': 'tfidf', 'Jaccard': 'jaccard', 'Max Containment': 'max_containment'}
    """
    if metrics is None:
        metrics = {
            "TF-IDF": "tfidf",
            "Jaccard": "jaccard",
            "Max Containment": "max_containment",
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 1: Loading data")
    print("=" * 80)

    # Load all matching files with automatic size handling
    df = load_kmerseek_results(
        results_dir,
        pattern=pattern,
        add_ksize_from_filename=True,
        max_rows=15_000_000,
        size_threshold_gb=8.0,
    )

    print(f"\nLoaded {df.shape[0]:,} rows with {df.shape[1]} columns")
    print(f"Columns: {df.columns}")

    print("\n" + "=" * 80)
    print("Step 2: Adding SCOPe hierarchical levels")
    print("=" * 80)

    # Add hierarchical level columns
    df = add_scope_hierarchical_levels(df)
    print(f"Added hierarchical level columns")
    print(
        f"New columns: {[c for c in df.columns if 'family' in c or 'fold' in c or 'class' in c]}"
    )

    # Save processed data with hierarchical levels
    processed_file = output_dir / f"processed_with_levels.parquet"
    save_processed_data(df, processed_file)

    print("\n" + "=" * 80)
    print("Step 3: Calculating sensitivity for each metric and level")
    print("=" * 80)

    levels = {
        "Family": "family_match",
        "Superfamily": "superfamily_match",
        "Fold": "fold_match",
    }

    # Calculate sensitivity for each metric and level combination
    for level_name, level_match_col in levels.items():
        print(f"\n{level_name} level:")

        for metric_name, metric_col in metrics.items():
            print(f"  Computing {metric_name}...", end=" ")

            # Calculate sensitivity
            sensitivity_df = calculate_sensitivity_to_first_fp(
                df,
                metric_col=metric_col,
                level_match_col=level_match_col,
                group_by_cols=["ksize"],  # Group by ksize to get separate results per k
            )

            # Save results
            output_file = (
                output_dir
                / f"sensitivity_{level_name.lower()}_{metric_name.lower().replace(' ', '_').replace('-', '_')}.parquet"
            )
            save_processed_data(sensitivity_df, output_file)

            print(f"Saved {sensitivity_df.shape[0]:,} queries to {output_file.name}")

    print("\n" + "=" * 80)
    print("Done! All results saved to:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python example_use_scope_utils.py <results_dir> [output_dir] [pattern]"
        )
        print("\nExample:")
        print(
            "  python example_use_scope_utils.py /path/to/results /path/to/output '*hp*.csv'"
        )
        sys.exit(1)

    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./processed_results"
    pattern = sys.argv[3] if len(sys.argv) > 3 else "*hp*.csv"

    process_kmerseek_results(results_dir, output_dir, pattern)
