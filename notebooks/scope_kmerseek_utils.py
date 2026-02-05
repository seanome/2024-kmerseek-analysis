"""
Utilities for processing SCOPe protein search results from KmerSeek/Sourmash.

This module provides functions to:
1. Load CSV/Parquet files with size-aware reading
2. Extract hierarchical levels from SCOPe protein names
3. Calculate sensitivity until first false positive
4. Save processed results as Parquet files for reuse
"""

import polars as pl
from pathlib import Path
from typing import List, Optional, Union, Tuple
import re
import numpy as np
from scipy.integrate import trapezoid


# File I/O utilities
# ==================

def read_csv_with_size_limit(
    file_path: Union[str, Path],
    max_rows: int = 15_000_000,
    size_threshold_gb: float = 8.0
) -> pl.DataFrame:
    """
    Read a CSV file with automatic row limiting for large files.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file
    max_rows : int, optional
        Maximum number of rows to read for large files (default: 15M)
    size_threshold_gb : float, optional
        File size threshold in GB to trigger row limiting (default: 8.0)

    Returns
    -------
    pl.DataFrame
        Loaded dataframe
    """
    file_path = Path(file_path)
    file_size_gb = file_path.stat().st_size / (1024**3)

    if file_size_gb > size_threshold_gb:
        print(f"Loading {file_path.name} ({file_size_gb:.2f} GB) - reading first {max_rows:,} rows...")
        return pl.read_csv(file_path, n_rows=max_rows)
    else:
        print(f"Loading {file_path.name} ({file_size_gb:.2f} GB) - reading all rows...")
        return pl.read_csv(file_path)


def load_kmerseek_results(
    results_dir: Union[str, Path],
    pattern: str = "*.csv",
    add_ksize_from_filename: bool = True,
    max_rows: int = 15_000_000,
    size_threshold_gb: float = 8.0
) -> pl.DataFrame:
    """
    Load and concatenate multiple KmerSeek result files.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing result files
    pattern : str, optional
        Glob pattern for files to load (default: "*.csv")
    add_ksize_from_filename : bool, optional
        Whether to extract ksize from filename and add as column (default: True)
    max_rows : int, optional
        Maximum rows to read from large files (default: 15M)
    size_threshold_gb : float, optional
        Size threshold for row limiting (default: 8.0 GB)

    Returns
    -------
    pl.DataFrame
        Combined dataframe from all matching files
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(pattern))

    if not files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {results_dir}")

    print(f"Found {len(files)} files matching '{pattern}'")

    dfs = []
    for file_path in files:
        df = read_csv_with_size_limit(file_path, max_rows, size_threshold_gb)

        # Extract ksize from filename if requested
        if add_ksize_from_filename:
            ksize = extract_ksize_from_filename(file_path.stem)
            if ksize is not None:
                df = df.with_columns(pl.lit(ksize).alias('ksize'))

        dfs.append(df)
        print(f"  Loaded {df.shape[0]:,} rows from {file_path.name}")

    combined = pl.concat(dfs)
    print(f"\nCombined data shape: {combined.shape}")
    return combined


def extract_ksize_from_filename(filename: str) -> Optional[int]:
    """
    Extract ksize value from filename.

    Examples: 'hp.k15' -> 15, 'dayhoff.k20.csv' -> 20

    Parameters
    ----------
    filename : str
        Filename or stem to extract ksize from

    Returns
    -------
    int or None
        Extracted ksize or None if not found
    """
    match = re.search(r'\.k(\d+)', filename)
    return int(match.group(1)) if match else None


# SCOPe hierarchical level extraction
# =====================================

def extract_scope_levels(name: str) -> Optional[dict]:
    """
    Extract SCOPe hierarchical levels from protein name.

    SCOPe format: d[SCOP_ID] [lineage]
    Example: "d1a0a_ a.1.1.1" -> family=a.1.1, superfamily=a.1.1, fold=a.1, class=a

    Parameters
    ----------
    name : str
        Protein name in SCOPe format

    Returns
    -------
    dict or None
        Dictionary with keys: 'family', 'superfamily', 'fold', 'class'
        or None if extraction fails
    """
    # SCOPe lineage pattern: class.fold.superfamily.family
    # Example: a.1.1.1 means class=a, fold=a.1, superfamily=a.1.1, family=a.1.1.1

    # Split by space to separate ID from lineage
    parts = name.split()
    if len(parts) < 2:
        return None

    lineage = parts[1]

    # Parse lineage using regex
    # Pattern: (?P<class>[a-z])\.(?P<fold_num>\d+)\.(?P<superfam_num>\d+)\.(?P<fam_num>\d+)
    pattern = r'(?P<class>[a-z])\.(?P<fold_num>\d+)\.(?P<superfam_num>\d+)\.(?P<fam_num>\d+)'
    match = re.match(pattern, lineage)

    if not match:
        return None

    cls = match.group('class')
    fold_num = match.group('fold_num')
    superfam_num = match.group('superfam_num')
    fam_num = match.group('fam_num')

    return {
        'class': cls,
        'fold': f"{cls}.{fold_num}",
        'superfamily': f"{cls}.{fold_num}.{superfam_num}",
        'family': f"{cls}.{fold_num}.{superfam_num}.{fam_num}"
    }


def add_scope_hierarchical_levels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add SCOPe hierarchical level columns to dataframe.

    Adds columns for query and target: family, superfamily, fold, class
    Also adds match columns indicating if query/target share same level.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe with 'query_name' and 'target_name' columns

    Returns
    -------
    pl.DataFrame
        Dataframe with added hierarchical level and match columns
    """
    # Define function for extraction
    def extract_scope_dict(name_series: pl.Series) -> pl.Series:
        return name_series.map_elements(extract_scope_levels, return_dtype=pl.Struct)

    # Extract query levels
    query_levels = extract_scope_dict(df['query_name']).alias('query_levels')
    target_levels = extract_scope_dict(df['target_name']).alias('target_levels')

    # Add all level columns
    df = df.with_columns([
        query_levels.struct.field('family').alias('query_family'),
        query_levels.struct.field('superfamily').alias('query_superfamily'),
        query_levels.struct.field('fold').alias('query_fold'),
        query_levels.struct.field('class').alias('query_class'),
        target_levels.struct.field('family').alias('target_family'),
        target_levels.struct.field('superfamily').alias('target_superfamily'),
        target_levels.struct.field('fold').alias('target_fold'),
        target_levels.struct.field('class').alias('target_class'),
    ])

    # Add match indicators (True Positive vs False Positive)
    df = df.with_columns([
        (pl.col('query_family') == pl.col('target_family')).alias('family_match'),
        (pl.col('query_superfamily') == pl.col('target_superfamily')).alias('superfamily_match'),
        (pl.col('query_fold') == pl.col('target_fold')).alias('fold_match'),
        (pl.col('query_class') == pl.col('target_class')).alias('class_match'),
    ])

    return df


# Sensitivity calculation
# =======================

def calculate_sensitivity_to_first_fp(
    df: pl.DataFrame,
    metric_col: str,
    level_match_col: str,
    group_by_cols: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Calculate sensitivity until first false positive for each query.

    For each query, targets are ranked by metric_col (descending = better).
    Counts how many true positives are found before the first false positive.

    This implementation is based on the correct code from notebook 53.

    Parameters
    ----------
    df : pl.DataFrame
        Data with metric and match columns
    metric_col : str
        Column to rank by (e.g., 'tfidf', 'jaccard', 'max_containment')
    level_match_col : str
        Column indicating TP/FP (e.g., 'family_match', 'superfamily_match')
    group_by_cols : list of str, optional
        Additional columns to group by (e.g., ['ksize'])

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: query_name, sensitivity_to_first_fp,
        tps_to_first_fp, total_tps (and any group_by_cols)
    """
    if group_by_cols is None:
        group_by_cols = []

    # Remove self-hits if query_md5 and target_md5 columns exist
    if 'query_md5' in df.columns and 'target_md5' in df.columns:
        df = df.filter(pl.col('query_md5') != pl.col('target_md5'))

    # Define group columns
    group_cols = ['query_name'] + group_by_cols

    # Get unique queries
    queries_df = df.select(group_cols).unique()

    results = []

    # Iterate through each group
    for group_key in queries_df.iter_rows():
        # Build filter conditions
        if len(group_cols) == 1:
            query_name = group_key[0]
            filter_cond = pl.col('query_name') == query_name
            group_dict = {'query_name': query_name}
        else:
            # Multiple grouping columns
            filter_cond = pl.lit(True)
            group_dict = {}
            for i, col_name in enumerate(group_cols):
                filter_cond = filter_cond & (pl.col(col_name) == group_key[i])
                group_dict[col_name] = group_key[i]

        # Filter and sort by metric (descending = better scores first)
        query_df = df.filter(filter_cond).sort(metric_col, descending=True)

        if len(query_df) == 0:
            continue

        # Get match values
        same_vals = query_df[level_match_col].to_list()

        # Count total positives for this query
        n_positives = sum(same_vals)

        if n_positives == 0:
            continue

        # Find first false positive
        first_fp = next((i for i, v in enumerate(same_vals) if not v), None)

        if first_fp is None:
            # No false positives - retrieved all positives
            sensitivity = 1.0
        elif first_fp == 0:
            # First hit is FP
            sensitivity = 0.0
        else:
            # Sensitivity = fraction of positives retrieved before first FP
            sensitivity = min(first_fp / n_positives, 1.0)

        # Store result
        result = {
            **group_dict,
            'sensitivity_to_first_fp': sensitivity,
            'tps_to_first_fp': first_fp if first_fp is not None else n_positives,
            'total_tps': n_positives
        }
        results.append(result)

    return pl.DataFrame(results)


# Data saving utilities
# =====================

def save_processed_data(
    df: pl.DataFrame,
    output_path: Union[str, Path],
    compression: str = 'snappy'
) -> None:
    """
    Save processed dataframe as Parquet for efficient reuse.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe to save
    output_path : str or Path
        Output file path (should end in .parquet or .pq)
    compression : str, optional
        Compression method (default: 'snappy')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    df.write_parquet(output_path, compression=compression)
    print(f"Saved {df.shape[0]:,} rows")


# TEA benchmark comparison utilities
# ===================================

def parse_scop_simple(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse SCOP lineages for query and target from protein names.

    This is a simplified version that extracts the full lineage as the family level.
    For more detailed hierarchical parsing, use add_scope_hierarchical_levels().

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with 'query_name' and 'target_name' columns in format:
        "d[SCOP_ID] [lineage]" (e.g., "d1a0a_ a.1.1.1")

    Returns
    -------
    pl.DataFrame
        DataFrame with added columns:
        - query_lineage, target_lineage: Full SCOP lineage
        - query_family, target_family: Family level (same as lineage)
        - same_family: Boolean indicating if query and target are in same family
    """
    df = df.with_columns([
        pl.col("query_name").str.split(" ").list.get(1).alias("query_lineage"),
        pl.col("target_name").str.split(" ").list.get(1).alias("target_lineage"),
    ])

    # Extract family (full lineage)
    df = df.with_columns([
        pl.col("query_lineage").alias("query_family"),
        pl.col("target_lineage").alias("target_family"),
    ])

    # Check if same family
    df = df.with_columns([
        (pl.col("query_family") == pl.col("target_family")).alias("same_family")
    ])

    return df


def compute_sensitivity_curve(
    df: pl.DataFrame,
    score_col: str,
    scop_level: str = "family"
) -> Tuple[np.ndarray, list]:
    """
    Compute sensitivity curve for homology detection.

    For each query, calculates the fraction of true positives retrieved
    before encountering the first false positive, then aggregates across
    all queries to create a sensitivity curve.

    This implements the "sensitivity until first FP" metric used in
    TEA, FoldSeek, and other homology detection benchmarks.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with query/target pairs and similarity scores
        Must have columns: query_name, target_name, query_md5, target_md5, score_col
    score_col : str
        Column name to use for ranking (higher = more similar)
    scop_level : str, optional
        SCOP level for defining homology: "family", "superfamily", or "fold"
        Default: "family"

    Returns
    -------
    tuple of (np.ndarray, list)
        - fractions: Array of query fractions (x-axis)
        - sensitivities: List of sensitivity values (y-axis)

    Notes
    -----
    The sensitivity curve shows what fraction of queries can retrieve
    a given fraction of their homologs before hitting a false positive.
    Higher curves indicate better performance.
    """
    # Parse SCOP lineages
    df = df.with_columns([
        pl.col("query_name").str.split(" ").list.get(1).alias("query_lineage"),
        pl.col("target_name").str.split(" ").list.get(1).alias("target_lineage"),
    ])

    # Extract SCOP level
    if scop_level == "family":
        df = df.with_columns([
            pl.col("query_lineage").alias("query_scop"),
            pl.col("target_lineage").alias("target_scop"),
        ])
    elif scop_level == "superfamily":
        parts_q = pl.col("query_lineage").str.split(".")
        parts_t = pl.col("target_lineage").str.split(".")
        df = df.with_columns([
            (parts_q.list.get(0) + pl.lit(".") + parts_q.list.get(1) + pl.lit(".") + parts_q.list.get(2)).alias("query_scop"),
            (parts_t.list.get(0) + pl.lit(".") + parts_t.list.get(1) + pl.lit(".") + parts_t.list.get(2)).alias("target_scop"),
        ])
    elif scop_level == "fold":
        parts_q = pl.col("query_lineage").str.split(".")
        parts_t = pl.col("target_lineage").str.split(".")
        df = df.with_columns([
            (parts_q.list.get(0) + pl.lit(".") + parts_q.list.get(1)).alias("query_scop"),
            (parts_t.list.get(0) + pl.lit(".") + parts_t.list.get(1)).alias("target_scop"),
        ])

    df = df.with_columns([
        (pl.col("query_scop") == pl.col("target_scop")).alias("same_scop")
    ])

    # Remove self-hits
    df = df.filter(pl.col("query_md5") != pl.col("target_md5"))

    # Get unique queries
    queries = df["query_name"].unique().to_list()

    sensitivities = []
    for query in queries:
        qdf = df.filter(pl.col("query_name") == query).sort(score_col, descending=True)

        if len(qdf) == 0:
            continue

        same_vals = qdf["same_scop"].to_list()

        # Count total positives for this query
        n_positives = sum(same_vals)

        if n_positives == 0:
            continue

        # Find first false positive
        first_fp = next((i for i, v in enumerate(same_vals) if not v), None)

        if first_fp is None:
            # No false positives - retrieved all positives
            sensitivity = 1.0
        elif first_fp == 0:
            # First hit is FP
            sensitivity = 0.0
        else:
            # Sensitivity = fraction of positives retrieved before first FP
            sensitivity = min(first_fp / n_positives, 1.0)

        sensitivities.append(sensitivity)

    # Sort sensitivities descending
    sensitivities = sorted(sensitivities, reverse=True)

    # Create fraction of queries
    fractions = np.arange(len(sensitivities)) / len(sensitivities)

    return fractions, sensitivities


def compute_query_coverage(
    df: pl.DataFrame,
    score_col: str,
    scop_level: str = "family"
) -> Tuple[list, list]:
    """
    Compute query coverage metrics for homology detection.

    For each query, computes the number of true positive hits and
    approximate coverage before encountering the first false positive.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with query/target pairs and similarity scores
    score_col : str
        Column name to use for ranking (higher = more similar)
    scop_level : str, optional
        SCOP level for defining homology: "family" or "superfamily"
        Default: "family"

    Returns
    -------
    tuple of (list, list)
        - tp_hits: Number of TP hits before first FP for each query
        - coverages: Approximate coverage fraction for each query
    """
    # Parse SCOP
    df = df.with_columns([
        pl.col("query_name").str.split(" ").list.get(1).alias("query_lineage"),
        pl.col("target_name").str.split(" ").list.get(1).alias("target_lineage"),
    ])

    # Extract SCOP level
    if scop_level == "family":
        df = df.with_columns([
            pl.col("query_lineage").alias("query_scop"),
            pl.col("target_lineage").alias("target_scop"),
        ])
    elif scop_level == "superfamily":
        parts_q = pl.col("query_lineage").str.split(".")
        parts_t = pl.col("target_lineage").str.split(".")
        df = df.with_columns([
            (parts_q.list.get(0) + pl.lit(".") + parts_q.list.get(1) + pl.lit(".") + parts_q.list.get(2)).alias("query_scop"),
            (parts_t.list.get(0) + pl.lit(".") + parts_t.list.get(1) + pl.lit(".") + parts_t.list.get(2)).alias("target_scop"),
        ])

    df = df.with_columns([
        (pl.col("query_scop") == pl.col("target_scop")).alias("same_scop")
    ])

    # Remove self-hits
    df = df.filter(pl.col("query_md5") != pl.col("target_md5"))

    queries = df["query_name"].unique().to_list()

    tp_hits = []
    coverages = []

    for query in queries:
        qdf = df.filter(pl.col("query_name") == query).sort(score_col, descending=True)

        if len(qdf) == 0:
            continue

        same_vals = qdf["same_scop"].to_list()

        # Find first FP
        first_fp = next((i for i, v in enumerate(same_vals) if not v), len(same_vals))

        # Get TPs before first FP
        tp_count = sum(same_vals[:first_fp])

        # Calculate coverage (simple approximation)
        if first_fp > 0:
            coverage = min(first_fp / (len(same_vals) + 1), 1.0)
        else:
            coverage = 0.0

        tp_hits.append(tp_count)
        coverages.append(coverage)

    return tp_hits, coverages


def compute_auc(fractions: np.ndarray, sensitivities: list) -> float:
    """
    Compute area under sensitivity curve using trapezoidal rule.

    Parameters
    ----------
    fractions : np.ndarray
        X-axis values (fraction of queries)
    sensitivities : list
        Y-axis values (sensitivity scores)

    Returns
    -------
    float
        Area under the curve
    """
    return trapezoid(sensitivities, fractions)


def load_rocx_file(filepath: Union[str, Path]) -> pl.DataFrame:
    """
    Load a .rocx file from TEA paper benchmark.

    The .rocx format is a tab-separated file containing:
    - NAME: Query ID
    - SCOP: SCOP lineage
    - FAM, SFAM, FOLD: Sensitivity at each level
    - FAMCNT, SFAMCNT, FOLDCNT: Count of homologs at each level
    - FP: Whether first hit was a false positive

    Parameters
    ----------
    filepath : str or Path
        Path to .rocx file

    Returns
    -------
    pl.DataFrame
        Loaded benchmark data
    """
    return pl.read_csv(filepath, separator="\t")


def kmerseek_to_rocx(
    df: pl.DataFrame,
    score_col: str = "jaccard",
    overlap_threshold: float = 0.0
) -> pl.DataFrame:
    """
    Convert KmerSeek results to TEA .rocx format for benchmarking.

    Computes sensitivity at family, superfamily, and fold levels for
    each query, matching the format used in TEA paper benchmarks.

    Parameters
    ----------
    df : pl.DataFrame
        KmerSeek results with columns: query_name, target_name,
        query_md5, target_md5, score columns
    score_col : str, optional
        Column to use for ranking hits (default: "jaccard")
    overlap_threshold : float, optional
        Minimum overlap_probability to include hits (default: 0.0)

    Returns
    -------
    pl.DataFrame
        DataFrame in .rocx format with columns:
        - NAME: Query SCOP ID
        - SCOP: SCOP lineage
        - FAM, SFAM, FOLD: Sensitivity at each level
        - FAMCNT, SFAMCNT, FOLDCNT: Count of homologs
        - FP: 1 if first hit is false positive, 0 otherwise
    """
    # Filter by overlap probability if threshold > 0
    if overlap_threshold > 0:
        df = df.filter(pl.col("overlap_probability") > overlap_threshold)

    # Parse SCOP lineages
    df = df.with_columns([
        pl.col("query_name").str.split(" ").list.get(0).alias("query_scop_id"),
        pl.col("query_name").str.split(" ").list.get(1).alias("query_lineage"),
        pl.col("target_name").str.split(" ").list.get(1).alias("target_lineage"),
    ])

    # Extract SCOP levels
    parts_q = pl.col("query_lineage").str.split(".")
    parts_t = pl.col("target_lineage").str.split(".")

    df = df.with_columns([
        pl.col("query_lineage").alias("query_family"),
        pl.col("target_lineage").alias("target_family"),
        (parts_q.list.get(0) + pl.lit(".") + parts_q.list.get(1) + pl.lit(".") + parts_q.list.get(2)).alias("query_superfamily"),
        (parts_t.list.get(0) + pl.lit(".") + parts_t.list.get(1) + pl.lit(".") + parts_t.list.get(2)).alias("target_superfamily"),
        (parts_q.list.get(0) + pl.lit(".") + parts_q.list.get(1)).alias("query_fold"),
        (parts_t.list.get(0) + pl.lit(".") + parts_t.list.get(1)).alias("target_fold"),
    ])

    # Remove self-hits
    df = df.filter(pl.col("query_md5") != pl.col("target_md5"))

    queries = df["query_scop_id"].unique().to_list()

    results = []
    for query in queries:
        qdf = df.filter(pl.col("query_scop_id") == query).sort(score_col, descending=True)

        if len(qdf) == 0:
            continue

        row = {"NAME": query, "SCOP": qdf["query_lineage"][0]}

        # For each SCOP level, compute sensitivity
        for level_name, q_col, t_col in [
            ("FAM", "query_family", "target_family"),
            ("SFAM", "query_superfamily", "target_superfamily"),
            ("FOLD", "query_fold", "target_fold")
        ]:
            same_vals = (qdf[q_col] == qdf[t_col]).to_list()
            n_same = sum(same_vals)

            if n_same == 0:
                row[level_name] = 0
                row[f"{level_name}CNT"] = 0
            else:
                first_fp = next((i for i, v in enumerate(same_vals) if not v), None)

                if first_fp is None:
                    sensitivity = 1.0
                elif first_fp == 0:
                    sensitivity = 0.0
                else:
                    sensitivity = min(first_fp / n_same, 1.0)

                row[level_name] = sensitivity
                row[f"{level_name}CNT"] = n_same + 1  # +1 for self

        # FP: 1 if first hit is FP
        row["FP"] = 0 if (qdf["query_family"][0] == qdf["target_family"][0]) else 1

        results.append(row)

    return pl.DataFrame(results)


def plot_sensitivity_from_rocx(
    rocx_data: pl.DataFrame,
    level_col: str = 'SFAM'
) -> Tuple[np.ndarray, list]:
    """
    Extract sensitivity curve data from .rocx format DataFrame.

    Parameters
    ----------
    rocx_data : pl.DataFrame
        DataFrame in .rocx format
    level_col : str, optional
        SCOP level column: 'FAM', 'SFAM', or 'FOLD' (default: 'SFAM')

    Returns
    -------
    tuple of (np.ndarray, list)
        - fractions: Array of query fractions (x-axis)
        - sensitivities: List of sensitivity values (y-axis)
    """
    sensitivities = rocx_data[level_col].sort(descending=True).to_list()
    fractions = np.arange(len(sensitivities)) / len(sensitivities)
    return fractions, sensitivities


def compute_sensitivity_at_threshold(
    rocx_df: pl.DataFrame,
    threshold_df: pl.DataFrame,
    scop_level: str = 'SFAM'
) -> Tuple[float, int]:
    """
    Compute sensitivity AUC for queries in a specific SCOPE threshold dataset.

    Used to analyze performance across different sequence identity thresholds
    (e.g., SCOPE40 10%, 20%, 25%, etc.).

    Parameters
    ----------
    rocx_df : pl.DataFrame
        Results in .rocx format (from TEA or KmerSeek)
    threshold_df : pl.DataFrame
        SCOPE threshold dataset with 'sid' column containing query IDs
    scop_level : str, optional
        SCOP level column: 'FAM', 'SFAM', or 'FOLD' (default: 'SFAM')

    Returns
    -------
    tuple of (float, int)
        - auc_val: Area under the sensitivity curve
        - n_queries: Number of queries in this threshold
    """
    # Get set of query IDs in this threshold dataset
    threshold_sids = set(threshold_df['sid'].to_list())

    # Filter rocx to only queries in this threshold
    filtered_rocx = rocx_df.filter(pl.col('NAME').is_in(threshold_sids))

    if len(filtered_rocx) == 0:
        return 0.0, 0

    # Compute sensitivity curve
    sensitivities = filtered_rocx[scop_level].sort(descending=True).to_list()
    fractions = np.arange(len(sensitivities)) / len(sensitivities)

    # Compute AUC
    if len(sensitivities) > 0:
        auc_val = trapezoid(sensitivities, fractions)
    else:
        auc_val = 0.0

    return auc_val, len(filtered_rocx)
