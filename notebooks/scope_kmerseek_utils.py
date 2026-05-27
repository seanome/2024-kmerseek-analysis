"""
Utilities for processing SCOPe protein search results from KmerSeek/Sourmash.

This module provides functions to:
1. Load CSV/Parquet files with size-aware reading
2. Extract hierarchical levels from SCOPe protein names
3. Calculate sensitivity until first false positive
4. Save processed results as Parquet files for reuse
5. Convert eval TSV to TEA .rocx benchmark format
6. Load and restrict baseline ROCX files (FoldSeek, TEA)
7. Compute AUC at sequence-identity thresholds
8. Add composite scoring columns for multi-metric evaluation
"""

import polars as pl
from pathlib import Path
from typing import List, Optional, Union, Tuple
import re
import numpy as np
from scipy.integrate import trapezoid


# ---------------------------------------------------------------------------
# SCOPe domain constants
# ---------------------------------------------------------------------------

#: Human-readable descriptions for SCOPe top-level classes.
SCOPE_CLASS_DESCRIPTIONS = {
    'a': 'All alpha proteins',
    'b': 'All beta proteins',
    'c': 'Alpha and beta proteins (a/b) — alternating α/β, e.g. TIM barrel, Rossmann fold',
    'd': 'Alpha and beta proteins (a+b) — segregated α and β regions',
    'e': 'Multi-domain proteins (alpha and beta)',
    'f': 'Membrane and cell surface proteins',
    'g': 'Small proteins',
    'h': 'Coiled coil proteins',
    'i': 'Low resolution protein structures',
    'j': 'Peptides',
    'k': 'Designed proteins',
}

#: Canonical SCOP level ordering used in all plots.
#: Each tuple: (rocx_col, eval_col, label).
#: rocx_col — column name in .rocx baseline files (FAM/SFAM/FOLD only; CLASS absent).
#: eval_col — boolean column name in scope_eval DataFrames.
SCOP_LEVELS: List[Tuple[str, str, str]] = [
    ('FAM',   'same_family',       'Family'),
    ('SFAM',  'same_superfamily',  'Superfamily'),
    ('FOLD',  'same_fold',         'Fold'),
    ('CLASS', 'same_class',        'Class'),
]

#: Score columns present in scope_eval files.
#: Each entry is (column_name, ascending, human_label).
#: ascending=True means lower value = better (p-values).
#: ascending=False means higher value = better (similarity scores).
SCOPE_SCORE_COLS: List[Tuple[str, bool, str]] = [
    # --- Similarity metrics (higher = better) ---
    ('containment',            False, 'Containment'),
    ('max_containment',        False, 'Max Containment'),
    ('jaccard',                False, 'Jaccard'),
    ('query_tfidf',            False, 'TF-IDF'),
    ('enrichment',             False, 'Enrichment'),
    ('mean_matched_kmer_freq', False, 'Mean k-mer Freq'),
    # --- P-value metrics (lower = better) ---
    ('poisson_pvalue',         True,  'Poisson p-value (raw)'),
    ('bonferroni',             True,  'Bonferroni (n=reported, WRONG)'),
    ('bonferroni_correct',     True,  'Bonferroni (n=all pairs, correct)'),
    ('bh',                     True,  'BH (FDR)'),
    ('by',                     True,  'BY'),
    # --- Composite scores added by add_composite_scores_scope() ---
    ('neg_log10_bh',           False, '−log10(BH q)'),
    ('neg_log10_bh_x_cont',    False, '−log10(BH q) × Containment'),
    ('enr_x_cont',             False, 'Enrichment × Containment'),
    ('tfidf_x_cont',           False, 'TF-IDF × Containment'),
]

#: Default benchmark data directory.
BENCH_DIR = Path('/Users/olga/data/scope/results-scope-pvalue-benchmark')

#: Default TEA/FoldSeek baseline directory.
TEA_DIR = Path('/Users/olga/code/2024-kmerseek-analysis/data/tea_scope40_rocx_files')

#: Total number of proteins in SCOPe40 v2.08 (40% identity cutoff).
#: Used to compute correct Bonferroni multiplier = N_SCOPE40 * (N_SCOPE40 - 1).
N_SCOPE40_PROTEINS: int = 15177

#: Correct Bonferroni multiplier: all ordered non-self pairs.
#: The pipeline-computed 'bonferroni' column uses n_reported_pairs as the
#: multiplier, which is ~137,000× too small. Use this for correct statistics.
N_SCOPE40_PAIRS: int = N_SCOPE40_PROTEINS * (N_SCOPE40_PROTEINS - 1)  # 230,326,152

#: Minimal columns needed for any ROCX conversion (reduces memory ~10×).
EVAL_USECOLS: List[str] = [
    'query_domain', 'target_domain', 'q_scop_id',
    'same_class', 'same_family', 'same_superfamily', 'same_fold',
    'containment', 'max_containment', 'jaccard',
    'query_tfidf', 'enrichment', 'mean_matched_kmer_freq',
    'poisson_pvalue', 'bonferroni', 'bh', 'by',
]


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


# ---------------------------------------------------------------------------
# Eval file loading and ROCX conversion (deduplicated from notebook 065)
# ---------------------------------------------------------------------------

def load_eval(
    k: int,
    bench_dir: Union[str, Path] = BENCH_DIR,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Load scope_eval data for a given k-size.

    Prefers .parquet over .tsv.gz over .tsv for speed.
    Pass ``columns=EVAL_USECOLS`` to load only the columns needed for ROCX
    conversion — reduces peak memory ~10× for large files.

    Parameters
    ----------
    k : int
        K-mer size to load.
    bench_dir : Path, optional
        Directory containing scope_eval files.
    columns : list of str, optional
        Column subset to load.  None loads all columns (default).

    Returns
    -------
    pl.DataFrame
        Evaluation pairs with SCOP labels and p-values.
    """
    bench_dir = Path(bench_dir)
    for suffix in [
        f'scope_eval.hp.k{k}.parquet',
        f'scope_eval.hp.k{k}.tsv.gz',
        f'scope_eval.hp.k{k}.tsv',
    ]:
        path = bench_dir / suffix
        if not path.exists():
            continue
        if suffix.endswith('.parquet'):
            return pl.read_parquet(path, columns=columns)
        else:
            print(f'  (parquet not found, reading {suffix} for k={k})')
            df = pl.read_csv(path, separator='\t')
            if columns is not None:
                df = df.select([c for c in columns if c in df.columns])
            return df
    raise FileNotFoundError(f'No scope_eval file found for k={k} in {bench_dir}')


def add_composite_scores_scope(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add composite scoring columns to a scope_eval DataFrame.

    Adds:
    - neg_log10_bh       : -log10(BH q-value), clipped to [-log10(1e-300), 0]
    - neg_log10_bh_x_cont: neg_log10_bh × containment
    - enr_x_cont         : enrichment × containment
    - tfidf_x_cont       : query_tfidf × containment

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns 'bh', 'containment', 'enrichment', 'query_tfidf'.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with composite columns appended.
    """
    bh_clipped = pl.col('bh').clip(lower_bound=1e-300, upper_bound=1.0)
    neg_log10_bh = (-bh_clipped.log(base=10)).alias('neg_log10_bh')

    return df.with_columns([
        neg_log10_bh,
        ((-bh_clipped.log(base=10)) * pl.col('containment')).alias('neg_log10_bh_x_cont'),
        (pl.col('enrichment') * pl.col('containment')).alias('enr_x_cont'),
        (pl.col('query_tfidf') * pl.col('containment')).alias('tfidf_x_cont'),
    ])


def recompute_bonferroni(
    df: pl.DataFrame,
    n_total: int = N_SCOPE40_PAIRS,
    pvalue_col: str = 'poisson_pvalue',
    out_col: str = 'bonferroni_correct',
) -> pl.DataFrame:
    """
    Add a correctly-computed Bonferroni column using all possible pairs.

    The pipeline-generated 'bonferroni' column multiplies the raw p-value by
    the number of *reported* (non-zero) pairs, which is ~137,000× too small
    for the SCOPe40 all-vs-all benchmark (15,177 proteins → 230,326,152 pairs).
    This function computes the correct correction using ``n_total``.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain ``pvalue_col`` (default: 'poisson_pvalue').
    n_total : int, optional
        Total number of comparisons.  Defaults to N_SCOPE40_PAIRS
        (15177 * 15176 = 230,326,152).
    pvalue_col : str, optional
        Raw p-value column name (default: 'poisson_pvalue').
    out_col : str, optional
        Output column name (default: 'bonferroni_correct').

    Returns
    -------
    pl.DataFrame
        Input DataFrame with ``out_col`` added.
    """
    return df.with_columns(
        (pl.col(pvalue_col) * n_total).clip(upper_bound=1.0).alias(out_col)
    )


def eval_tsv_to_rocx(
    df: pl.DataFrame,
    score_col: str = 'bh',
    ascending: bool = True,
    exclude_gray_zone: bool = True,
) -> pl.DataFrame:
    """
    Convert a scope_eval DataFrame to TEA .rocx format.

    Fully vectorized using Polars group_by — no Python loop over queries.
    ~50–200× faster than the previous per-query Python loop and uses a
    fraction of the peak memory.

    The scope_eval file already contains SCOP labels (same_family,
    same_superfamily, same_fold) and corrected p-values (bonferroni, bh, by).
    NOTE: The 'bonferroni' column in scope_eval files uses n_reported_pairs
    as the multiplier; use recompute_bonferroni() for correct statistics.
    This function ranks hits per query by ``score_col`` and computes
    sensitivity-to-first-FP at family, superfamily, and fold levels.

    Sensitivity-to-first-FP for level L:
      - 0.0  if the top-ranked hit is a false positive
      - 1.0  if no false positive exists in the hit list
      - min(position_of_first_FP / total_TPs, 1.0)  otherwise
    All rows before the first FP are true positives by definition.

    Parameters
    ----------
    df : pl.DataFrame
        scope_eval pairs (must have: query_domain, target_domain,
        q_scop_id, same_family, same_superfamily, same_fold, score_col).
    score_col : str, optional
        Column to rank by (default: 'bh').
    ascending : bool, optional
        True → lower values rank first (p-values).
        False → higher values rank first (similarity scores).
    exclude_gray_zone : bool, optional
        If True (default), use the Foldseek/TEA paper convention: FP at each
        level is defined as a hit outside the *next broader* SCOP level, not
        merely outside the current level.  E.g. for SFAM, first FP = first
        cross-fold hit; same-fold-but-different-sfam hits are ignored when
        detecting the FP boundary.  TP counts (n_same) are unchanged.

    Returns
    -------
    pl.DataFrame
        ROCX-format DataFrame with columns:
        NAME, SCOP, FAM, SFAM, FOLD, FP, FAMCNT, SFAMCNT, FOLDCNT.
    """
    _ROCX_SCHEMA = {
        'NAME': pl.Utf8, 'SCOP': pl.Utf8,
        'CLASS': pl.Float64, 'FAM': pl.Float64, 'SFAM': pl.Float64, 'FOLD': pl.Float64,
        'FP': pl.Int32,
        'CLASSCNT': pl.Int32, 'FAMCNT': pl.Int32, 'SFAMCNT': pl.Int32, 'FOLDCNT': pl.Int32,
    }

    df = df.filter(pl.col('query_domain') != pl.col('target_domain'))
    if len(df) == 0:
        return pl.DataFrame(schema=_ROCX_SCHEMA)

    # Sort globally; group_by with maintain_order preserves per-group sort order.
    df_sorted = df.sort(
        ['query_domain', score_col],
        descending=[False, not ascending],
        nulls_last=True,
    )

    # --- Vectorized sensitivity computation --------------------------------
    # Sensitivity-to-first-FP for level L:
    #   n_tp    = count of TPs in ranked list (level-specific, see below)
    #   has_fp  = any FP exists
    #   pos     = rank position of first FP (0-based)
    #   result  → 0 if no TPs, 1 if no FPs, 0 if first hit is FP,
    #             else clip(pos / n_tp, 0, 1)
    #
    # Foldseek/TEA paper definition (exclude_gray_zone=True, default):
    #   TP at FAM:  same_family   (fp boundary = NOT same_fold)
    #   TP at SFAM: same_sfam AND NOT same_fam  (fp boundary = NOT same_fold)
    #   TP at FOLD: same_fold AND NOT same_sfam (fp boundary = NOT same_fold)
    #   → exclusive ranges; all levels share the same FP boundary (cross-fold)
    #
    # Legacy definition (exclude_gray_zone=False):
    #   TP at FAM:  same_family   (fp boundary = NOT same_family)
    #   TP at SFAM: same_sfam     (fp boundary = NOT same_sfam)
    #   TP at FOLD: same_fold     (fp boundary = NOT same_fold)

    _has_class = 'same_class' in df.columns

    if exclude_gray_zone:
        # Exclusive-range TP columns; FP boundary = NOT same_fold for all levels.
        # Sensitivity = count(exclusive TPs before first cross-fold FP) / count(exclusive TPs)
        df_sorted = df_sorted.with_columns([
            (pl.col('same_superfamily') & ~pl.col('same_family')).alias('_tp_sfam'),
            (pl.col('same_fold')        & ~pl.col('same_superfamily')).alias('_tp_fold'),
        ])
        _tp_col = {'fam': 'same_family', 'sfam': '_tp_sfam', 'fold': '_tp_fold'}
        _fp_col = {'fam': 'same_fold',   'sfam': 'same_fold', 'fold': 'same_fold'}
    else:
        _tp_col = {'fam': 'same_family', 'sfam': 'same_superfamily', 'fold': 'same_fold'}
        _fp_col = {'fam': 'same_family', 'sfam': 'same_superfamily', 'fold': 'same_fold'}

    def _sens_expr(tp_col: str, fp_col: str, abbr: str):
        n      = pl.col(tp_col).sum().cast(pl.Int32).alias(f'n_{abbr}')
        fp_any = pl.col(fp_col).not_().any().alias(f'hasfp_{abbr}')
        # Count TPs that appear strictly before the first FP.
        # (~fp_col).cum_sum() == 0 is True for every hit that precedes any FP.
        before_fp = pl.col(fp_col).not_().cum_sum() == 0
        tp_bfp = (
            (pl.col(tp_col) & before_fp).sum().cast(pl.Int32).alias(f'tp_bfp_{abbr}')
        )
        return [n, fp_any, tp_bfp]

    agg_exprs = (
        [pl.col('q_scop_id').first().alias('SCOP'),
         pl.col('same_family').first().alias('_first_is_tp')]
        + (_sens_expr('same_class', 'same_class', 'cls') if _has_class else [])
        + _sens_expr(_tp_col['fam'],  _fp_col['fam'],  'fam')
        + _sens_expr(_tp_col['sfam'], _fp_col['sfam'], 'sfam')
        + _sens_expr(_tp_col['fold'], _fp_col['fold'], 'fold')
    )

    stats = df_sorted.group_by('query_domain', maintain_order=True).agg(agg_exprs)

    def _build_sens(abbr: str) -> pl.Expr:
        n      = pl.col(f'n_{abbr}')
        has    = pl.col(f'hasfp_{abbr}')
        tp_bfp = pl.col(f'tp_bfp_{abbr}').cast(pl.Float64)
        return (
            pl.when(n == 0).then(pl.lit(0.0))
              .when(~has).then(pl.lit(1.0))
              .when(tp_bfp == 0).then(pl.lit(0.0))
              .otherwise((tp_bfp / n.cast(pl.Float64)).clip(0.0, 1.0))
        )

    result = (
        stats
        # Only keep queries that have at least one TP at the family level
        .filter(pl.col('n_fam') > 0)
        .with_columns([
            (_build_sens('cls').alias('CLASS') if _has_class else pl.lit(0.0).alias('CLASS')),
            _build_sens('fam').alias('FAM'),
            _build_sens('sfam').alias('SFAM'),
            _build_sens('fold').alias('FOLD'),
            pl.when(pl.col('_first_is_tp')).then(0).otherwise(1).alias('FP'),
            (pl.col('n_cls').cast(pl.Int32).alias('CLASSCNT') if _has_class else pl.lit(0).cast(pl.Int32).alias('CLASSCNT')),
        ])
        .rename({'query_domain': 'NAME', 'n_fam': 'FAMCNT',
                 'n_sfam': 'SFAMCNT', 'n_fold': 'FOLDCNT'})
        .select(list(_ROCX_SCHEMA.keys()))
    )

    return result


def rocx_restrict(rocx_df: pl.DataFrame, ref_queries: set) -> pl.DataFrame:
    """
    Restrict a ROCX DataFrame to a reference query set.

    Queries in ``ref_queries`` but absent from ``rocx_df`` are assigned
    sensitivity = 0 (they are genuine misses, not excluded).  This ensures
    all methods are evaluated on the same denominator for fair AUC comparison.

    Parameters
    ----------
    rocx_df : pl.DataFrame
        ROCX-format DataFrame with a 'NAME' column.
    ref_queries : set
        Set of query IDs to restrict to (typically FoldSeek's query set).

    Returns
    -------
    pl.DataFrame
        ROCX DataFrame covering exactly ``ref_queries``.
    """
    present = rocx_df.filter(pl.col('NAME').is_in(ref_queries))
    missing_ids = sorted(ref_queries - set(present['NAME'].to_list()))
    if not missing_ids:
        return present

    n = len(missing_ids)
    # Use explicit pl.Series with the dtype taken from `present` to avoid
    # Int64/Int32 mismatches (Python int literals default to Int64 in Polars).
    s = present.schema
    zero_rows = pl.DataFrame([
        pl.Series('NAME',    missing_ids,  dtype=s['NAME']),
        pl.Series('SCOP',    [''] * n,     dtype=s['SCOP']),
        pl.Series('FAM',     [0.0] * n,    dtype=s['FAM']),
        pl.Series('SFAM',    [0.0] * n,    dtype=s['SFAM']),
        pl.Series('FOLD',    [0.0] * n,    dtype=s['FOLD']),
        pl.Series('FP',      [1] * n,      dtype=s['FP']),
        pl.Series('FAMCNT',  [0] * n,      dtype=s['FAMCNT']),
        pl.Series('SFAMCNT', [0] * n,      dtype=s['SFAMCNT']),
        pl.Series('FOLDCNT', [0] * n,      dtype=s['FOLDCNT']),
    ])
    return pl.concat([present, zero_rows])


def sensitivity_stats(
    rocx_df: pl.DataFrame,
    level_col: str = 'SFAM',
) -> Tuple[np.ndarray, list, float]:
    """
    Compute sensitivity curve and AUC from a ROCX DataFrame.

    Parameters
    ----------
    rocx_df : pl.DataFrame
        ROCX-format DataFrame.
    level_col : str, optional
        Column to use: 'FAM', 'SFAM', or 'FOLD' (default: 'SFAM').

    Returns
    -------
    tuple of (np.ndarray, list, float)
        fractions, sensitivities, auc
    """
    frac, sens = plot_sensitivity_from_rocx(rocx_df, level_col=level_col)
    auc = compute_auc(frac, sens)
    return frac, sens, auc



def precision_recall_from_eval(
    eval_df: pl.DataFrame,
    score_col: str,
    ascending: bool,
    level_col: str = 'same_superfamily',
    n_points: int = 100,
    exclude_gray_zone: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute mean precision-recall curve averaged over queries with ≥1 TP.

    Parameters
    ----------
    eval_df : pl.DataFrame
        scope_eval pairs (must contain query_domain, target_domain, level_col,
        same_family, same_superfamily, same_fold, score_col).
    score_col : str
        Column to rank by.
    ascending : bool
        True → lower is better (p-values). False → higher is better (similarity).
    level_col : str
        Boolean column naming the *inclusive* SCOP level:
        'same_family', 'same_superfamily', 'same_fold', or 'same_class'.
    n_points : int
        Number of equally spaced recall points in [0, 1].
    exclude_gray_zone : bool
        If True (default), use the Foldseek/TEA paper TP/FP convention:
          - TPs are the *exclusive* range at level_col:
              same_family          → same_family
              same_superfamily     → same_sfam AND NOT same_fam
              same_fold            → same_fold AND NOT same_sfam
          - FPs are cross-fold hits (NOT same_fold) for all levels.
          - All other hits (gray zone) are excluded from the ranked list
            before computing precision/recall.
        If False, TPs = level_col=True, FPs = level_col=False (inclusive,
        no gray-zone exclusion).

    Returns
    -------
    recall_grid, mean_precision, map_score
    """
    df = eval_df.filter(pl.col('query_domain') != pl.col('target_domain'))

    if exclude_gray_zone and level_col in ('same_family', 'same_superfamily', 'same_fold'):
        # Build exclusive TP column and filter to TP | FP (drop gray zone).
        #   FAM:  TP = same_family,                keep = same_fam | ~same_fold
        #   SFAM: TP = same_sfam & ~same_fam,      keep = (same_sfam & ~same_fam) | ~same_fold
        #   FOLD: TP = same_fold & ~same_sfam,      keep = ~same_sfam  (equivalent)
        if level_col == 'same_family':
            tp_expr   = pl.col('same_family').cast(pl.Boolean)
            keep_expr = pl.col('same_family').cast(pl.Boolean) | ~pl.col('same_fold').cast(pl.Boolean)
        elif level_col == 'same_superfamily':
            tp_expr   = pl.col('same_superfamily').cast(pl.Boolean) & ~pl.col('same_family').cast(pl.Boolean)
            keep_expr = tp_expr | ~pl.col('same_fold').cast(pl.Boolean)
        else:  # same_fold
            tp_expr   = pl.col('same_fold').cast(pl.Boolean) & ~pl.col('same_superfamily').cast(pl.Boolean)
            keep_expr = ~pl.col('same_superfamily').cast(pl.Boolean)

        df = (
            df
            .with_columns(tp_expr.alias('_tp'))
            .filter(keep_expr)
        )
        hits_col = '_tp'
    else:
        hits_col = level_col

    df_sorted = df.sort(
        ['query_domain', score_col],
        descending=[False, not ascending],
        nulls_last=True,
    )

    grouped = (
        df_sorted
        .group_by('query_domain', maintain_order=True)
        .agg(pl.col(hits_col).cast(pl.Boolean).alias('hits'))
    )

    recall_grid = np.linspace(0, 1, n_points)
    per_query_prec: list[np.ndarray] = []

    for row in grouped.iter_rows(named=True):
        hits = np.asarray(row['hits'], dtype=np.float64)
        total_tp = hits.sum()
        if total_tp == 0:
            continue
        ranks   = np.arange(1, len(hits) + 1, dtype=np.float64)
        cum_tp  = np.cumsum(hits)
        prec    = cum_tp / ranks
        rec     = cum_tp / total_tp
        prec_interp = np.interp(recall_grid, rec, prec, left=prec[0], right=0.0)
        per_query_prec.append(prec_interp)

    if not per_query_prec:
        return recall_grid, np.zeros(n_points), 0.0

    mean_prec = np.mean(per_query_prec, axis=0)
    from scipy.integrate import trapezoid as _trapz
    map_score = float(_trapz(mean_prec, recall_grid))
    return recall_grid, mean_prec, map_score


def pr_operating_point_from_rocx(
    rocx_df: pl.DataFrame,
    level_col: str = 'SFAM',
) -> Tuple[float, float]:
    """
    Compute a single mean (recall, precision) operating point from a .rocx DataFrame.

    The .rocx format stores per-query sensitivity at first FP (= recall at first FP).
    At that operating point:
        precision_i = n_tp_before_fp / (n_tp_before_fp + 1)
    where n_tp_before_fp = sensitivity * (CNT - 1)  (CNT includes self).

    Queries with CNT <= 1 (no true positives) are skipped.

    Parameters
    ----------
    rocx_df : pl.DataFrame
        DataFrame loaded from a .rocx file.
    level_col : str
        'FAM', 'SFAM', or 'FOLD'.

    Returns
    -------
    (mean_recall, mean_precision)
    """
    cnt_col = f'{level_col}CNT'
    rows = rocx_df.filter(pl.col(cnt_col) > 1).select([level_col, cnt_col]).iter_rows()
    recalls, precisions = [], []
    for sensitivity, cnt in rows:
        n_tp = sensitivity * (cnt - 1)  # cnt includes self-hit
        recalls.append(sensitivity)
        precisions.append(n_tp / (n_tp + 1) if n_tp > 0 else 0.0)
    if not recalls:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(precisions))


def load_baselines(
    tea_dir: Union[str, Path] = TEA_DIR,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load FoldSeek and TEA baseline ROCX files.

    Parameters
    ----------
    tea_dir : Path, optional
        Directory containing foldseek.rocx and tea_all.rocx.

    Returns
    -------
    tuple of (pl.DataFrame, pl.DataFrame)
        foldseek_rocx, tea_rocx
    """
    tea_dir = Path(tea_dir)
    foldseek = load_rocx_file(tea_dir / 'foldseek.rocx')
    tea_all  = pl.read_csv(tea_dir / 'tea_all.rocx', separator=',')
    return foldseek, tea_all


def compute_all_metric_aucs(
    eval_df: pl.DataFrame,
    ref_queries: set,
    score_cols: Optional[List[Tuple[str, bool, str]]] = None,
    level_col: str = 'SFAM',
) -> List[dict]:
    """
    Compute sensitivity AUC for every scoring metric in ``score_cols``.

    Adds composite scores to ``eval_df`` first, then converts to ROCX format
    for each metric and computes AUC restricted to ``ref_queries``.

    Parameters
    ----------
    eval_df : pl.DataFrame
        scope_eval DataFrame for a single k-size.
    ref_queries : set
        Reference query set (typically FoldSeek's).
    score_cols : list of (col, ascending, label), optional
        Metrics to evaluate.  Defaults to SCOPE_SCORE_COLS.
    level_col : str, optional
        ROCX level column for AUC (default: 'SFAM').

    Returns
    -------
    list of dict
        One dict per metric with keys: score_col, label, auc, n_queries.
    """
    if score_cols is None:
        score_cols = SCOPE_SCORE_COLS

    # Add bonferroni_correct if requested and not already present
    if 'bonferroni_correct' not in eval_df.columns:
        if any(col == 'bonferroni_correct' for col, _, _ in score_cols):
            eval_df = recompute_bonferroni(eval_df)

    # Add composite columns if not present
    composite_cols = {'neg_log10_bh', 'neg_log10_bh_x_cont', 'enr_x_cont', 'tfidf_x_cont'}
    if composite_cols & set(score_cols[c][0] for c in range(len(score_cols))):
        eval_df = add_composite_scores_scope(eval_df)

    results = []
    for col, ascending, label in score_cols:
        if col not in eval_df.columns:
            continue
        rocx = eval_tsv_to_rocx(eval_df, score_col=col, ascending=ascending)
        rocx_r = rocx_restrict(rocx, ref_queries)
        _, _, auc = sensitivity_stats(rocx_r, level_col)
        results.append({
            'score_col': col,
            'label':     label,
            'auc':       round(auc, 5),
            'n_queries': len(rocx_r),
        })
    return results


# ---------------------------------------------------------------------------
# Alignment display helpers
# ---------------------------------------------------------------------------

#: BLOSUM62 positive-score residue pairs (conservative substitutions).
_BLOSUM62_POS: frozenset = frozenset(
    {frozenset(p) for p in [
        ('A','S'),('A','T'),('A','G'),('D','E'),('D','N'),('E','Q'),('E','K'),
        ('N','S'),('N','T'),('Q','K'),('R','K'),('I','L'),('I','V'),('I','M'),
        ('L','V'),('L','M'),('F','Y'),('F','W'),('Y','W'),('H','N'),('H','Q'),
        ('S','T'),('V','M'),
    ]}
    | {frozenset((aa, aa)) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
)


def pct_id_sim(q_seq: str, t_seq: str) -> tuple:
    """Return (% identity, % similarity) over the aligned window."""
    n = min(len(q_seq), len(t_seq))
    if n == 0:
        return 0.0, 0.0
    n_id  = sum(q == t for q, t in zip(q_seq, t_seq))
    n_sim = sum(frozenset((q.upper(), t.upper())) in _BLOSUM62_POS
                for q, t in zip(q_seq, t_seq))
    return 100.0 * n_id / n, 100.0 * n_sim / n


def print_kmer_alignments_by_class(
    eval_df: "pl.DataFrame",
    query_names: set,
    dom_map: dict,
    scop_map: dict,
    overlap_disorder: Optional[dict] = None,
    sort_by: str = 'poisson_pvalue',
    best_hit_only: bool = True,
    width: int = 80,
    class_descriptions: Optional[dict] = None,
) -> None:
    """Print HP k-mer alignment blocks grouped by SCOPe class.

    Parameters
    ----------
    eval_df : pl.DataFrame
        Filtered eval DataFrame with columns: query_domain, target_domain,
        query_subseq, target_subseq, moltype_seq, query_start, target_start,
        poisson_pvalue, jaccard, containment, region_length, q_scop_id.
    query_names : set
        Set of query domain IDs to include.
    dom_map : dict
        {domain_id → protein name}
    scop_map : dict
        {domain_id → SCOP lineage string}
    overlap_disorder : dict, optional
        {domain_id → np.ndarray of per-residue disorder scores for the overlap region}
    sort_by : str
        Column to sort hits within each query (default 'poisson_pvalue').
    best_hit_only : bool
        If True, show only the best hit per query (default True).
    width : int
        Characters per alignment row (default 80).
    class_descriptions : dict, optional
        {class_char → description}. Defaults to SCOPE_CLASS_DESCRIPTIONS.
    """
    import numpy as np

    if class_descriptions is None:
        class_descriptions = SCOPE_CLASS_DESCRIPTIONS

    hits = eval_df.filter(pl.col('query_domain').is_in(query_names))
    if 'query_subseq' in hits.columns:
        hits = hits.filter(pl.col('query_subseq').is_not_null())

    if best_hit_only:
        hits = (hits.sort(sort_by, descending=(sort_by == 'jaccard'))
                    .group_by('query_domain')
                    .agg([pl.first(c) for c in hits.columns if c != 'query_domain']))

    # Attach SCOPe class
    scop_col = 'q_scop_id' if 'q_scop_id' in hits.columns else None
    rows = hits.with_columns(
        pl.col('query_domain')
          .map_elements(
              lambda d: (scop_map.get(d, '') if scop_col is None
                         else None),
              return_dtype=pl.Utf8)
          .alias('_qscop_fallback')
    )
    if scop_col:
        rows = rows.with_columns(
            pl.when(pl.col(scop_col).is_not_null() & (pl.col(scop_col) != ''))
              .then(pl.col(scop_col))
              .otherwise(pl.col('_qscop_fallback'))
              .alias('query_scop')
        )
    else:
        rows = rows.with_columns(pl.col('_qscop_fallback').alias('query_scop'))
    rows = rows.drop('_qscop_fallback')
    rows = rows.with_columns(
        pl.col('query_scop').str.slice(0, 1).alias('scop_class')
    ).sort(['scop_class', 'query_domain'])

    for cls in sorted(rows['scop_class'].unique().to_list()):
        cls_desc  = class_descriptions.get(cls, cls)
        cls_rows  = rows.filter(pl.col('scop_class') == cls)
        print()
        print('=' * 100)
        print(f'  SCOPe CLASS {cls.upper()} — {cls_desc}  ({cls_rows.height} queries)')
        print('=' * 100)

        for row in cls_rows.iter_rows(named=True):
            qid    = row['query_domain']
            tid    = row['target_domain']
            q_scop = row.get('query_scop', scop_map.get(qid, '?'))
            t_scop = scop_map.get(tid, '?')
            q_prot = dom_map.get(qid, '?')
            t_prot = dom_map.get(tid, '?')

            print(f'  QUERY:  {qid}  SCOP={q_scop}  {q_prot}')
            print(f'  TARGET: {tid}  SCOP={t_scop}  {t_prot}')

            q_seq = row.get('query_subseq')  or ''
            t_seq = row.get('target_subseq') or ''
            hp    = row.get('moltype_seq')   or ''
            pid, psim = pct_id_sim(q_seq, t_seq)

            ov_val = overlap_disorder.get(qid) if overlap_disorder else None
            if ov_val is None:
                ov_str = ''
            elif np.isscalar(ov_val):
                ov_str = f'  disorder={ov_val:.2f}'
            elif len(ov_val) > 0:
                ov_str = f'  disorder={np.mean(ov_val):.2f} (max={np.max(ov_val):.2f})'
            else:
                ov_str = ''

            pval = row.get('poisson_pvalue', float('nan'))
            jac  = row.get('jaccard',        float('nan'))
            cont = row.get('containment',    float('nan'))
            rlen = row.get('region_length',  len(q_seq))
            print(f'  pval={pval:.2e}  jaccard={jac:.4f}  containment={cont:.4f}'
                  f'  region_len={rlen}  %id={pid:.1f}%  %sim={psim:.1f}%{ov_str}')
            print()

            q_start = row.get('query_start',  0)
            t_start = row.get('target_start', 0)
            for start in range(0, max(len(q_seq), len(t_seq), len(hp)), width):
                chunk_q = q_seq[start:start + width]
                chunk_t = t_seq[start:start + width]
                match   = ''.join(
                    '|' if a == b
                    else ':' if frozenset((a.upper(), b.upper())) in _BLOSUM62_POS
                    else '.'
                    for a, b in zip(chunk_q, chunk_t)
                )
                spacer = ' ' * len(f'    Q[{q_start + start:>4}] ')
                print(f'    Q[{q_start + start:>4}] {chunk_q}')
                print(f'{spacer}{match}')
                print(f'    T[{t_start + start:>4}] {chunk_t}')
                print(f'    HP      {hp[start:start + width]}')
                print()

            print('  ' + '-' * 98)
