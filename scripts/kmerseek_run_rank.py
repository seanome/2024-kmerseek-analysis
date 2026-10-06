"""The run's rule for ranking the target proteins one kmerseek search hit.

As in evaluate_domain_calls.load_regions: keep a region when its tail probability times
region_search_space times db_n_targets (Bonferroni, capped at 1) is under 0.05, then order
target proteins by their best region_enrichment. Ties go to the lower accession.
"""

from __future__ import annotations

import polars as pl


def ranked_targets(
    rows: pl.DataFrame | pl.LazyFrame, by: list[str] | None = None
) -> pl.LazyFrame:
    """rank (1 = best) and n_ranked per target_acc, within each group of ``by`` columns
    (one group when ``by`` is empty). ``rows`` are region rows carrying target_acc."""
    by = list(by or [])
    n_tests = pl.col("region_search_space").cast(pl.Float64) * pl.col("db_n_targets")
    keep = pl.min_horizontal(pl.col("region_tail_probability") * n_tests, 1.0) < 0.05
    group = by or ["_all"]
    return (
        rows.lazy()
        .with_columns(_all=pl.lit(0))
        .filter(keep)
        .group_by(group + ["target_acc"])
        .agg(pl.col("region_enrichment").max())
        .sort(
            group + ["region_enrichment", "target_acc"],
            descending=[False] * len(group) + [True, False],
        )
        .with_columns(
            rank=pl.int_range(1, pl.len() + 1).over(group),
            n_ranked=pl.len().over(group),
        )
        .drop("_all", strict=False)
    )
