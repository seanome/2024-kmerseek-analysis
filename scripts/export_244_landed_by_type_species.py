#!/usr/bin/env python3
"""Count, per feature type, target species and tool, how the notebook-244 cases were called.

Reads tables/244_hero_candidates.csv (one row per case, ``case_id`` = its 0-based row index)
and tables/244_case_calls.csv (one row per case and tool, nine tools per case). Writes
tables/244_landed_by_type_species_tool.csv, one row per (feature_type, species, tool):

  n_cases     cases of that feature type in that species (every case has all nine tools)
  n_landed    the tool's call on the human protein has IoU >= 0.5 with the feature
  n_spills    less than half of the call is inside the feature
  n_no_call   the tool made no call on the feature
  mean_iou    mean IoU over the cases with a call (no-call cases left out)
  median_iou  median IoU over the same cases

A call with at least half its length inside the feature but IoU below 0.5 is neither
landed nor spilled, so n_landed + n_spills + n_no_call can be less than n_cases. IoU >= 0.5
already puts at least half of the call inside, so no call is counted as both.

IoU is recomputed here from the coordinates, on 1-based inclusive intervals for every tool:
a residue range a..b has length b - a + 1. tables/244_case_calls.csv already stores the
kmerseek calls 1-based inclusive (converted from kmerseek's 0-based end-exclusive regions),
so all nine tools are compared the same way. The ``iou`` column of the calls table must
agree with the recomputed one to within its rounding (3 decimals); the script stops if it does not.

Run with the 2025-kmerseek-analysis env:
    python scripts/export_244_landed_by_type_species.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
CANDIDATES = TAB / "244_hero_candidates.csv"
CALLS = TAB / "244_case_calls.csv"
OUT = TAB / "244_landed_by_type_species_tool.csv"

LANDED_IOU = 0.5
SPILL_INSIDE = 0.5


def closed_overlap(
    start: pl.Expr, end: pl.Expr, feat_start: pl.Expr, feat_end: pl.Expr
) -> dict[str, pl.Expr]:
    """IoU and share of the call inside the feature, both intervals 1-based inclusive.

    Null when the call has no coordinates (min/max_horizontal skip nulls, so without the
    guard a missing call would score IoU 1.0).
    """
    ov = (
        pl.min_horizontal(end, feat_end) - pl.max_horizontal(start, feat_start) + 1
    ).clip(lower_bound=0)
    union = pl.max_horizontal(end, feat_end) - pl.min_horizontal(start, feat_start) + 1
    missing = start.is_null() | end.is_null()
    return {
        "iou_closed": pl.when(missing).then(None).otherwise(ov / union),
        "inside": pl.when(missing).then(None).otherwise(ov / (end - start + 1)),
    }


def per_call(cases: pl.DataFrame, calls: pl.DataFrame) -> pl.DataFrame:
    """One row per (case, tool) with the recomputed IoU and the three outcomes."""
    key = ["query", "feature_type", "feature_start", "feature_end", "species"]
    c = cases.with_row_index("case_id").select(pl.col("case_id").cast(pl.Int64), *key)
    got = calls.select("case_id", *key).unique().sort("case_id")
    assert got.equals(
        c
    ), "244_case_calls.csv and 244_hero_candidates.csv disagree on the cases"

    d = calls.with_columns(
        **closed_overlap(
            pl.col("query_start"),
            pl.col("query_end"),
            pl.col("feature_start"),
            pl.col("feature_end"),
        )
    ).with_columns(
        no_call=pl.col("query_start").is_null(),
        landed=(pl.col("iou_closed") >= LANDED_IOU).fill_null(False),
        spills=(pl.col("inside") < SPILL_INSIDE).fill_null(False),
    )
    # The calls table rounds half to even (Python round), so allow half a unit of the 3rd decimal.
    bad = d.filter((pl.col("iou_closed") - pl.col("iou")).abs() > 0.0005 + 1e-9)
    assert bad.height == 0, bad.select("case_id", "tool", "iou", "iou_closed")
    assert (d["no_call"] == d["iou"].is_null()).all()
    return d


def summarise(d: pl.DataFrame) -> pl.DataFrame:
    tools = d["tool"].unique(maintain_order=True).to_list()
    return (
        d.group_by("feature_type", "species", "tool")
        .agg(
            n_cases=pl.len(),
            n_landed=pl.col("landed").sum(),
            n_spills=pl.col("spills").sum(),
            n_no_call=pl.col("no_call").sum(),
            mean_iou=pl.col("iou_closed").mean(),
            median_iou=pl.col("iou_closed").median(),
        )
        .with_columns(
            pl.col("mean_iou").round(3),
            pl.col("median_iou").round(3),
            tool_order=pl.col("tool").replace_strict(
                {t: i for i, t in enumerate(tools)}, return_dtype=pl.Int64
            ),
        )
        .sort("feature_type", "species", "tool_order")
        .drop("tool_order")
    )


def main():
    cases = pl.read_csv(CANDIDATES, infer_schema_length=None)
    calls = pl.read_csv(CALLS, infer_schema_length=None)
    out = summarise(per_call(cases, calls))
    out.write_csv(OUT)
    print(
        f"wrote {OUT}: {out.height} rows ({out['feature_type'].n_unique()} feature types, "
        f"{out['species'].n_unique()} species, {out['tool'].n_unique()} tools; "
        f"{cases.height} cases)"
    )


if __name__ == "__main__":
    main()
