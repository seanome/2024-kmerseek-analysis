#!/usr/bin/env python3
"""Concatenate every per-(tool, variant, species) metrics row and PR/ROC curve.

Emits parquet for notebooks and CSV for reading at the terminal, plus a leaderboard so
the sweep's headline is visible without opening anything.
"""

import sys
from pathlib import Path

import polars as pl

LEAD = ["truth_set", "tool", "variant", "species", "split", "stratum_axis", "stratum"]

# Threshold-free and therefore comparable across tools with different default cutoffs.
HEADLINE = ["fmax", "auprc", "roc_auc", "smin", "ndo", "recall_reachable", "precision"]

# The leaderboard cut. `heldout` because the sweep picks its best combo on `selection`,
# and scoring the winner on the data that chose it is optimistically biased; `all`
# stratum because the per-axis cuts answer a different question.
LEADERBOARD_SPLIT = "heldout"


def concat(dirpath: Path, what: str) -> pl.DataFrame:
    files = sorted(dirpath.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no {what} parquet files under {dirpath}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    # Curves carry no stratum columns by design (they are emitted only for the ungrouped
    # cut), so order by whichever lead columns this table actually has.
    lead = [c for c in LEAD if c in df.columns]
    return df.select(lead + [c for c in df.columns if c not in lead])


def main():
    metrics_dir, curves_dir = Path(sys.argv[1]), Path(sys.argv[2])
    parquet_out, csv_out, curves_out = (Path(a) for a in sys.argv[3:6])

    metrics = concat(metrics_dir, "metrics").sort(
        ["fmax", "auprc"], descending=True, nulls_last=True
    )
    metrics.write_parquet(parquet_out, compression="zstd")
    metrics.write_csv(csv_out)

    curves = concat(curves_dir, "curve")
    curves.write_parquet(curves_out, compression="zstd")

    print(f"{metrics.height} metric rows, {curves.height} curve points")
    print()

    board = metrics.filter(
        (pl.col("split") == LEADERBOARD_SPLIT) & (pl.col("stratum_axis") == "all")
    )
    # Never pool truth sets into one leaderboard: Pfam is circular with the profile
    # baselines and Swiss-Prot is not, so a mean across them has no interpretation.
    if "truth_set" in board.columns and board.height:
        for ts in board["truth_set"].unique().sort().to_list():
            print(f"\n--- truth set: {ts} ---")
            _print_board(board.filter(pl.col("truth_set") == ts))
        return
    if board.height == 0:
        # No holdout column at all (e.g. an older truth table) -- fall back rather than
        # print an empty leaderboard, and say which cut is being shown.
        board = metrics.filter(pl.col("stratum_axis") == "all")
        print("NOTE: no heldout rows found; leaderboard below is over all instances")
    else:
        print(f"Leaderboard: split={LEADERBOARD_SPLIT}, ungrouped, averaged over species")

    _print_board(board)


def _print_board(board: pl.DataFrame):
    # Average over species first, then rank. Summing would let the species with the most
    # annotated proteins decide the winner. kmerseek has 113 variants against every other
    # tool's one, so pick each tool's best variant rather than letting the sweep bury the
    # baselines.
    per_variant = (
        board.group_by("tool", "variant")
        .agg([pl.col(c).mean() for c in HEADLINE] + [pl.col("species").n_unique().alias("n_species")])
        .sort("fmax", descending=True, nulls_last=True)
    )
    best = (
        per_variant.group_by("tool")
        .agg([pl.col("variant").first().alias("best_variant"),
              pl.col("n_species").first()]
             + [pl.col(c).first() for c in HEADLINE])
        .sort("fmax", descending=True, nulls_last=True)
    )
    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=40):
        print(best)


if __name__ == "__main__":
    main()
