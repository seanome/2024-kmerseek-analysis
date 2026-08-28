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


def check_for_dead_arms(metrics: pl.DataFrame) -> str | None:
    """Name any tool that scored zero calls in every row, when others scored plenty.

    The per-species checks in hhblitsSearch and folddiscoMerge are the first line and the
    better one, because they fire while the task log that explains the emptiness still
    exists. This is the backstop for an arm added later without one.

    The test is deliberately the weakest statement that is still impossible biologically:
    not "few calls", not "zero for this species", but zero for every species and every
    truth set at once, while some other tool in the same run made calls. A distant pair or
    a strict alphabet legitimately returns nothing; a tool that returns nothing for
    anything, anywhere, is not measuring what it claims to measure. hhblits and folddisco
    were in exactly that state for the whole life of this pipeline, and it read as a bar of
    length zero rather than as an error.
    """
    if "n_calls" not in metrics.columns or metrics.height == 0:
        return None
    per_tool = metrics.group_by("tool").agg(
        pl.col("n_calls").fill_null(0).sum().alias("total_calls"),
        pl.len().alias("n_rows"),
        pl.col("species").n_unique().alias("n_species"),
    )
    dead = per_tool.filter(pl.col("total_calls") == 0).sort("tool")
    alive = per_tool.filter(pl.col("total_calls") > 0)
    if dead.height == 0 or alive.height == 0:
        return None
    lines = [
        "",
        "=" * 72,
        "DEAD ARM: every scored call is zero, across every species and truth set.",
        "=" * 72,
    ]
    for row in dead.iter_rows(named=True):
        lines.append(
            f"  {row['tool']}: 0 calls over {row['n_rows']} metric rows, "
            f"{row['n_species']} species"
        )
    lines += [
        "",
        "  These arms produced search output and scored nothing from it, which is a",
        "  broken arm rather than a result. Start at the published regions file:",
        "  a 20-byte .tsv.gz is a gzip header with no payload, meaning the search step",
        "  emitted nothing and exited 0 anyway.",
        "=" * 72,
    ]
    return "\n".join(lines)


def concat(dirpath: Path, what: str) -> pl.DataFrame:
    files = sorted(dirpath.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no {what} parquet files under {dirpath}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    # Curves carry no stratum columns by design (they are emitted only for the ungrouped
    # cut), so order by whichever lead columns this table has.
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

    dead = check_for_dead_arms(metrics)

    board = metrics.filter(
        (pl.col("split") == LEADERBOARD_SPLIT) & (pl.col("stratum_axis") == "all")
    )
    # Never pool truth sets into one leaderboard: Pfam is circular with the profile
    # baselines and Swiss-Prot is not, so a mean across them has no interpretation.
    if "truth_set" in board.columns and board.height:
        for ts in board["truth_set"].unique().sort().to_list():
            for cut, note in _by_dedup_mode(board.filter(pl.col("truth_set") == ts)):
                print(f"\n--- truth set: {ts}{note} ---")
                _print_board(cut)
        # Raised after the parquet and CSV are on disk, so the run fails loudly without
        # taking the tables needed to debug it down with it.
        if dead:
            raise SystemExit(dead)
        return
    if board.height == 0:
        # No holdout column at all (e.g. an older truth table) -- fall back rather than
        # print an empty leaderboard, and say which cut is being shown.
        board = metrics.filter(pl.col("stratum_axis") == "all")
        print("NOTE: no heldout rows found; leaderboard below is over all instances")
    else:
        print(f"Leaderboard: split={LEADERBOARD_SPLIT}, ungrouped, averaged over species")

    for cut, note in _by_dedup_mode(board):
        if note:
            print(f"\n---{note} ---")
        _print_board(cut)

    if dead:
        raise SystemExit(dead)


def _by_dedup_mode(board: pl.DataFrame):
    """Split a board into one frame per dedup-transfer setting, never pooling them.

    Every arm is scored twice: once charging each redundant copy of a call as a false
    positive, once collapsing calls of one family over one query region. Those are two
    measurements of the same arm, so the mean over both is neither, and _print_board's
    group_by(tool, variant) would silently produce exactly that mean.
    """
    if "dedup_transfers" not in board.columns or board.height == 0:
        return [(board, "")]
    modes = sorted(board["dedup_transfers"].unique().to_list())
    if len(modes) < 2:
        return [(board, "")]
    return [(board.filter(pl.col("dedup_transfers") == m),
             ", one call per region" if m else ", calls as reported")
            for m in modes]


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
