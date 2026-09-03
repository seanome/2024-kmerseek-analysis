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
# family_fmax sits next to fmax rather than replacing it: fmax gates on interval placement,
# family_fmax on (protein, family) set membership, and the difference between the two is
# what separates a tool that never recognised a family from one that recognised it and drew
# the boundary wrong.
# `ndo` was a second name for residue_recall, not a second measurement, so it is gone
# rather than renamed in place: a run whose parquet still carries the old column simply
# does not have it summarised.
HEADLINE = ["fmax", "family_fmax", "auprc", "roc_auc", "smin", "residue_recall",
            "recall_reachable", "precision"]

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


# A target species whose annotation table is this far below the median of its peers is not
# a distant species, it is a species nobody can transfer anything to. Set at a twentieth
# because the observed failure was 30-130x and a legitimately sparse proteome (E. coli
# against a human query set) sits within 5x.
THIN_TARGET_RATIO = 0.05

# Reachability is only a ceiling when `pfam_id` is a family. Below this many distinct
# labels in the answer key it is a category vocabulary, every proteome has nearly all of
# it, and reachable / truth is ~1.0 for every species by construction.
MIN_REACHABILITY_VOCAB = 50


def check_thin_target_annotation(metrics: pl.DataFrame) -> str | None:
    """Name any target species whose annotation table collapsed against its peers.

    This is the check that the Ciona column needed and did not have. Ciona intestinalis
    has 28 UniProtKB/Swiss-Prot entries against 2_309 - 20_417 for every other target
    species, so on the Swiss-Prot truth set its transfer table is ~1/100th the size and
    every one of the ten tools lost 30-130x of its calls at 550 Mya. Nothing failed: the
    tasks ran, exited 0, and published a valid file holding a near-zero result, which the
    report drew as an evolutionary cliff.

    Deliberately cross-species and per truth set, because the number is only readable in
    comparison -- a thin table is not wrong on its own, it is wrong next to eight fat ones.
    The per-arm empty-map assertion in evaluate_domain_calls.py catches the total case on
    first occurrence; this catches the partial case, which is the one that got through.
    """
    col = "n_target_map_proteins"
    if col not in metrics.columns or metrics.height == 0:
        return None
    per = (
        metrics.filter(pl.col(col).is_not_null() & (pl.col("species") != "all"))
        .group_by("truth_set", "species").agg(pl.col(col).max().alias("n_prot"))
    )
    if per.height == 0:
        return None
    flagged = []
    for ts in per["truth_set"].unique().sort().to_list():
        sub = per.filter(pl.col("truth_set") == ts)
        if sub.height < 3:
            continue
        median = float(sub["n_prot"].median())
        if median <= 0:
            continue
        for row in sub.filter(pl.col("n_prot") < median * THIN_TARGET_RATIO).sort(
            "n_prot"
        ).iter_rows(named=True):
            flagged.append((ts, row["species"], row["n_prot"], median))
    if not flagged:
        return None
    lines = [
        "",
        "=" * 72,
        "THIN TARGET ANNOTATION: a species has almost nothing to transfer from.",
        "=" * 72,
    ]
    for ts, sp, n, median in flagged:
        lines.append(
            f"  {ts}/{sp}: {n} annotated target proteins against a median of "
            f"{median:.0f} across the other species in this truth set "
            f"({n / median:.1%})"
        )
    lines += [
        "",
        "  Every transfer-scored arm shares this table, so every arm's recall for this",
        "  species is capped by it. The resulting numbers measure annotation coverage,",
        "  not divergence, and must not be read as an evolutionary result.",
        "  Fix the target annotation, or drop the species from the divergence axis.",
        "=" * 72,
    ]
    return "\n".join(lines)


def check_degenerate_reachability(metrics: pl.DataFrame) -> str | None:
    """Warn when a truth set's `recall_reachable` is plain recall wearing another name."""
    need = {"n_truth_families", "n_truth_instances", "n_reachable_instances"}
    if not need.issubset(set(metrics.columns)) or metrics.height == 0:
        return None
    cut = metrics.filter(
        (pl.col("stratum_axis") == "all") & (pl.col("species") != "all")
    )
    if cut.height == 0:
        return None
    lines = []
    for ts in cut["truth_set"].unique().sort().to_list():
        sub = cut.filter(pl.col("truth_set") == ts)
        vocab = int(sub["n_truth_families"].max() or 0)
        if vocab >= MIN_REACHABILITY_VOCAB:
            continue
        per = sub.group_by("species").agg(
            pl.col("n_truth_instances").max().alias("truth"),
            pl.col("n_reachable_instances").max().alias("reach"),
        )
        frac = per.select(
            (pl.col("reach") / pl.col("truth")).mean()
        ).item()
        lines.append(
            f"  {ts}: {vocab} distinct labels in the answer key, "
            f"reachable / truth averages {frac:.3f} over {per.height} species"
        )
    if not lines:
        return None
    return "\n".join(
        ["", "=" * 72,
         "DEGENERATE REACHABILITY: recall_reachable is plain recall here.",
         "=" * 72]
        + lines
        + ["",
           "  `pfam_id` on these truth sets is a category label, not a family. Every",
           "  proteome carries nearly the whole vocabulary, so the reachability join",
           "  matches everything and the denominator stops being a ceiling. It also",
           "  means the reachability bar cannot detect a target species whose",
           "  annotation has collapsed -- see check_thin_target_annotation.",
           "=" * 72])


def concat(dirpath: Path, what: str) -> pl.DataFrame:
    files = sorted(dirpath.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no {what} parquet files under {dirpath}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    # Curves carry no stratum columns by design (they are emitted only for the ungrouped
    # cut), so order by whichever lead columns this table has.
    lead = [c for c in LEAD if c in df.columns]
    return df.select(lead + [c for c in df.columns if c not in lead])


def report_dead(dead: str | None, allow: bool) -> None:
    """Raise on a dead arm, or say it is being reported anyway.

    A dead arm is a broken run and stops a report by default -- see check_for_dead_arms.
    `allow` exists for the partial report (`make multiqc-partial`), which is a snapshot of a
    run still in progress: an arm can be dead there simply because its search has not
    finished, and refusing to draw the other arms because of it defeats the point of
    looking early. The banner is printed either way, so the report is never quietly built
    over a zero.
    """
    if not dead:
        return
    if not allow:
        raise SystemExit(dead)
    print(dead, file=sys.stderr)
    print("  Reported anyway (--allow-dead-arms). In a snapshot of a run still going, an\n"
          "  arm reads as dead until its search finishes. In a FINISHED run it is a bug.",
          file=sys.stderr)


def report_thin(thin: str | None, allow: bool) -> None:
    """Raise on a collapsed target annotation table, or say it is being reported anyway."""
    if not thin:
        return
    if not allow:
        raise SystemExit(thin)
    print(thin, file=sys.stderr)
    print("  Reported anyway. Every number for the flagged species is a statement about\n"
          "  target annotation coverage, not about divergence.", file=sys.stderr)


def main():
    known = {"--allow-dead-arms", "--allow-thin-targets"}
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    unknown = [f for f in flags if f not in known]
    if unknown:
        raise SystemExit(f"unknown flag(s): {' '.join(unknown)}. "
                         f"Accepted: {' '.join(sorted(known))}.")
    allow_dead = "--allow-dead-arms" in flags
    # A partial report is a snapshot of a run still going, so a species can look thin
    # simply because its scoring has not finished. In a FINISHED run it is a bug, which is
    # why --allow-thin-targets exists separately and has to be asked for on purpose.
    allow_thin = allow_dead or "--allow-thin-targets" in flags

    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(positional) != 5:
        raise SystemExit(
            "usage: aggregate_domain_metrics.py [--allow-dead-arms] "
            "[--allow-thin-targets] METRICS_DIR CURVES_DIR "
            "METRICS_PARQUET METRICS_CSV CURVES_PARQUET"
        )
    metrics_dir, curves_dir = Path(positional[0]), Path(positional[1])
    parquet_out, csv_out, curves_out = (Path(a) for a in positional[2:5])

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
    thin = check_thin_target_annotation(metrics)
    # Always printed, never fatal: a small label vocabulary is a property of the truth set
    # rather than a run failure. It is loud because three report sections read as results
    # while it is true -- the recall ceiling, the divergence axis, and percent identity.
    degenerate = check_degenerate_reachability(metrics)
    if degenerate:
        print(degenerate, file=sys.stderr)

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
        report_dead(dead, allow_dead)
        report_thin(thin, allow_thin)
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

    report_dead(dead, allow_dead)
    report_thin(thin, allow_thin)


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
    # Only the headline columns this table actually has, so a metrics parquet written before
    # a metric existed still prints a board instead of raising on the missing column.
    cols = [c for c in HEADLINE if c in board.columns]
    per_variant = (
        board.group_by("tool", "variant")
        .agg([pl.col(c).mean() for c in cols] + [pl.col("species").n_unique().alias("n_species")])
        .sort("fmax", descending=True, nulls_last=True)
    )
    best = (
        per_variant.group_by("tool")
        .agg([pl.col("variant").first().alias("best_variant"),
              pl.col("n_species").first()]
             + [pl.col(c).first() for c in cols])
        .sort("fmax", descending=True, nulls_last=True)
    )
    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=40):
        print(best)


if __name__ == "__main__":
    main()
