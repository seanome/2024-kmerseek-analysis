#!/usr/bin/env python3
"""Score one set of calls: the 5% decoy threshold, and which query features it finds.

A set is one tool x index x decoy window x kmerseek setting. Calls arrive in the shared
format every search step writes (calls.tsv.gz):

  query, qstart, qend, target, tstart, tend, evalue, qseq, tseq

with coordinates 1-based and inclusive on both proteins.

Rules (PREDICTIONS.md):
- A call to a target whose name starts with DECOY_ is a decoy call.
- Threshold: calls are ranked by E-value; the threshold is the largest E-value at which
  decoy calls / target calls <= --max-decoy-rate among calls at or below it.
- The hit entry's feature types:
    regions index         the type in the entry's name (accession|type|description|start-end)
    whole-protein index   the types of the target protein's features whose interval,
                          widened by --flank residues each side, overlaps the target side
                          of the call. The widening mirrors the regions index, whose
                          entries carry the same flank.
- A call is correct for a query feature when the feature's type is one of the hit's types
  and landing = (call residues inside the feature) / (call length) >= --min-landing.
  Coverage = (feature residues inside the call) / (feature length) is reported.
- Secondary: a name match, when the hit feature's description (cleaned the way
  extract_regions.py cleans it) equals the query feature's.

Outputs:
  <prefix>.features.parquet  one row per query feature: found (at the threshold),
                             found_name, best_landing, best_coverage, best_evalue,
                             best_target, best_qstart/qend/tstart/tend, best_qseq, best_tseq
  <prefix>.threshold.json    the threshold and call counts
"""

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl

WS_RE = re.compile(r"\s+")


def clean(desc: str | None) -> str:
    d = WS_RE.sub("_", (desc or "").strip()).replace("|", "/").replace(",", ";")
    return d or "none"


def decoy_threshold(calls: pl.DataFrame, max_rate: float) -> dict:
    """Largest E-value E with (decoys <= E) / (targets <= E) <= max_rate."""
    if calls.height == 0:
        return {"threshold": None, "n_target": 0, "n_decoy": 0}
    g = (
        calls.group_by("evalue")
        .agg(
            pl.col("is_decoy").sum().alias("d"),
            (~pl.col("is_decoy")).sum().alias("t"),
        )
        .sort("evalue")
        .with_columns(
            pl.col("d").cum_sum().alias("cd"), pl.col("t").cum_sum().alias("ct")
        )
    )
    ok = g.filter((pl.col("ct") > 0) & (pl.col("cd") <= max_rate * pl.col("ct")))
    if ok.height == 0:
        return {"threshold": None, "n_target": 0, "n_decoy": 0}
    row = ok.sort("evalue").tail(1).row(0, named=True)
    return {
        "threshold": float(row["evalue"]),
        "n_target": int(row["ct"]),
        "n_decoy": int(row["cd"]),
    }


def hit_types_regions(calls: pl.DataFrame) -> pl.DataFrame:
    parts = pl.col("target").str.split("|")
    return calls.with_columns(
        parts.list.get(1).alias("hit_type"), parts.list.get(2).alias("hit_desc")
    )


def hit_types_whole(
    calls: pl.DataFrame, tfeat: pl.DataFrame, flank: int
) -> pl.DataFrame:
    """One row per (call, target feature) whose widened interval overlaps the call."""
    j = calls.join(
        tfeat.select(
            pl.col("accession").alias("target"),
            pl.col("feature_type").alias("hit_type"),
            pl.col("description").alias("hit_desc_raw"),
            (pl.col("start") - flank).alias("_fs"),
            (pl.col("end") + flank).alias("_fe"),
        ),
        on="target",
        how="inner",
    ).filter((pl.col("_fs") <= pl.col("tend")) & (pl.col("_fe") >= pl.col("tstart")))
    return j.with_columns(
        pl.col("hit_desc_raw")
        .map_elements(clean, return_dtype=pl.Utf8)
        .alias("hit_desc")
    ).drop("_fs", "_fe", "hit_desc_raw")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--index-kind", choices=["regions", "whole"], required=True)
    ap.add_argument(
        "--target-features",
        type=Path,
        help="sprot_features.parquet; needed for the whole-protein index",
    )
    ap.add_argument("--flank", type=int, required=True)
    ap.add_argument("--max-decoy-rate", type=float, default=0.05)
    ap.add_argument("--min-landing", type=float, default=0.5)
    ap.add_argument(
        "--searched",
        type=Path,
        required=True,
        help="the FASTA of the queries searched; truth is limited to these proteins",
    )
    ap.add_argument(
        "--nofit",
        action="store_true",
        help="the index holds no Karlin-Altschul fit for this setting: no E-values",
    )
    ap.add_argument("--label", required=True, help="tool.index.window.setting")
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    calls = pl.read_csv(
        args.calls,
        separator="\t",
        schema_overrides={"query": pl.Utf8, "target": pl.Utf8, "evalue": pl.Float64},
        quote_char=None,
    ).with_columns(pl.col("target").str.starts_with("DECOY_").alias("is_decoy"))
    bad = calls.filter(
        (pl.col("qstart") < 1)
        | (pl.col("qend") < pl.col("qstart"))
        | pl.col("evalue").is_null()
    )
    if bad.height:
        sys.exit(
            f"{bad.height} calls with bad coordinates or no E-value, e.g. {bad.row(0)}"
        )

    thr = (
        {"threshold": None, "n_target": 0, "n_decoy": 0}
        if args.nofit
        else decoy_threshold(calls, args.max_decoy_rate)
    )
    thr["nofit"] = args.nofit
    thr.update(
        {
            "label": args.label,
            "calls_total": calls.height,
            "decoy_calls_total": int(calls["is_decoy"].sum()),
            "max_decoy_rate": args.max_decoy_rate,
        }
    )

    with open(args.searched) as fh:
        searched = [ln[1:].split()[0] for ln in fh if ln.startswith(">")]
    truth = pl.read_parquet(args.truth).filter(pl.col("accession").is_in(searched))
    truth = truth.with_columns(
        pl.col("description")
        .map_elements(clean, return_dtype=pl.Utf8)
        .alias("desc_clean")
    )
    real = calls.filter(~pl.col("is_decoy")).with_row_index("call_id")
    if args.index_kind == "regions":
        typed = hit_types_regions(real)
    else:
        tfeat = pl.read_parquet(args.target_features)
        typed = hit_types_whole(real, tfeat, args.flank)

    # Pair each typed call with every query feature of the same type on the same protein.
    pairs = typed.join(
        truth.select(
            "truth_id",
            pl.col("accession").alias("query"),
            pl.col("feature_type").alias("hit_type"),
            pl.col("start").alias("fstart"),
            pl.col("end").alias("fend"),
            "desc_clean",
        ),
        on=["query", "hit_type"],
        how="inner",
    )
    ov = (
        pl.min_horizontal("qend", "fend") - pl.max_horizontal("qstart", "fstart") + 1
    ).clip(lower_bound=0)
    pairs = pairs.with_columns(
        (ov / (pl.col("qend") - pl.col("qstart") + 1)).alias("landing"),
        (ov / (pl.col("fend") - pl.col("fstart") + 1)).alias("coverage"),
        (pl.col("hit_desc") == pl.col("desc_clean")).alias("name_match"),
    ).filter(pl.col("landing") >= args.min_landing)
    pairs = pairs.with_columns(
        (
            pl.lit(thr["threshold"] is not None)
            & (
                pl.col("evalue")
                <= (thr["threshold"] if thr["threshold"] is not None else -1.0)
            )
        ).alias("passes")
    )

    best = (
        pairs.filter("passes")
        .sort(
            ["truth_id", "evalue", "landing", "call_id"],
            descending=[False, False, True, False],
        )
        .group_by("truth_id", maintain_order=True)
        .agg(
            pl.col("landing").first().alias("best_landing"),
            pl.col("coverage").first().alias("best_coverage"),
            pl.col("evalue").first().alias("best_evalue"),
            pl.col("target").first().alias("best_target"),
            pl.col("qstart").first().alias("best_qstart"),
            pl.col("qend").first().alias("best_qend"),
            pl.col("tstart").first().alias("best_tstart"),
            pl.col("tend").first().alias("best_tend"),
            pl.col("qseq").first().alias("best_qseq"),
            pl.col("tseq").first().alias("best_tseq"),
            pl.col("name_match").any().alias("found_name"),
        )
    )
    out = (
        truth.select("truth_id")
        .join(best, on="truth_id", how="left")
        .with_columns(
            pl.col("best_landing").is_not_null().alias("found"),
            pl.col("found_name").fill_null(False),
            pl.lit(args.label).alias("label"),
        )
        .sort("truth_id")
    )
    out.write_parquet(f"{args.prefix}.features.parquet")
    thr["features_found"] = int(out["found"].sum())
    thr["features_total"] = out.height
    Path(f"{args.prefix}.threshold.json").write_text(json.dumps(thr, indent=2))
    print(json.dumps(thr), file=sys.stderr)


if __name__ == "__main__":
    main()
