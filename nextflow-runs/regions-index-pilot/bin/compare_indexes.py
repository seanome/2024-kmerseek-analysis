#!/usr/bin/env python3
"""Recall at 5% decoy error for every tool, index, decoy window and kmerseek setting, and
the paired tests of PREDICTIONS.md.

Inputs: every <label>.features.parquet and <label>.threshold.json from score_calls.py
(label = tool.index.wNN.setting), the truth table, and feature_identity.tsv.

Outputs:
  thresholds.tsv       one row per label: threshold, target and decoy calls at it
  recall_by_kind.tsv   recall per label x feature kind (and per evidence class, and per
                       chromosome 6 gene group)
  mcnemar.tsv          per tool x setting x decoy window x feature kind: features found
                       with the whole-protein index only, the regions index only, both,
                       neither; recall change; exact two-sided McNemar p
  composition.tsv      kmerseek against the composition-only classifier, regions index,
                       disordered features (prediction 4)
  identity_bins.tsv    recall per label split by each query feature's identity to its
                       closest same-type regions-index entry (feature_identity.py)
  feature_calls.parquet every label x feature row, for the notebook

Feature kinds are those of build_query_truth.py; "short" (under 60 aa) overlaps the
others. The exact McNemar p is the two-sided binomial test on the discordant pairs.
"""

import argparse
import json
import math
from pathlib import Path

import polars as pl

MC_SCHEMA = {
    "landing_rule": pl.Utf8,
    "tool": pl.Utf8,
    "setting": pl.Utf8,
    "window": pl.Utf8,
    "feature_kind": pl.Utf8,
    "features": pl.Int64,
    "whole_only": pl.Int64,
    "regions_only": pl.Int64,
    "both": pl.Int64,
    "neither": pl.Int64,
    "recall_whole": pl.Float64,
    "recall_regions": pl.Float64,
    "recall_change_points": pl.Float64,
    "mcnemar_p": pl.Float64,
}
KINDS = ["folded_domain", "short", "disordered", "motif", "composition", "other", "all"]


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def kind_filter(kind: str) -> pl.Expr:
    if kind == "all":
        return pl.lit(True)
    if kind == "short":
        return pl.col("is_short")
    return pl.col("feature_kind") == kind


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scored", type=Path, nargs="+", required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--feature-identity", type=Path, required=True)
    args = ap.parse_args()

    feats = [p for p in args.scored if p.name.endswith(".features.parquet")]
    thrs = [p for p in args.scored if p.name.endswith(".threshold.json")]
    thr = pl.DataFrame([json.loads(p.read_text()) for p in thrs])
    thr = thr.with_columns(
        pl.col("label")
        .str.split(".")
        .list.to_struct(fields=["tool", "index", "window", "setting"])
        .alias("_p")
    ).unnest("_p")
    thr.sort("tool", "setting", "window", "index").write_csv(
        "thresholds.tsv", separator="\t"
    )

    # A set whose index holds no Karlin-Altschul fit has no E-values and so no threshold.
    # It is listed in thresholds.tsv and left out of every recall and test, never counted
    # as zero recall.
    nofit = set(thr.filter(pl.col("nofit"))["label"].to_list())
    truth = pl.read_parquet(args.truth)
    fc = pl.concat([pl.read_parquet(p) for p in feats], how="diagonal_relaxed")
    fc = fc.filter(~pl.col("label").is_in(list(nofit)))
    fc = fc.with_columns(
        pl.col("label")
        .str.split(".")
        .list.to_struct(fields=["tool", "index", "window", "setting"])
        .alias("_p")
    ).unnest("_p")
    fc = fc.join(
        truth.select(
            "truth_id",
            "accession",
            "gene",
            "feature_type",
            "feature_kind",
            "is_short",
            "experimental",
            "gene_group",
            "start",
            "end",
            "length",
            "description",
        ),
        on="truth_id",
        how="left",
    )

    # Identity split: each feature's identity to its closest same-type regions-index entry,
    # from one MMseqs2 search outside the compared tools (feature_identity.py).
    id_df = pl.read_csv(
        args.feature_identity,
        separator="\t",
        schema_overrides={"truth_id": pl.UInt32, "identity": pl.Float64},
    )
    fc = fc.join(id_df, on="truth_id", how="left")
    # Both landing rules (score_calls.py): the whole call, as written in PREDICTIONS.md, and
    # the piece of the call aligned to the target feature, added after the 20-query test.
    fc = pl.concat(
        [
            fc.with_columns(pl.lit("whole call").alias("landing_rule")),
            fc.with_columns(
                pl.col("found_aligned").alias("found"),
                pl.col("found_name_aligned").alias("found_name"),
                pl.lit("aligned piece").alias("landing_rule"),
            ),
        ]
    )
    fc.write_parquet("feature_calls.parquet")

    rows = []
    for kind in KINDS:
        for ev, evf in [
            ("all", pl.lit(True)),
            ("experimental", pl.col("experimental")),
        ]:
            for grp in ["all", "MHC", "histone", "olfactory_receptor"]:
                gf = pl.lit(True) if grp == "all" else pl.col("gene_group") == grp
                sub = fc.filter(kind_filter(kind) & evf & gf)
                agg = sub.group_by(
                    "landing_rule", "tool", "index", "window", "setting"
                ).agg(
                    pl.len().alias("features"),
                    pl.col("found").sum().alias("found"),
                    pl.col("found_name").sum().alias("found_name"),
                )
                rows.append(
                    agg.with_columns(
                        pl.lit(kind).alias("feature_kind"),
                        pl.lit(ev).alias("evidence"),
                        pl.lit(grp).alias("gene_group"),
                    )
                )
    rec = (
        pl.concat(rows)
        .with_columns((pl.col("found") / pl.col("features")).alias("recall"))
        .sort(
            "feature_kind",
            "evidence",
            "gene_group",
            "tool",
            "setting",
            "window",
            "index",
        )
    )
    rec.write_csv("recall_by_kind.tsv", separator="\t")

    wide = (
        fc.select(
            "landing_rule", "tool", "setting", "window", "index", "truth_id", "found"
        )
        .pivot(
            on="index",
            index=["landing_rule", "tool", "setting", "window", "truth_id"],
            values="found",
        )
        .join(
            truth.select("truth_id", "feature_kind", "is_short", "experimental"),
            on="truth_id",
        )
    )
    mc = []
    for kind in KINDS:
        sub = wide.filter(kind_filter(kind))
        for (rule, tool, setting, window), g in sub.group_by(
            "landing_rule", "tool", "setting", "window"
        ):
            # A tool run against one index only (the composition classifier) has no pair.
            if any(
                c not in g.columns or g[c].null_count() == g.height
                for c in ("whole", "regions")
            ):
                continue
            w, r = g["whole"].fill_null(False), g["regions"].fill_null(False)
            b = int((w & ~r).sum())
            c = int((~w & r).sum())
            n = g.height
            mc.append(
                {
                    "landing_rule": rule,
                    "tool": tool,
                    "setting": setting,
                    "window": window,
                    "feature_kind": kind,
                    "features": n,
                    "whole_only": b,
                    "regions_only": c,
                    "both": int((w & r).sum()),
                    "neither": int((~w & ~r).sum()),
                    "recall_whole": int(w.sum()) / n if n else None,
                    "recall_regions": int(r.sum()) / n if n else None,
                    "recall_change_points": 100 * (c - b) / n if n else None,
                    "mcnemar_p": mcnemar_p(b, c),
                }
            )
    pl.DataFrame(mc, schema=MC_SCHEMA).sort(
        "feature_kind", "tool", "setting", "window"
    ).write_csv("mcnemar.tsv", separator="\t")

    comp = fc.filter(
        (pl.col("index") == "regions") & (pl.col("feature_kind") == "disordered")
    )
    cl = comp.filter(pl.col("tool") == "composition")
    out = []
    for (rule, tool, setting, window), g in comp.filter(
        pl.col("tool") == "kmerseek"
    ).group_by("landing_rule", "tool", "setting", "window"):
        cw = cl.filter(
            (pl.col("window") == window) & (pl.col("landing_rule") == rule)
        ).select("truth_id", pl.col("found").alias("comp"))
        j = (
            g.select("truth_id", "found")
            .join(cw, on="truth_id", how="left")
            .with_columns(pl.col("comp").fill_null(False))
        )
        b = int((j["found"] & ~j["comp"]).sum())
        c = int((~j["found"] & j["comp"]).sum())
        out.append(
            {
                "landing_rule": rule,
                "setting": setting,
                "window": window,
                "features": j.height,
                "kmerseek_found": int(j["found"].sum()),
                "composition_found": int(j["comp"].sum()),
                "kmerseek_only": b,
                "composition_only": c,
                "mcnemar_p": mcnemar_p(b, c),
            }
        )
    pl.DataFrame(
        out,
        schema={
            "landing_rule": pl.Utf8,
            "setting": pl.Utf8,
            "window": pl.Utf8,
            "features": pl.Int64,
            "kmerseek_found": pl.Int64,
            "composition_found": pl.Int64,
            "kmerseek_only": pl.Int64,
            "composition_only": pl.Int64,
            "mcnemar_p": pl.Float64,
        },
    ).write_csv("composition.tsv", separator="\t")

    (
        fc.group_by(
            "landing_rule", "tool", "index", "window", "setting", "identity_bin"
        )
        .agg(pl.len().alias("features"), pl.col("found").sum().alias("found"))
        .with_columns((pl.col("found") / pl.col("features")).alias("recall"))
        .sort("identity_bin", "tool", "setting", "window", "index")
        .write_csv("identity_bins.tsv", separator="\t")
    )


if __name__ == "__main__":
    main()
