#!/usr/bin/env python3
"""Recall at 5% decoy error for every tool, index, decoy window and kmerseek setting, and
the paired tests of PREDICTIONS.md.

Inputs: every <label>.features.parquet and <label>.threshold.json from score_calls.py
(label = tool.index.wNN.setting), the truth table, and feature_identity.tsv.

Outputs:
  thresholds.tsv       one row per label: threshold, target and decoy calls at it
  recall_by_kind.tsv   recall per label x feature kind (and per evidence class, and per
                       chromosome 6 gene group)
  mcnemar.tsv          per landing rule x tool x setting x decoy window x feature kind x
                       index pair: features found with index a only, b only, both,
                       neither; recall change (b - a); exact two-sided McNemar p. Index
                       pairs: whole vs regions (the test), whole vs wholeannot (whole-
                       protein index, annotated targets only: the effect of searching only
                       annotated targets), wholeannot vs regions (the effect of the smaller
                       index alone)
  composition.tsv      kmerseek against the composition-only classifier on the regions
                       index, with the classifier given as many target calls as kmerseek
                       makes at its 5% threshold (prediction 4; Olga, 2026-10-13)
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

PAIRS = [("whole", "regions"), ("whole", "wholeannot"), ("wholeannot", "regions")]
MC_SCHEMA = {
    "landing_rule": pl.Utf8,
    "tool": pl.Utf8,
    "setting": pl.Utf8,
    "window": pl.Utf8,
    "feature_kind": pl.Utf8,
    "index_a": pl.Utf8,
    "index_b": pl.Utf8,
    "features": pl.Int64,
    "a_only": pl.Int64,
    "b_only": pl.Int64,
    "both": pl.Int64,
    "neither": pl.Int64,
    "recall_a": pl.Float64,
    "recall_b": pl.Float64,
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
    comp_hits = {
        p.name.replace(".call_hits.parquet", ""): pl.read_parquet(p)
        for p in args.scored
        if p.name.endswith(".call_hits.parquet")
    }
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
    thr_n = dict(thr.select("label", "n_target").iter_rows())
    truth = pl.read_parquet(args.truth)
    fc = pl.concat([pl.read_parquet(p) for p in feats], how="diagonal_relaxed")
    # The composition classifier is scored on all its calls (score_calls.py --all-calls),
    # so it is compared only in composition.tsv, at kmerseek's call count.
    fc = fc.filter(~pl.col("label").is_in(list(nofit))).filter(
        ~pl.col("label").str.starts_with("composition.")
    )
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
            for ia, ib in PAIRS:
                # A tool not run against both indexes of the pair has no pair.
                if any(
                    c not in g.columns or g[c].null_count() == g.height
                    for c in (ia, ib)
                ):
                    continue
                x, y = g[ia].fill_null(False), g[ib].fill_null(False)
                b_ = int((x & ~y).sum())
                c_ = int((~x & y).sum())
                n = g.height
                mc.append(
                    {
                        "landing_rule": rule,
                        "tool": tool,
                        "setting": setting,
                        "window": window,
                        "feature_kind": kind,
                        "index_a": ia,
                        "index_b": ib,
                        "features": n,
                        "a_only": b_,
                        "b_only": c_,
                        "both": int((x & y).sum()),
                        "neither": int((~x & ~y).sum()),
                        "recall_a": int(x.sum()) / n if n else None,
                        "recall_b": int(y.sum()) / n if n else None,
                        "recall_change_points": 100 * (c_ - b_) / n if n else None,
                        "mcnemar_p": mcnemar_p(b_, c_),
                    }
                )
    pl.DataFrame(mc, schema=MC_SCHEMA).sort(
        "landing_rule",
        "feature_kind",
        "tool",
        "setting",
        "window",
        "index_a",
        "index_b",
    ).write_csv("mcnemar.tsv", separator="\t")

    # Prediction 4: the composition classifier gets as many target calls as kmerseek makes
    # at its 5% threshold on the regions index (same decoy window), best-scoring first.
    out = []
    for kind in ["disordered", "all"]:
        kfeat = fc.filter(
            (pl.col("tool") == "kmerseek")
            & (pl.col("index") == "regions")
            & kind_filter(kind)
        )
        for (rule, setting, window), g in kfeat.group_by(
            "landing_rule", "setting", "window"
        ):
            label = f"kmerseek.regions.{window}.{setting}"
            n_calls = thr_n.get(label, 0)
            hits = comp_hits.get(f"composition.regions.{window}.composition")
            found_c = (
                set()
                if hits is None
                else set(
                    hits.filter(
                        (pl.col("landing_rule") == rule) & (pl.col("rank") < n_calls)
                    )["truth_id"].to_list()
                )
            )
            j = g.select("truth_id", "found").with_columns(
                pl.col("truth_id").is_in(list(found_c)).alias("comp")
            )
            b_ = int((j["found"] & ~j["comp"]).sum())
            c_ = int((~j["found"] & j["comp"]).sum())
            out.append(
                {
                    "landing_rule": rule,
                    "feature_kind": kind,
                    "setting": setting,
                    "window": window,
                    "calls_each": n_calls,
                    "features": j.height,
                    "kmerseek_found": int(j["found"].sum()),
                    "composition_found": int(j["comp"].sum()),
                    "kmerseek_only": b_,
                    "composition_only": c_,
                    "mcnemar_p": mcnemar_p(b_, c_),
                }
            )
    pl.DataFrame(
        out,
        schema={
            "landing_rule": pl.Utf8,
            "feature_kind": pl.Utf8,
            "setting": pl.Utf8,
            "window": pl.Utf8,
            "calls_each": pl.Int64,
            "features": pl.Int64,
            "kmerseek_found": pl.Int64,
            "composition_found": pl.Int64,
            "kmerseek_only": pl.Int64,
            "composition_only": pl.Int64,
            "mcnemar_p": pl.Float64,
        },
    ).sort("landing_rule", "feature_kind", "setting", "window").write_csv(
        "composition.tsv", separator="\t"
    )

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
