#!/usr/bin/env python3
"""ROC AUC, average precision and precision at 1 of each kmerseek ranking metric, for
every arm (one alphabet at one k) of the notebook 243 all-against-all search of the 998
Pfam-annotated human proteins. Input for notebook 255.

Each arm's regions (`/Users/olga/data/alphabet-logreg-pfam998/regions/<alphabet>.k<k>.parquet`,
written by 243_pfam998_search.py) are labelled with the four overlap rules of
ranking_metrics_utils.label_regions, reduced to one row per unordered pair of region spans
(the direction with the lower E-value), and scored. That search kept every region
(--threshold 0 --min-shared-kmers 1 --min-region-score 0), so an arm has many more
matches than the 5_649 of labeled_pairs_overlap_rules.parquet.

243_pfam998_search.py kept only region-level columns, so region_poisson_score,
containment, query_enrichment and query_poisson_pvalue cannot be scored per arm.

Usage: 255_ranking_metrics_per_arm.py [--out PATH] [--arms hp_lehninger2.k24,...]
Resumes: arms already in the output file are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_metrics_utils as rm  # noqa: E402

COLS = ["region_evalue", "region_ka_bits", "region_mean_idf", "region_tfidf", "region_enrichment",
        "region_tail_probability", "region_n_shared_kmers", "region_length"]
READ = ["query_name", "target_name", "region_start", "region_end", "target_start", "target_end",
        "region_ka_lambda"] + COLS


def precision_at_1(df: pl.DataFrame, s: np.ndarray, rule: str) -> tuple[float, float, int]:
    """Over queries with at least one correct match: the chance the top-scored match is
    correct (ties averaged), for the metric and for a random order of the query's matches."""
    t = pl.DataFrame({"q": df["query_name"], "s": np.nan_to_num(s, nan=-np.inf), "y": df[rule]})
    g = (t.with_columns(pl.col("s").max().over("q").alias("m"))
         .group_by("q").agg(pl.col("y").any().alias("any"),
                            pl.col("y").filter(pl.col("s") == pl.col("m")).mean().alias("p1"),
                            pl.col("y").mean().alias("rand"))
         .filter(pl.col("any")))
    return float(g["p1"].mean()), float(g["rand"].mean()), g.height


def one_arm(path: Path, truth: pl.DataFrame, bits: dict) -> pl.DataFrame:
    tag = path.stem
    alphabet, k = tag.rsplit(".k", 1)
    t0 = time.time()
    raw = pl.read_parquet(path, columns=READ)
    n_raw = raw.height
    df = rm.independent_matches(rm.label_regions(raw, truth))
    del raw
    L = rm.score(df, "region_length")
    rows = []
    for rule, _ in rm.RULES:
        y = df[rule].to_numpy()
        for c in COLS:
            s = rm.score(df, c)
            ok = np.isfinite(s)
            p1, p1_rand, nq = precision_at_1(df, s, rule) if ok.any() else (np.nan, np.nan, 0)
            rows.append(dict(alphabet=alphabet, ksize=int(k), bits=bits.get(tag, np.nan), metric=rm.NAME[c], column=c,
                             rule=rm.RULE_NAME[rule], n_regions=n_raw, n_matches=df.height, n_scored=int(ok.sum()),
                             n_correct=int(y[ok].sum()), n_correct_all=int(y.sum()),
                             base_rate=float(y[ok].mean()) if ok.any() else np.nan,
                             auc=rm.auc(s[ok], y[ok]), ap=rm.ap(s[ok], y[ok]),
                             length_auc=rm.auc(L[ok], y[ok]), length_ap=rm.ap(L[ok], y[ok]),
                             precision_at_1=p1, precision_at_1_random=p1_rand, n_queries_with_correct=nq))
    print(f"{tag}: {n_raw:_} regions, {df.height:_} matches, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=rm.PER_ARM)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    plan = json.loads(rm.PLAN.read_text())
    bits = {f"{p['alphabet']}.k{p['ksize']}": p["bits"] for p in plan}
    truth = pl.read_parquet(rm.TRUTH)
    paths = sorted((rm.PF998 / "regions").glob("*.parquet"), key=lambda p: p.stat().st_size)
    if args.arms:
        paths = [p for p in paths if p.stem in args.arms.split(",")]
    done = pl.read_parquet(args.out) if args.out.exists() else None
    have = set() if done is None else {f"{a}.k{k}" for a, k in done.select("alphabet", "ksize").unique().iter_rows()}
    for p in paths:
        if p.stem in have:
            continue
        res = one_arm(p, truth, bits)
        done = res if done is None else pl.concat([done, res], how="diagonal_relaxed")
        tmp = args.out.with_suffix(".tmp.parquet")
        done.write_parquet(tmp)
        tmp.replace(args.out)


if __name__ == "__main__":
    main()
