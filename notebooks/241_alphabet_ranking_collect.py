#!/usr/bin/env python3
"""Collect the three-case alphabet sweep into two tidy tables.

Reads what 241_alphabet_ranking_driver.py wrote under
/Users/olga/data/botryllus/alphabet-ranking-three-cases/ and writes, in the same
directory:

  arms.csv     one row per (alphabet, ksize): bits per seed, whether the Karlin-Altschul
               fit succeeded, lambda and K, region counts per query, and the pairwise
               shared k-mer counts for Ced9/BCL2 and P66/CD47.
  ranks.csv    one row per (alphabet, ksize, query, metric): where the known partner
               ranks among all human targets under that metric, the partner's value,
               the best value over all targets, and how many targets tie with the
               partner. For BHF, which has no known partner, the row carries the top
               target instead.
  regions.parquet  every region of every search, with alphabet, ksize, bits and gene
               symbol added, for any figure that wants the raw distribution.

A target is one human protein. Its score under a metric is its best region (lowest
E-value, highest IDF, and so on). Rank is 1 + the number of targets strictly better;
n_tied says how many share the partner's exact value, so rank 1 with n_tied 40 means
"tied for first with 39 others", not "found".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

OUT = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
PARTNER = {"Ced9": "BCL2", "P66": "CD47"}
QUERIES = ["Ced9", "P66", "BHF"]

# (column, lower_is_better, label)
METRICS = [
    ("region_evalue", True, "E-value"),
    ("region_ka_bits", False, "bit score"),
    ("region_mean_idf", False, "mean IDF"),
    ("region_tfidf", False, "tf-idf"),
    ("region_enrichment", False, "enrichment"),
    ("region_poisson_score", False, "Poisson score"),
    ("region_tail_probability", True, "Poisson p-value"),
    ("region_n_shared_kmers", False, "shared k-mers"),
    ("containment", False, "containment"),
    ("query_tfidf", False, "protein tf-idf"),
    ("query_enrichment", False, "protein enrichment"),
    ("query_poisson_pvalue", True, "protein Poisson p-value"),
]

# The gold-standard homologous window, in 0-based coordinates as kmerseek pair
# reports them: Ced9 162-181 against BCL2 138-157 (the BH3-binding groove; the
# 19-residue core notebook 220 found and this sweep reproduces at hp_lehninger2 k=17).
GOLD_WINDOW = {"Ced9": (150, 195), "BCL2": (125, 170)}


def gene_symbol(target_name: str) -> str:
    parts = target_name.split("|")
    return parts[6] if len(parts) > 6 else target_name


def read_plan() -> pl.DataFrame:
    return pl.DataFrame(json.loads((OUT / "plan.json").read_text()))


def read_survival(tag: str) -> dict:
    """fitted, lambda, K from the ka_survival CSV the index build wrote."""
    # The driver retries a refused fit with more queries and shuffles and writes the
    # retry's curve beside the first one; the retry is the fit the search then used.
    p = OUT / "ka_survival" / f"{tag}.retry.csv"
    if not p.exists():
        p = OUT / "ka_survival" / f"{tag}.csv"
    if not p.exists():
        return {"fitted": None, "ka_lambda": None, "ka_k": None,
                "ka_lambda_analytic": None, "match_probability": None,
                "n_reference_regions": None}
    # Columns as kmerseek 0.4 writes them: `slope` is the fitted lambda, `k` is K,
    # `lambda_analytic` and `match_probability` are the closed form, `fitted` says
    # whether the fit was accepted. Every row repeats the fit, so the first row is enough.
    df = pl.read_csv(p, infer_schema_length=10_000)
    cols = set(df.columns)
    out = {"fitted": None, "ka_lambda": None, "ka_k": None,
           "ka_lambda_analytic": None, "match_probability": None,
           "n_reference_regions": None}
    if df.height == 0:
        return out

    def first(col):
        v = df[col][0] if col in cols else None
        return None if v is None else v

    v = first("fitted")
    out["fitted"] = (str(v).lower() == "true") if v is not None else None
    for src, dst in (("slope", "ka_lambda"), ("k", "ka_k"),
                     ("lambda_analytic", "ka_lambda_analytic"),
                     ("match_probability", "match_probability"),
                     ("n_reference_regions", "n_reference_regions")):
        v = first(src)
        out[dst] = float(v) if v is not None else None
    return out


def read_pair(tag: str, qname: str) -> dict:
    p = OUT / "pair" / f"{tag}.{qname}.json"
    key = f"pair_{qname}"
    if not p.exists():
        return {f"{key}_shared_kmers": None, f"{key}_regions": None,
                f"{key}_longest": None, f"{key}_in_window": None}
    d = json.loads(p.read_text())
    regs = d.get("regions", [])
    longest = max((r["length"] for r in regs), default=0)
    in_window = None
    if qname in GOLD_WINDOW:
        qlo, qhi = GOLD_WINDOW["Ced9"]
        tlo, thi = GOLD_WINDOW["BCL2"]
        in_window = any(
            r["query_start"] < qhi and r["query_end"] > qlo
            and r["target_start"] < thi and r["target_end"] > tlo
            for r in regs
        )
    return {f"{key}_shared_kmers": len(d.get("shared_kmers", [])),
            f"{key}_regions": len(regs),
            f"{key}_longest": longest,
            f"{key}_in_window": in_window}


def read_search(tag: str) -> pl.DataFrame | None:
    p = OUT / "search" / f"{tag}.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pl.read_csv(p, infer_schema_length=0)
    if df.height == 0:
        return None
    num = [c for c, _, _ in METRICS] + [
        "region_start", "region_end", "region_length", "region_n_mismatches",
        "target_start", "target_end", "n_intersecting_hashes", "region_ka_lambda",
    ]
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False) for c in num if c in df.columns
    ])
    return df.with_columns(
        pl.col("query_name").str.strip_chars().alias("query_name"),
        pl.col("target_name").map_elements(gene_symbol, return_dtype=pl.Utf8).alias("gene"),
    )


def rank_rows(df: pl.DataFrame, alphabet: str, k: int, bits: float) -> list[dict]:
    rows = []
    for q in QUERIES:
        sub = df.filter(pl.col("query_name") == q)
        n_regions = sub.height
        n_targets = sub["target_name"].n_unique() if n_regions else 0
        partner = PARTNER.get(q)
        for col, lower, label in METRICS:
            if col not in sub.columns or n_regions == 0:
                rows.append(dict(alphabet=alphabet, ksize=k, bits=bits, query=q, metric=label,
                                 n_regions=n_regions, n_targets=n_targets, partner=partner,
                                 partner_found=False, rank=None, n_tied=None,
                                 partner_value=None, best_value=None, top_gene=None))
                continue
            # A target's score is its best region under this metric.
            agg = pl.col(col).min() if lower else pl.col(col).max()
            per = (sub.group_by("target_name", "gene").agg(agg.alias("v"))
                      .filter(pl.col("v").is_not_null() & pl.col("v").is_finite()))
            if per.height == 0:
                rows.append(dict(alphabet=alphabet, ksize=k, bits=bits, query=q, metric=label,
                                 n_regions=n_regions, n_targets=n_targets, partner=partner,
                                 partner_found=False, rank=None, n_tied=None,
                                 partner_value=None, best_value=None, top_gene=None))
                continue
            best = float(per["v"].min() if lower else per["v"].max())
            top = per.filter(pl.col("v") == best)["gene"][0]
            rec = dict(alphabet=alphabet, ksize=k, bits=bits, query=q, metric=label,
                       n_regions=n_regions, n_targets=n_targets, partner=partner,
                       best_value=best, top_gene=top)
            if partner is None:
                rec.update(partner_found=None, rank=None, n_tied=None, partner_value=None)
            else:
                hit = per.filter(pl.col("gene") == partner)
                if hit.height == 0:
                    rec.update(partner_found=False, rank=None, n_tied=None, partner_value=None)
                else:
                    pv = float(hit["v"][0])
                    better = (per["v"] < pv).sum() if lower else (per["v"] > pv).sum()
                    tied = (per["v"] == pv).sum() - 1
                    rec.update(partner_found=True, rank=int(better) + 1, n_tied=int(tied),
                               partner_value=pv)
            rows.append(rec)
    return rows


def main() -> None:
    plan = read_plan()
    arms, ranks, regions = [], [], []
    for a, k, bits, c, x in plan.select("alphabet", "ksize", "bits", "penalty", "xdrop").iter_rows():
        tag = f"{a}.k{k}"
        arm = dict(alphabet=a, ksize=k, bits=bits, penalty=c, xdrop=x)
        arm.update(read_survival(tag))
        for q in PARTNER:
            arm.update(read_pair(tag, q))
        df = read_search(tag)
        if df is None:
            for q in QUERIES:
                arm[f"n_regions_{q}"] = 0
                arm[f"n_targets_{q}"] = 0
            arm["searched"] = (OUT / "search" / f"{tag}.csv").exists()
            arms.append(arm)
            continue
        arm["searched"] = True
        for q in QUERIES:
            sub = df.filter(pl.col("query_name") == q)
            arm[f"n_regions_{q}"] = sub.height
            arm[f"n_targets_{q}"] = sub["target_name"].n_unique()
        arms.append(arm)
        ranks.extend(rank_rows(df, a, k, bits))
        regions.append(df.with_columns(
            pl.lit(a).alias("alphabet"), pl.lit(k).alias("ksize_arm"), pl.lit(bits).alias("bits")))

    pl.DataFrame(arms, infer_schema_length=None).write_csv(OUT / "arms.csv")
    rank_schema = {"alphabet": pl.Utf8, "ksize": pl.Int64, "bits": pl.Float64, "query": pl.Utf8,
                   "metric": pl.Utf8, "n_regions": pl.Int64, "n_targets": pl.Int64,
                   "partner": pl.Utf8, "partner_found": pl.Boolean, "rank": pl.Int64,
                   "n_tied": pl.Int64, "partner_value": pl.Float64, "best_value": pl.Float64,
                   "top_gene": pl.Utf8}
    pl.DataFrame(ranks, schema=rank_schema).write_csv(OUT / "ranks.csv")
    if regions:
        keep = list(dict.fromkeys(
            ["alphabet", "ksize_arm", "bits", "query_name", "target_name", "gene",
             "region_start", "region_end", "region_length", "region_n_mismatches",
             "target_start", "target_end", "region_ka_lambda"] + [c for c, _, _ in METRICS]))
        pl.concat([r.select([c for c in keep if c in r.columns]) for r in regions],
                  how="diagonal").write_parquet(OUT / "regions.parquet")
    print(f"arms: {len(arms)}  rank rows: {len(ranks)}  "
          f"searched: {sum(a['searched'] for a in arms)}", file=sys.stderr)


if __name__ == "__main__":
    main()
