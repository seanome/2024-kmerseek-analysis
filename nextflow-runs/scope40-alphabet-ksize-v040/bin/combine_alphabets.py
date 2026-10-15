#!/usr/bin/env python3
"""Combine several SCOPe40 searches (one per alphabet) and score the combinations.

Each input is one search's pair table (`<alphabet>.k<k>.<setting>.pairs.parquet`, written by
score_scope40_setting.py --pairs-parquet), ranked by region_evalue (extension settings, smaller
first) or jaccard (exact, larger first). Every method is scored with the same code as a single
search (score_scope40_setting.score_column and band_rows):

  single:<search>     each search alone; reproduces its row of scope40_v040_auc.tsv
  best_of_n           a pair's smallest region_evalue over the extension searches that report it
  rank_fusion         reciprocal rank fusion: sum over searches of 1 / (RRF_K + rank), rank being
                      the pair's position in that search's ranking for its query (1 = best)
  union_upper_limit   a true pair counts if any search ranks it above that search's own first
                      false positive (a different-fold hit; for class pairs a different-class
                      hit). No single ranking can be made from this, so it is reported on the
                      database scale (auc_ref_db) only: the most that combining these
                      rankings could find before the first false positive.

With --truncate, each search keeps, per query, only its hits up to and including its first
different-class hit (all hits when the query has none), plus every same-family hit (see the
cut below for why). Searches at small k report up to 132
million pairs; this keeps the merged table small. Every database-scale number stays exact: the
first different-class hit comes at or after the first different-fold hit, so no level's first
false positive is cut; a cut pair scores worse than a false positive of its own search, so its
smallest E-value cannot place it above the merged list's first false positive; and the union
flags are set before cutting. Two things do depend on the cut hits and are not reported: rank
fusion, and the reported-pairs scale (auc_ref, auc_all and the identity bands' n_reported, which
count every reported true pair). They are left empty, and rows carry truncated = true.

  combine_alphabets.py --domains scope_domains.tsv --ref-rocx foldseek.rocx --utils
      scope_kmerseek_utils.py --pair-identity identity.tsv.gz --out-auc combined_auc.tsv
      --out-bands combined_bands.tsv [--query-set shuffled_k2_seed1] [--truncate] a.pairs.parquet ...
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("score_scope40_setting", HERE / "score_scope40_setting.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

RRF_K = 60  # Cormack, Clarke and Buettcher, SIGIR 2009: the constant of reciprocal rank fusion


def parse_name(path):
    m = re.match(r"(.+)\.k(\d+)\.(.+)\.pairs\.parquet$", Path(path).name)
    if not m:
        sys.exit(f"cannot read alphabet, k and setting from {path}")
    return m.group(1), int(m.group(2)), m.group(3)


def main():
    p = argparse.ArgumentParser()
    for x in ("domains", "ref-rocx", "utils", "pair-identity", "out-auc", "out-bands"):
        p.add_argument(f"--{x}", required=True)
    p.add_argument("--query-set", help="label for shuffled queries (a column)")
    p.add_argument("--truncate", action="store_true",
                   help="keep each query's hits only up to its first different-class hit")
    p.add_argument("pairs", nargs="+")
    a = p.parse_args()

    su = ss.load_utils(a.utils)
    dom = pl.read_csv(a.domains, separator="\t")
    domain = pl.Enum(dom["domain_id"].to_list())
    ref = set(pl.read_csv(a.ref_rocx, separator="\t")["NAME"])
    counts = ss.database_counts(dom)
    ident = ss.read_identity(a.pair_identity, domain)
    base = {"queries": a.query_set} if a.query_set else {}
    if a.truncate:
        base["truncated"] = True
    # Columns that count every reported true pair; empty when hits beyond the cut are gone.
    reported_scale = ("auc_all", "auc_all_optimistic", "n_queries_all", "auc_ref", "auc_ref_optimistic")

    def blank(r, keys):
        return {**r, **{k: None for k in keys if k in r}} if a.truncate else r

    rows, bands, long = [], [], []
    searches = []
    for path in sorted(a.pairs):
        alphabet, k, setting = parse_name(path)
        slug = f"{alphabet}.k{k}.{setting}"
        col = "jaccard" if setting == "exact" else "region_evalue"
        smaller = ss.SCORES[col]
        searches.append(slug)
        pairs = (pl.read_parquet(path, columns=["query_domain", "target_domain", col])
                 .with_columns(pl.col("query_domain").cast(pl.Utf8).cast(domain),
                               pl.col("target_domain").cast(pl.Utf8).cast(domain)))
        df = ss.label_pairs(pairs, dom, domain)
        n_reported = df.height

        # Per pair: rank in its query, and whether it sits above this search's first false
        # positive at fold level (different fold) and class level (different class).
        s = pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None)
        best = s.min() if smaller else s.max()
        # As in eval_tsv_to_rocx: with no false-positive hit at all, every true pair counts; with
        # only unscored false positives, every scored true pair; otherwise a pair must score
        # strictly better than the best false positive (a tie counts against kmerseek).
        fp = (df.filter(~pl.col("same_fold")).group_by("query_domain")
              .agg(best.alias("fp_fold"), pl.len().alias("n_fp_fold"))
              .join(df.filter(~pl.col("same_class")).group_by("query_domain")
                    .agg(best.alias("fp_class"), pl.len().alias("n_fp_class")),
                    on="query_domain", how="full", coalesce=True))
        better = (lambda x, y: x < y) if smaller else (lambda x, y: x > y)

        def above(fp_col, n_col):
            return ((pl.col(n_col).fill_null(0) == 0)
                    | (pl.col("score").is_not_null()
                       & (pl.col(fp_col).is_null() | better(pl.col("score"), pl.col(fp_col))))).fill_null(False)

        df = (df.with_columns(s.alias("score")).join(fp, on="query_domain", how="left")
              .with_columns(
                  pl.col("score").rank(method="min", descending=not smaller).over("query_domain").alias("rank"),
                  above("fp_fold", "n_fp_fold").alias("above_fold_fp"),
                  above("fp_class", "n_fp_class").alias("above_class_fp")))
        if a.truncate:
            # Keep hits that score no worse than the first different-class hit (ties kept), or
            # every hit of a query with no different-class hit; unscored hits only then. Every
            # same-family hit is kept too: eval_tsv_to_rocx scores only queries with a reported
            # same-family hit, so cutting the last one would drop the query from every level.
            # A kept hit past the cut ranks after the first false positive and changes no count.
            not_worse = (pl.col("score") <= pl.col("fp_class")) if smaller else (pl.col("score") >= pl.col("fp_class"))
            df = df.filter((pl.col("n_fp_class").fill_null(0) == 0) | pl.col("same_family")
                           | (pl.col("score").is_not_null() & (pl.col("fp_class").is_null() | not_worse)))
        info = {**base, "method": f"single:{slug}", "n_searches": 1, "score": col, "n_pairs": n_reported}
        rows += [blank(r, reported_scale) for r in ss.score_column(df, col, smaller, ref, counts, su, True, info)]
        bands += [blank({**info, "status": "ok", **r}, ("n_reported",)) for r in ss.band_rows(df, ident, col, smaller)]

        part = (df.select("query_domain", "target_domain",
                        (pl.col("score") if col == "region_evalue" else pl.lit(None, pl.Float64)).alias("evalue"),
                        "rank", "above_fold_fp", "above_class_fp"))
        long.append(part)
        del df, pairs
        print(f"{slug}: {n_reported} pairs, {part.height} kept", file=sys.stderr)

    merged = (pl.concat(long).group_by("query_domain", "target_domain")
              .agg(pl.col("evalue").min().alias("best_of_n_evalue"),
                   (1.0 / (RRF_K + pl.col("rank"))).sum().alias("rank_fusion"),
                   pl.col("above_fold_fp").any(), pl.col("above_class_fp").any()))
    del long
    df = ss.label_pairs(merged, dom, domain)
    del merged
    n = len(searches)
    methods = [("best_of_n", "best_of_n_evalue", True)]
    if not a.truncate:  # rank fusion needs every hit's rank
        methods.append(("rank_fusion", "rank_fusion", False))
    for method, col, smaller in methods:
        info = {**base, "method": method, "n_searches": n, "score": col, "n_pairs": df.height}
        rows += [blank(r, reported_scale) for r in ss.score_column(df, col, smaller, ref, counts, su, True, info)]
        bands += [blank({**info, "status": "ok", **r}, ("n_reported",)) for r in ss.band_rows(df, ident, col, smaller)]

    # Union upper limit: true pairs found above the first false positive by any search.
    found = {
        "FAM": pl.col("same_family") & pl.col("above_fold_fp"),
        "SFAM": pl.col("same_superfamily") & ~pl.col("same_family") & pl.col("above_fold_fp"),
        "FOLD": pl.col("same_fold") & ~pl.col("same_superfamily") & pl.col("above_fold_fp"),
        "CLASS": pl.col("same_class") & ~pl.col("same_fold") & pl.col("above_class_fp"),
    }
    per_query = (df.group_by("query_domain").agg([e.sum().alias(f"{k}_found") for k, e in found.items()])
                 .with_columns(pl.col("query_domain").cast(pl.Utf8).alias("NAME")))
    level_name = {"FAM": "family", "SFAM": "superfamily", "FOLD": "fold", "CLASS": "class"}
    info = {**base, "method": "union_upper_limit", "n_searches": n, "score": "", "n_pairs": df.height}
    for short, level in level_name.items():
        r = (pl.DataFrame({"NAME": sorted(ref)})
             .join(per_query.select("NAME", f"{short}_found"), on="NAME", how="left")
             .join(counts, on="NAME", how="left")
             .with_columns(pl.when(pl.col(ss.DB_COUNT[short]) > 0)
                           .then((pl.col(f"{short}_found").fill_null(0) / pl.col(ss.DB_COUNT[short])).clip(0, 1))
                           .otherwise(0.0).alias(short)))
        rows.append({**info, "status": "ok", "level": level, "n_queries_ref": r.height,
                     "auc_ref_db": su.sensitivity_stats(r, short)[2]})
    union_band = (ident.join(df.select("query_domain", "target_domain", "above_fold_fp", "above_class_fp"),
                             on=["query_domain", "target_domain"], how="left")
                  .with_columns(pl.col("above_fold_fp").is_not_null().alias("reported"),
                                pl.when(pl.col("category") == "diff_fold_same_class")
                                .then(pl.col("above_class_fp")).otherwise(pl.col("above_fold_fp"))
                                .fill_null(False).alias("above_first_fp"))
                  .group_by("category", "identity_band")
                  .agg(pl.len().alias("n_pairs"), pl.col("reported").sum().alias("n_reported"),
                       pl.col("above_first_fp").sum().alias("n_above_first_fp"),
                       pl.col("query_domain").n_unique().alias("n_queries"))
                  .sort("category", "identity_band"))
    bands += [blank({**info, "status": "ok", **r}, ("n_reported",)) for r in union_band.to_dicts()]

    pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.lit(",".join(searches)).alias("searches")).write_csv(a.out_auc, separator="\t")
    pl.DataFrame(bands, infer_schema_length=None).write_csv(a.out_bands, separator="\t")


if __name__ == "__main__":
    main()
