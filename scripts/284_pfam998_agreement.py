"""Notebook 284, part 3: on proteins with known Pfam domains, does agreement between
alphabets make a call more likely to be right, once each alphabet's calls pass an E-value cut?

Data: notebook 243's all-against-all search of the 998 Pfam-annotated human proteins
(PR 48, 243_pfam998_search.py), every one of notebook 241's 152 alphabet-ksize settings,
every region kept (/Users/olga/data/alphabet-logreg-pfam998/regions/, 235_895_940 regions).
Self matches (query = target) are dropped. E-values are for this 998-protein database.

A spot is built as in alphabet_agreement_utils (same diagonal, overlapping query
stretches). A spot is correct when one Pfam family covers at least 20% of its query
stretch and at least 20% of its target stretch (PR 44's rule, as in notebook 243).

For each E-value cut: precision at each agreement count, for the three counts, and the
average precision of four rankings of all spots: lowest E-value alone, and each count
with ties broken by the lowest E-value.

Output: notebooks/284_pfam998_precision_by_agreement.csv, notebooks/284_pfam998_average_precision.csv
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import alphabet_agreement_utils as au  # noqa: E402

REGIONS = Path("/Users/olga/data/alphabet-logreg-pfam998/regions")
TRUTH = Path("/Users/olga/data/qfo-pfam-region-benchmark/midi-plus/truth/human_domain_truth.parquet")
CUTS = [1e5, 1e4, 1e3, 1e2, 10, 1]


def label(s: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """correct: one Pfam family covers >= 20% of the query stretch and of the target stretch."""
    dom = truth.select("accession", pl.col("pfam_id").alias("fam"),
                       (pl.col("domain_start") - 1).alias("d0"), pl.col("domain_end").alias("d1")).lazy()
    s = s.with_row_index("sid")
    sides = []
    for acc, s0, s1 in (("query_name", "qs", "qe"), ("target_name", "ts", "te")):
        x = (s.lazy().select("sid", pl.col(acc).alias("accession"), pl.col(s0).alias("s0"), pl.col(s1).alias("s1"))
             .join(dom, on="accession")
             .with_columns(o=pl.min_horizontal("s1", "d1") - pl.max_horizontal("s0", "d0"))
             .filter(pl.col("o") >= 0.2 * pl.max_horizontal(pl.col("s1") - pl.col("s0"), pl.lit(1)))
             .select("sid", "fam").unique())
        sides.append(x)
    ok = sides[0].join(sides[1], on=["sid", "fam"]).select("sid").unique().with_columns(correct=pl.lit(True))
    return (s.lazy().join(ok, on="sid", how="left").with_columns(pl.col("correct").fill_null(False))
            .drop("sid").collect())


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    """Mean of the precision at each correct spot, spots sorted by score, highest first.
    Ties are broken by the order given, so callers sort by a tie-breaker first."""
    order = np.argsort(-score, kind="stable")
    yy = y[order]
    hits = np.cumsum(yy)
    prec = hits / np.arange(1, len(yy) + 1)
    return float(prec[yy].mean()) if yy.any() else float("nan")


def main():
    truth = pl.read_parquet(TRUTH)
    regions = (pl.scan_parquet(sorted(REGIONS.glob("*.parquet")))
               .filter(pl.col("query_name") != pl.col("target_name"))
               .with_columns(gene=pl.col("target_name")))
    d = au.prepare(regions)
    prec_rows, ap_rows = [], []
    for cut in CUTS:
        s = au.spots(d.filter(pl.col("region_evalue") < cut).with_columns(e=pl.col("region_evalue")), None)
        # lowest E-value per spot (spots() does not keep it); recompute with the same keys
        emin = (d.filter(pl.col("region_evalue") < cut)
                .sort("query_name", "target_name", "diag", "qs")
                .with_columns(prev_end=pl.col("qe").cum_max().shift(1).over("query_name", "target_name", "diag"))
                .with_columns(spot=(pl.col("prev_end").is_null() | (pl.col("qs") >= pl.col("prev_end")))
                              .cast(pl.Int32).cum_sum().over("query_name", "target_name", "diag"))
                .group_by("query_name", "target_name", "diag", "spot").agg(e_min=pl.col("region_evalue").min())
                .collect())
        s = s.join(emin, on=["query_name", "target_name", "diag", "spot"])
        s = s.with_columns(ts=pl.col("qs") + pl.col("diag"), te=pl.col("qe") + pl.col("diag"))
        s = label(s, truth)
        n, npos = s.height, int(s["correct"].sum())
        print(f"E < {cut:g}: {n} spots, {npos} correct ({npos / n:.4f})", flush=True)
        for c in au.COUNTS:
            t = (s.group_by(c).agg(n_spots=pl.len(), n_correct=pl.col("correct").sum())
                 .with_columns(precision=pl.col("n_correct") / pl.col("n_spots")).sort(c)
                 .rename({c: "agreement"}).with_columns(count=pl.lit(c), e_cut=pl.lit(float(cut)), agreement=pl.col("agreement").cast(pl.Int64)))
            prec_rows.append(t)
        # rankings: tie-breaker first (lower E better), then a stable sort on the main score
        s = s.sort("e_min", "query_name", "target_name", "diag", "qs")
        y = s["correct"].to_numpy()
        e_score = -np.log10(np.maximum(s["e_min"].to_numpy(), 1e-300))
        row = dict(e_cut=float(cut), n_spots=n, n_correct=npos, base_rate=npos / n,
                   ap_lowest_e=average_precision(e_score, y))
        for c in au.COUNTS:
            row[f"ap_{c}_then_e"] = average_precision(s[c].to_numpy().astype(float), y)
        print(row, flush=True)
        ap_rows.append(row)
    pl.concat(prec_rows).select("e_cut", "count", "agreement", "n_spots", "n_correct", "precision") \
        .write_csv(ROOT / "notebooks/284_pfam998_precision_by_agreement.csv")
    pl.DataFrame(ap_rows).write_csv(ROOT / "notebooks/284_pfam998_average_precision.csv")


if __name__ == "__main__":
    main()
