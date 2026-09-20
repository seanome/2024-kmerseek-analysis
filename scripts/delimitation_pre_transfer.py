"""Experiment 2: boundary placement of pre-transfer regions against Pfam domains.

For every human Pfam domain instance in the truth and every tool arm, look at the arm's
raw regions on that query (any target, no label transfer) and record whether any region
covers the domain (overlap >= half the domain), whether any region delimits it (IoU >= 0.5)
and the best IoU. The axis of interest is the domain's length as a fraction of its protein.

Inputs are the midi-plus extract's region tables (`mhc_kmerseek_regions.parquet`,
`mhc_baseline_regions.parquet`, 255 MHC-window queries) by default; point --kmerseek and
--baseline at the full-run region files once they are pulled from Sherlock
(results/regions/*.parquet under the midi-plus outdir). Output: one row per (arm, domain).

Usage:
    python scripts/delimitation_pre_transfer.py --out /Users/olga/data/qfo-pfam-region-midi-plus/238_delimitation_pre_transfer.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

E = Path("/Users/olga/data/qfo-pfam-region-midi-plus/extract")


def score(regions: pl.DataFrame, truth: pl.DataFrame, arm: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """regions: query_acc, qstart, qend, already restricted to the arm's significant hits.

    Returns (per-domain table, per-region-hit table). Per domain: best IoU and best cover
    over the query's regions, plus covered (>= half the domain) and delimited (IoU >= 0.5).
    Per region hit: the IoU of every region with the domain it overlaps, which is the
    distribution that says whether an arm's calls are region-shaped or chain-shaped.
    """
    j = truth.join(regions, left_on="accession", right_on="query_acc", how="left")
    # min_horizontal/max_horizontal skip nulls, so an unmatched left join would score as a
    # full overlap; guard on the join key first (this repo's memory: polars_horizontal_skips_nulls).
    j = j.with_columns(
        pl.when(pl.col("qstart").is_null()).then(0)
          .otherwise((pl.min_horizontal("qend", "domain_end") - pl.max_horizontal("qstart", "domain_start")).clip(lower_bound=0)).alias("ov"))
    j = j.with_columns(
        (pl.col("ov") / (pl.max_horizontal("qend", "domain_end") - pl.min_horizontal("qstart", "domain_start"))).fill_null(0.0).alias("iou"),
        (pl.col("ov") / (pl.col("domain_end") - pl.col("domain_start"))).fill_null(0.0).alias("cover"))
    per_domain = (j.group_by("accession", "pfam_id", "domain_start", "domain_end", "protein_length")
                   .agg(pl.col("iou").max().alias("best_iou"), pl.col("cover").max().alias("best_cover"), (pl.col("qstart").is_not_null()).any().alias("query_has_regions"))
                   .with_columns((pl.col("best_cover") >= 0.5).alias("covered"), (pl.col("best_iou") >= 0.5).alias("delimited"), pl.lit(arm).alias("arm")))
    per_hit = (j.filter(pl.col("ov") > 0)
                .select("accession", "pfam_id", "domain_start", "domain_end", "protein_length", "qstart", "qend", "iou", "cover")
                .with_columns(pl.lit(arm).alias("arm")))
    return per_domain, per_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmerseek", default=str(E / "mhc_kmerseek_regions.parquet"))
    ap.add_argument("--baseline", default=str(E / "mhc_baseline_regions.parquet"))
    ap.add_argument("--truth", default=str(E / "human_domain_truth.parquet"))
    ap.add_argument("--species", default=None, help="restrict regions to one target species (default: pool all)")
    ap.add_argument("--max-evalue", type=float, default=1e-3, help="baseline regions kept at evalue <= this")
    ap.add_argument("--min-poisson-score", type=float, default=3.0, help="kmerseek regions kept at region_poisson_score >= this (3 = p <= 1e-3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    truth = pl.read_parquet(args.truth).select("accession", "pfam_id", "domain_start", "domain_end", "protein_length").unique()

    km = pl.scan_parquet(args.kmerseek)
    # The truth covers every query of the run; the region tables may cover a subset (the
    # MHC extract has 255 of 998). Score only the queries the region tables were built for.
    searched = set(km.select("query_acc").unique().collect()["query_acc"]) | set(pl.scan_parquet(args.baseline).select("query_acc").unique().collect()["query_acc"])
    truth = truth.filter(pl.col("accession").is_in(list(searched)))
    print(f"{truth.height} domain instances on {truth['accession'].n_unique()} searched queries", flush=True)
    if args.species:
        km = km.filter(pl.col("species") == args.species)
    arms = km.select("alphabet", "ksize", "lc").unique().collect()
    out, hits = [], []
    for a, k, lc in arms.iter_rows():
        r = (km.filter((pl.col("alphabet") == a) & (pl.col("ksize") == k) & (pl.col("lc") == lc) & (pl.col("region_poisson_score") >= args.min_poisson_score))
               .select("query_acc", pl.col("region_start").cast(pl.Int64).alias("qstart"), pl.col("region_end").cast(pl.Int64).alias("qend")).collect())
        d, h = score(r, truth, f"kmerseek.{a}_k{k}_lc{lc}"); out.append(d); hits.append(h)
        print("kmerseek", a, k, lc, r.height, flush=True)
    bl = pl.scan_parquet(args.baseline)
    if args.species:
        bl = bl.filter(pl.col("species") == args.species)
    for (tool,) in bl.select("tool").unique().collect().iter_rows():
        r = (bl.filter((pl.col("tool") == tool) & (pl.col("evalue").cast(pl.Float64) <= args.max_evalue))
               .select("query_acc", pl.col("qstart").cast(pl.Int64), pl.col("qend").cast(pl.Int64)).collect())
        d, h = score(r, truth, tool); out.append(d); hits.append(h)
        print(tool, r.height, flush=True)
    df = pl.concat(out)
    df.write_parquet(args.out)
    pl.concat(hits).write_parquet(str(args.out).replace(".parquet", ".hits.parquet"))
    print(df.group_by("arm").agg(pl.col("covered").mean().round(3), pl.col("delimited").mean().round(3), pl.len()).sort("delimited", descending=True))


if __name__ == "__main__":
    main()
