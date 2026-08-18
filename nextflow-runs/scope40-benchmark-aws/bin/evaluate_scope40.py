#!/usr/bin/env python3
"""
evaluate_scope40.py

Evaluate structure/sequence search hits against the SCOPe hierarchy.

Reads easy-search output (query, target, bits, evalue — no header) from any
tool (FoldSeek, MMseqs2, BLAST+, DIAMOND, Kmerseek, PLMSearch, TEA), joins
to SCOPe labels, computes sensitivity-to-first-FP at family / superfamily /
fold levels using TEA/FoldSeek-paper convention (gray-zone exclusion), writes:

  {label}.rocx       — per-query sensitivity at FAM/SFAM/FOLD
  {label}_auc.txt    — mean AUC summary

Usage:
  evaluate_scope40.py --hits hits.tsv.gz --domains scope_domains.tsv \\
      --label foldseek --outdir .
"""

import argparse
from pathlib import Path

import polars as pl


def load_hits(path: str) -> pl.DataFrame:
    p = Path(path)
    try:
        df = pl.read_csv(
            p,
            separator="\t",
            has_header=False,
            new_columns=["query", "target", "bits", "evalue"],
            schema_overrides={"bits": pl.Float64, "evalue": pl.Float64},
            ignore_errors=True,
        )
    except pl.exceptions.NoDataError:
        # Native pre-filtering (kmerseek --min-shared-kmers/--max-pvalue) or a small
        # test-mode FASTA can legitimately leave zero hits for a given encoding/ksize.
        df = pl.DataFrame(schema={"query": pl.String, "target": pl.String,
                                   "bits": pl.Float64, "evalue": pl.Float64})
    # FoldSeek appends .pdb to filenames — strip for all tools for consistency
    df = df.with_columns([
        pl.col("query").str.replace(r"\.pdb$", "").alias("query_domain"),
        pl.col("target").str.replace(r"\.pdb$", "").alias("target_domain"),
    ])
    df = df.filter(pl.col("query_domain") != pl.col("target_domain"))
    return df


def load_domains(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t")


def sensitivity_to_first_fp(hits: pl.DataFrame, domains: pl.DataFrame) -> pl.DataFrame:
    """
    TEA-style sensitivity-to-first-FP at family/superfamily/fold.

    Gray-zone exclusion (shared FP boundary = first cross-fold hit):
      FAM:  TP = same_family
      SFAM: TP = same_sfam & !same_fam
      FOLD: TP = same_fold & !same_sfam
    """
    q_labels = domains.select([
        pl.col("domain_id").alias("query_domain"),
        pl.col("scop_id").alias("q_scop_id"),
        pl.col("scop_family").alias("q_fam"),
        pl.col("scop_superfamily").alias("q_sfam"),
        pl.col("scop_fold").alias("q_fold"),
    ])
    t_labels = domains.select([
        pl.col("domain_id").alias("target_domain"),
        pl.col("scop_family").alias("t_fam"),
        pl.col("scop_superfamily").alias("t_sfam"),
        pl.col("scop_fold").alias("t_fold"),
    ])

    df = (
        hits
        .join(q_labels, on="query_domain", how="inner")
        .join(t_labels, on="target_domain", how="inner")
        .with_columns([
            (pl.col("q_fam")  == pl.col("t_fam") ).alias("same_family"),
            (pl.col("q_sfam") == pl.col("t_sfam")).alias("same_superfamily"),
            (pl.col("q_fold") == pl.col("t_fold")).alias("same_fold"),
        ])
        .sort(["query_domain", "bits"], descending=[False, True], nulls_last=True)
    )

    df = df.with_columns([
        (pl.col("same_superfamily") & ~pl.col("same_family")).alias("_tp_sfam"),
        (pl.col("same_fold")        & ~pl.col("same_superfamily")).alias("_tp_fold"),
    ])

    def sens_exprs(tp_col: str, label: str):
        return [
            pl.col(tp_col).sum().cast(pl.Int32).alias(f"n_{label}"),
            (~pl.col("same_fold")).any().alias(f"hasfp_{label}"),
            (pl.col(tp_col) & ((~pl.col("same_fold")).cum_sum() == 0))
                .sum().cast(pl.Int32).alias(f"tp_bfp_{label}"),
        ]

    agg = (
        df.group_by("query_domain", maintain_order=True).agg(
            [pl.col("q_scop_id").first().alias("SCOP"),
             pl.col("same_family").first().alias("_first_tp"),
             pl.col("q_fam").first().alias("q_fam")]
            + sens_exprs("same_family", "fam")
            + sens_exprs("_tp_sfam",    "sfam")
            + sens_exprs("_tp_fold",    "fold")
        )
        .filter(pl.col("n_fam") > 0)
    )

    def build_sens(lbl: str) -> pl.Expr:
        n   = pl.col(f"n_{lbl}")
        has = pl.col(f"hasfp_{lbl}")
        tp  = pl.col(f"tp_bfp_{lbl}").cast(pl.Float64)
        return (
            pl.when(n == 0).then(0.0)
              .when(~has).then(1.0)
              .when(tp == 0).then(0.0)
              .otherwise((tp / n.cast(pl.Float64)).clip(0.0, 1.0))
        )

    rocx = (
        agg
        .with_columns([
            build_sens("fam").alias("FAM"),
            build_sens("sfam").alias("SFAM"),
            build_sens("fold").alias("FOLD"),
            pl.when(pl.col("_first_tp")).then(0).otherwise(1).cast(pl.Int32).alias("FP"),
        ])
        .rename({"query_domain": "NAME", "n_fam": "FAMCNT",
                 "n_sfam": "SFAMCNT", "n_fold": "FOLDCNT"})
        .select(["NAME", "SCOP", "FAM", "SFAM", "FOLD", "FP",
                 "FAMCNT", "SFAMCNT", "FOLDCNT"])
        .sort("NAME")
    )
    return rocx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits",    required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--label",   default="hits")
    ap.add_argument("--outdir",  default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading hits ...", flush=True)
    hits = load_hits(args.hits)
    print(f"  {hits.height:,} non-self hits  {hits['query_domain'].n_unique():,} queries")

    print("Loading SCOPe domain labels ...", flush=True)
    domains = load_domains(args.domains)
    print(f"  {domains.height:,} domains")

    print("Computing sensitivity-to-first-FP ...", flush=True)
    rocx = sensitivity_to_first_fp(hits, domains)

    n_q      = rocx.height
    fam_auc  = rocx["FAM"].mean()  or 0.0
    sfam_auc = rocx["SFAM"].mean() or 0.0
    fold_auc = rocx["FOLD"].mean() or 0.0
    print(f"\n  Queries with ≥1 family member: {n_q:,}")
    print(f"  FAM  AUC = {fam_auc:.4f}")
    print(f"  SFAM AUC = {sfam_auc:.4f}")
    print(f"  FOLD AUC = {fold_auc:.4f}")

    rocx_path = outdir / f"{args.label}.rocx"
    rocx.write_csv(rocx_path, separator="\t")
    print(f"\nWrote {rocx_path}")

    auc_path = outdir / f"{args.label}_auc.txt"
    auc_path.write_text(
        f"n_queries\t{n_q}\n"
        f"FAM_AUC\t{fam_auc:.6f}\n"
        f"SFAM_AUC\t{sfam_auc:.6f}\n"
        f"FOLD_AUC\t{fold_auc:.6f}\n"
    )
    print(f"Wrote {auc_path}")


if __name__ == "__main__":
    main()
