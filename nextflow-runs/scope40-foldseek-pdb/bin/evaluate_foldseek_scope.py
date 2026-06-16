#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
Evaluate structure/sequence search hits against the SCOPe hierarchy.

Reads easy-search output (query, target, bits, evalue) from FoldSeek or
MMseqs2, joins to SCOPe labels, computes sensitivity-to-first-FP at family /
superfamily / fold levels (TEA / FoldSeek paper convention with gray-zone
exclusion), and writes:

  {label}.rocx        — drop-in for tea_scope40_rocx_files/foldseek.rocx
  {label}_auc.txt     — mean FAM/SFAM/FOLD AUC across all queries

Usage:
  evaluate_foldseek_scope.py --hits hits.tsv.gz --domains scope_domains.tsv \\
      --label foldseek_pdb --outdir .
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def load_hits(path: str) -> pl.DataFrame:
    """Load FoldSeek easy-search output. Handles .gz compression."""
    p = Path(path)
    df = pl.read_csv(
        p,
        separator="\t",
        has_header=False,
        new_columns=["query", "target", "bits", "evalue"],
        schema_overrides={"bits": pl.Float64, "evalue": pl.Float64},
        ignore_errors=True,
    )
    # FoldSeek may append chain info as _CHAIN to the filename stem; strip .pdb extension first
    df = df.with_columns([
        pl.col("query").str.replace(r"\.pdb$", "").alias("query_domain"),
        pl.col("target").str.replace(r"\.pdb$", "").alias("target_domain"),
    ])
    # Filter self-hits
    df = df.filter(pl.col("query_domain") != pl.col("target_domain"))
    return df


def parse_scop_levels(scop_id: str):
    """Parse a.1.1.1 → (class='a', fold='a.1', sfam='a.1.1', fam='a.1.1.1')."""
    parts = scop_id.split(".")
    cls  = parts[0] if len(parts) >= 1 else ""
    fold = ".".join(parts[:2]) if len(parts) >= 2 else cls
    sfam = ".".join(parts[:3]) if len(parts) >= 3 else fold
    fam  = scop_id
    return cls, fold, sfam, fam


def load_domains(path: str) -> pl.DataFrame:
    """Load scope_domains.tsv and explode SCOP levels."""
    df = pl.read_csv(path, separator="\t")
    return df


def sensitivity_to_first_fp(
    hits: pl.DataFrame,
    domains: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute TEA-style sensitivity-to-first-FP at family/superfamily/fold.

    Gray-zone exclusion (TEA/FoldSeek paper convention):
      TP at FAM:  same_family             (FP boundary = NOT same_fold)
      TP at SFAM: same_sfam & !same_fam   (FP boundary = NOT same_fold)
      TP at FOLD: same_fold & !same_sfam  (FP boundary = NOT same_fold)
    All three levels share the same FP boundary (first cross-fold hit).

    Returns ROCX-format DataFrame.
    """
    # Annotate query with SCOP labels
    q_labels = domains.select([
        pl.col("domain_id").alias("query_domain"),
        pl.col("scop_id").alias("q_scop_id"),
        pl.col("scop_family").alias("q_fam"),
        pl.col("scop_superfamily").alias("q_sfam"),
        pl.col("scop_fold").alias("q_fold"),
        pl.col("scop_class").alias("q_class"),
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

    # Gray-zone exclusive TP columns
    df = df.with_columns([
        (pl.col("same_superfamily") & ~pl.col("same_family")).alias("_tp_sfam"),
        (pl.col("same_fold")        & ~pl.col("same_superfamily")).alias("_tp_fold"),
    ])

    def sens_exprs(tp_col: str, label: str):
        n      = pl.col(tp_col).sum().cast(pl.Int32).alias(f"n_{label}")
        has_fp = (~pl.col("same_fold")).any().alias(f"hasfp_{label}")
        # TPs that appear before the first cross-fold FP
        before_fp = (~pl.col("same_fold")).cum_sum() == 0
        tp_bfp = (pl.col(tp_col) & before_fp).sum().cast(pl.Int32).alias(f"tp_bfp_{label}")
        return [n, has_fp, tp_bfp]

    agg = (
        df.group_by("query_domain", maintain_order=True).agg(
            [
                pl.col("q_scop_id").first().alias("SCOP"),
                pl.col("same_family").first().alias("_first_tp"),
                pl.col("q_fam").first().alias("q_fam"),
            ]
            + sens_exprs("same_family",  "fam")
            + sens_exprs("_tp_sfam",     "sfam")
            + sens_exprs("_tp_fold",     "fold")
        )
        .filter(pl.col("n_fam") > 0)  # keep only queries that have any family member hits
    )

    def build_sens(label: str) -> pl.Expr:
        n   = pl.col(f"n_{label}")
        has = pl.col(f"hasfp_{label}")
        tp  = pl.col(f"tp_bfp_{label}").cast(pl.Float64)
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
        .select(["NAME", "SCOP", "FAM", "SFAM", "FOLD", "FP", "FAMCNT", "SFAMCNT", "FOLDCNT"])
        .sort("NAME")
    )
    return rocx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits",    required=True, help="TSV from FoldSeek or MMseqs2 (query target bits evalue)")
    parser.add_argument("--domains", required=True, help="scope_domains.tsv")
    parser.add_argument("--label",   default="hits", help="Output file prefix (e.g. foldseek_pdb, mmseqs2_scope40)")
    parser.add_argument("--outdir",  default=".")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading FoldSeek hits ...", flush=True)
    hits = load_hits(args.hits)
    print(f"  {hits.height:,} non-self hits from {hits['query_domain'].n_unique():,} queries")

    print("Loading SCOPe domain labels ...", flush=True)
    domains = load_domains(args.domains)
    print(f"  {domains.height:,} domains")

    print("Computing sensitivity-to-first-FP ...", flush=True)
    rocx = sensitivity_to_first_fp(hits, domains)

    n_q = rocx.height
    fam_auc  = rocx["FAM"].mean()
    sfam_auc = rocx["SFAM"].mean()
    fold_auc = rocx["FOLD"].mean()
    print(f"\n  Queries with ≥1 family member hit: {n_q:,}")
    print(f"  FAM  AUC = {fam_auc:.4f}")
    print(f"  SFAM AUC = {sfam_auc:.4f}")
    print(f"  FOLD AUC = {fold_auc:.4f}")

    # Write ROCX file
    rocx_path = outdir / f"{args.label}.rocx"
    rocx.write_csv(rocx_path, separator="\t")
    print(f"\nWrote {rocx_path}")

    # Write summary
    summary_path = outdir / f"{args.label}_auc.txt"
    summary_path.write_text(
        f"n_queries\t{n_q}\n"
        f"FAM_AUC\t{fam_auc:.6f}\n"
        f"SFAM_AUC\t{sfam_auc:.6f}\n"
        f"FOLD_AUC\t{fold_auc:.6f}\n"
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
