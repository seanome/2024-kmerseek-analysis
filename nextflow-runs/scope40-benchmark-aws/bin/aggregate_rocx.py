#!/usr/bin/env python3
"""
aggregate_rocx.py

Collect all .rocx files from a directory and produce a summary TSV with
mean AUC per tool at family / superfamily / fold levels.

The .rocx format (TEA/FoldSeek paper convention):
  NAME  SCOP  FAM  SFAM  FOLD  FP  FAMCNT  SFAMCNT  FOLDCNT

Usage:
    aggregate_rocx.py <rocx_dir> <output_summary.tsv>
"""

import sys
from pathlib import Path

import polars as pl


TOOL_ORDER = [
    "foldseek", "foldseek_esmfold",
    "tea",
    "plmsearch",
    "mmseqs2", "blast", "diamond",
    "kmerseek",
]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    rocx_dir = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])

    rows = []
    for rocx_file in sorted(rocx_dir.glob("*.rocx")):
        tool = rocx_file.stem
        df   = pl.read_csv(rocx_file, separator="\t")
        n         = len(df)
        fam_auc   = df["FAM"].mean()  or 0.0
        sfam_auc  = df["SFAM"].mean() or 0.0
        fold_auc  = df["FOLD"].mean() or 0.0
        rows.append({
            "tool":      tool,
            "n_queries": n,
            "FAM_AUC":   round(fam_auc,  6),
            "SFAM_AUC":  round(sfam_auc, 6),
            "FOLD_AUC":  round(fold_auc, 6),
        })
        print(f"  {tool:30s}  FAM={fam_auc:.4f}  SFAM={sfam_auc:.4f}  FOLD={fold_auc:.4f}  n={n:,}")

    out = (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("tool")
              .map_elements(lambda t: TOOL_ORDER.index(t) if t in TOOL_ORDER else 99,
                            return_dtype=pl.Int32)
              .alias("_order")
        )
        .sort("_order")
        .drop("_order")
    )

    out.write_csv(out_path, separator="\t")
    print(f"\nWrote {len(rows)} tool(s) to {out_path}")

    # Pretty-print the summary table
    print("\n── SCOPe40 benchmark summary ──────────────────────────────────")
    print(f"{'Tool':<30}  {'FAM AUC':>8}  {'SFAM AUC':>9}  {'FOLD AUC':>9}  {'N queries':>10}")
    print("─" * 72)
    for r in out.iter_rows(named=True):
        print(f"{r['tool']:<30}  {r['FAM_AUC']:>8.4f}  {r['SFAM_AUC']:>9.4f}  "
              f"{r['FOLD_AUC']:>9.4f}  {r['n_queries']:>10,}")


if __name__ == "__main__":
    main()
