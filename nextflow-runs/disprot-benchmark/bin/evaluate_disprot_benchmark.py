#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
evaluate_disprot_benchmark.py

Evaluate one tool's results against the DisProt subset ground truth,
stratified by disorder level and disorder location.

Usage:
    evaluate_disprot_benchmark.py \\
        <results.tsv.gz> \\
        <species> \\
        <ground_truth.parquet> \\
        <tool_name> \\
        <disorder_scores.tsv> \\
        <out_metrics.parquet> \\
        <out_pr_curve.parquet>

Input TSV (gzipped, 4 cols, no header):
    query_id  target_id  score  [pvalue]
    score: higher = more similar (containment, bitscore, etc.)

Disorder stratification bins (from disorder_scores.tsv):
    ordered    : mean_disorder < 0.2
    partial    : 0.2 – 0.5
    disordered : > 0.5
"""

import gzip
import sys
from io import StringIO

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


DISORDER_CATEGORIES = ["ordered", "partial", "disordered", "all"]


def extract_accession(id_str: str) -> str:
    if "|" in id_str:
        parts = id_str.split("|")
        if len(parts) >= 2:
            return parts[1]
    return id_str


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute AUC-ROC, AUC-PR, and recall-at-thresholds for one stratum."""
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {
            "auc_roc": float("nan"), "auc_pr": float("nan"),
            "sens_at_99prec": float("nan"), "recall_at_fdr01": float("nan"),
            "recall_at_fdr05": float("nan"), "n_pairs": len(y_true),
            "n_positives": int(y_true.sum()), "n_found": int((y_score > 0).sum()),
        }

    auc_roc = float(roc_auc_score(y_true, y_score))
    auc_pr  = float(average_precision_score(y_true, y_score))

    prec, rec, thr = precision_recall_curve(y_true, y_score)

    # Sensitivity at 99% precision (FDR ≤ 1%)
    mask99 = prec[:-1] >= 0.99
    sens99 = float(rec[:-1][mask99].max()) if mask99.any() else 0.0

    # Recall at FDR ≤ 1% (precision ≥ 0.99)
    recall_fdr01 = sens99

    # Recall at FDR ≤ 5% (precision ≥ 0.95)
    mask95 = prec[:-1] >= 0.95
    recall_fdr05 = float(rec[:-1][mask95].max()) if mask95.any() else 0.0

    return {
        "auc_roc":         auc_roc,
        "auc_pr":          auc_pr,
        "sens_at_99prec":  sens99,
        "recall_at_fdr01": recall_fdr01,
        "recall_at_fdr05": recall_fdr05,
        "n_pairs":         len(y_true),
        "n_positives":     int(y_true.sum()),
        "n_found":         int((y_score > 0).sum()),
    }


def run_evaluation(
    tool_results_gz: str,
    species: str,
    ground_truth_parquet: str,
    tool_name: str,
    disorder_scores_tsv: str,
    metrics_out: str,
    pr_curve_out: str,
) -> None:

    # ------------------------------------------------------------------
    # Load ground truth (already filtered to DisProt proteins)
    # ------------------------------------------------------------------
    gt = pl.read_parquet(ground_truth_parquet)
    if "label" not in gt.columns:
        print(f"ERROR: no 'label' column in {ground_truth_parquet}", file=sys.stderr)
        sys.exit(1)
    gt = gt.with_columns(pl.col("label").cast(pl.Int8))

    # ------------------------------------------------------------------
    # Load disorder scores
    # ------------------------------------------------------------------
    disorder = pl.read_csv(disorder_scores_tsv, separator="\t", infer_schema_length=10_000)
    disorder = disorder.select([
        "uniprot_acc", "mean_disorder", "max_window_disorder",
        "fraction_disordered", "disorder_category",
    ])

    # Attach disorder scores to ground truth (join on human_accession)
    gt = gt.join(
        disorder.rename({"uniprot_acc": "human_accession"}),
        on="human_accession",
        how="left",
    ).with_columns(
        pl.col("disorder_category").fill_null("unknown"),
        pl.col("mean_disorder").fill_null(0.5),
    )

    # ------------------------------------------------------------------
    # Load tool results
    # ------------------------------------------------------------------
    with gzip.open(tool_results_gz, "rt") as fh:
        content = fh.read()

    if not content.strip():
        print(f"WARNING: {tool_results_gz} is empty — writing zero metrics", file=sys.stderr)
        _write_empty(tool_name, species, gt, DISORDER_CATEGORIES, metrics_out, pr_curve_out)
        return

    results = pl.read_csv(
        StringIO(content),
        separator="\t",
        has_header=False,
        new_columns=["query_raw", "target_raw", "score"],
        columns=[0, 1, 2],
        infer_schema_length=50_000,
        comment_prefix="#",
    ).with_columns([
        pl.col("query_raw")
          .map_elements(extract_accession, return_dtype=pl.String)
          .alias("human_accession"),
        pl.col("target_raw")
          .map_elements(extract_accession, return_dtype=pl.String)
          .alias("species_accession"),
        pl.col("score").cast(pl.Float64),
    ]).select(["human_accession", "species_accession", "score"])

    # Best score per pair
    results = results.group_by(["human_accession", "species_accession"]).agg(
        pl.col("score").max()
    )

    # Join with ground truth; missing pairs get score = 0
    merged = gt.join(results, on=["human_accession", "species_accession"], how="left")
    merged = merged.with_columns(pl.col("score").fill_null(0.0))

    # ------------------------------------------------------------------
    # Compute metrics per disorder stratum
    # ------------------------------------------------------------------
    metrics_rows = []
    pr_rows = []

    for cat in DISORDER_CATEGORIES:
        if cat == "all":
            stratum = merged
        else:
            stratum = merged.filter(pl.col("disorder_category") == cat)

        if len(stratum) == 0:
            continue

        y_true  = stratum["label"].to_numpy().astype(np.int8)
        y_score = stratum["score"].to_numpy().astype(np.float64)

        m = compute_metrics(y_true, y_score)

        metrics_rows.append({
            "tool":              tool_name,
            "species":           species,
            "disorder_category": cat,
            **m,
        })

        if cat == "all" and y_true.sum() > 0 and y_true.sum() < len(y_true):
            prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_score)
            pr_rows.append(pl.DataFrame({
                "tool":              tool_name,
                "species":           species,
                "disorder_category": cat,
                "precision":         prec_arr[:-1].tolist(),
                "recall":            rec_arr[:-1].tolist(),
                "threshold":         thr_arr.tolist(),
            }))

        print(
            f"{tool_name:25s} vs {species:12s}  [{cat:12s}]  "
            f"AUC-PR={m.get('auc_pr', float('nan')):.4f}  "
            f"recall@FDR5%={m.get('recall_at_fdr05', float('nan')):.4f}  "
            f"n={len(stratum)}"
        )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    metrics_df = pl.DataFrame(metrics_rows)
    metrics_df.write_parquet(metrics_out)

    if pr_rows:
        pl.concat(pr_rows).write_parquet(pr_curve_out)
    else:
        pl.DataFrame({
            "tool": [], "species": [], "disorder_category": [],
            "precision": [], "recall": [], "threshold": [],
        }).write_parquet(pr_curve_out)


def _write_empty(tool_name, species, gt, cats, metrics_out, pr_curve_out):
    rows = []
    for cat in cats:
        stratum = gt if cat == "all" else gt.filter(pl.col("disorder_category") == cat)
        rows.append({
            "tool": tool_name, "species": species, "disorder_category": cat,
            "auc_roc": float("nan"), "auc_pr": float("nan"),
            "sens_at_99prec": float("nan"), "recall_at_fdr01": float("nan"),
            "recall_at_fdr05": float("nan"),
            "n_pairs": len(stratum),
            "n_positives": int((stratum["label"] == 1).sum()) if len(stratum) else 0,
            "n_found": 0,
        })
    pl.DataFrame(rows).write_parquet(metrics_out)
    pl.DataFrame({
        "tool": [], "species": [], "disorder_category": [],
        "precision": [], "recall": [], "threshold": [],
    }).write_parquet(pr_curve_out)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(__doc__)
        sys.exit(1)

    run_evaluation(
        tool_results_gz=sys.argv[1],
        species=sys.argv[2],
        ground_truth_parquet=sys.argv[3],
        tool_name=sys.argv[4],
        disorder_scores_tsv=sys.argv[5],
        metrics_out=sys.argv[6],
        pr_curve_out=sys.argv[7],
    )
