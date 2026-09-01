#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
evaluate_benchmark.py

Evaluate one tool's search results against the Pfam subdomain benchmark ground truth.

Called from main.nf's evaluateBenchmark process, but also usable standalone:

    python evaluate_benchmark.py \\
        human_vs_mouse.diamond.tsv.gz \\
        mouse \\
        human_vs_mouse_ground_truth.parquet \\
        diamond \\
        diamond.mouse.metrics.parquet \\
        diamond.mouse.pr_curve.parquet

Input TSV (gzipped, no header, tab-separated):
    query_id  target_id  score  [evalue]

    - query_id:  human protein identifier  (e.g. sp|P12345|GENE_HUMAN or P12345)
    - target_id: species protein identifier
    - score:     alignment bitscore or similarity score — HIGHER = MORE SIMILAR
    - evalue:    (optional, ignored)

For annotation-based tools (hmmer3_hmmscan, psalm), the scores are pre-computed
pair scores (max shared domain bitscore or Jaccard).

Missing pairs (protein pair not in tool output) are assigned score = 0.

Output:
    metrics.parquet  — one row with AUC-ROC, AUC-PR, and related stats
    pr_curve.parquet — full precision/recall curve
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


def extract_accession(id_str: str) -> str:
    """Extract UniProt accession from full header.
    'sp|P12345|GENE_HUMAN' -> 'P12345'
    'P12345'               -> 'P12345'
    """
    if "|" in id_str:
        parts = id_str.split("|")
        if len(parts) >= 2:
            return parts[1]
    return id_str


def run_evaluation(
    tool_results_gz: str,
    species: str,
    ground_truth_parquet: str,
    tool_name: str,
    metrics_out: str,
    pr_curve_out: str,
) -> None:

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    gt = pl.read_parquet(ground_truth_parquet).select(
        ["human_accession", "species_accession", "label"]
    )
    gt = gt.with_columns(pl.col("label").cast(pl.Int8))

    n_pos = int(gt["label"].sum())
    n_neg = len(gt) - n_pos

    # ------------------------------------------------------------------
    # Load tool results
    # ------------------------------------------------------------------
    with gzip.open(tool_results_gz, "rt") as fh:
        content = fh.read()

    if not content.strip():
        print(f"WARNING: {tool_results_gz} is empty — skipping {tool_name} vs {species}")
        # Write zero-score metrics so the file exists for aggregation
        _write_empty_metrics(tool_name, species, n_pos, n_neg, metrics_out, pr_curve_out)
        return

    results = pl.read_csv(
        StringIO(content),
        separator="\t",
        has_header=False,
        new_columns=["query_raw", "target_raw", "score"],
        columns=[0, 1, 2],
        infer_schema_length=50_000,
        comment_prefix="#",
    )

    results = results.with_columns(
        [
            pl.col("query_raw")
            .map_elements(extract_accession, return_dtype=pl.String)
            .alias("human_accession"),
            pl.col("target_raw")
            .map_elements(extract_accession, return_dtype=pl.String)
            .alias("species_accession"),
            pl.col("score").cast(pl.Float64),
        ]
    ).select(["human_accession", "species_accession", "score"])

    # Keep best score per pair (multiple HSPs / domain hits)
    results = results.group_by(["human_accession", "species_accession"]).agg(
        pl.col("score").max()
    )

    # ------------------------------------------------------------------
    # Join with ground truth; missing pairs get score = 0
    # ------------------------------------------------------------------
    merged = gt.join(results, on=["human_accession", "species_accession"], how="left")
    merged = merged.with_columns(pl.col("score").fill_null(0.0))

    n_found = int((merged["score"] > 0).sum())

    # ------------------------------------------------------------------
    # Compute AUC-ROC and AUC-PR
    # ------------------------------------------------------------------
    y_true = merged["label"].to_numpy().astype(np.int8)
    y_score = merged["score"].to_numpy().astype(np.float64)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print(f"WARNING: degenerate labels for {tool_name} vs {species} — skipping")
        _write_empty_metrics(tool_name, species, n_pos, n_neg, metrics_out, pr_curve_out)
        return

    auc_roc = float(roc_auc_score(y_true, y_score))
    auc_pr = float(average_precision_score(y_true, y_score))

    # Sensitivity at 99% specificity: largest recall where precision >= 0.99
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_true, y_score)
    high_prec_mask = precision_arr[:-1] >= 0.99
    sens_at_99prec = float(recall_arr[:-1][high_prec_mask].max()) if high_prec_mask.any() else 0.0

    # Precision at 50% recall
    high_rec_mask = recall_arr[:-1] >= 0.50
    prec_at_50rec = float(precision_arr[:-1][high_rec_mask].max()) if high_rec_mask.any() else 0.0

    # ------------------------------------------------------------------
    # Write metrics
    # ------------------------------------------------------------------
    metrics_df = pl.DataFrame(
        {
            "tool": [tool_name],
            "species": [species],
            "auc_roc": [auc_roc],
            "auc_pr": [auc_pr],
            "sens_at_99prec": [sens_at_99prec],
            "prec_at_50rec": [prec_at_50rec],
            "n_positives": [n_pos],
            "n_negatives": [n_neg],
            "n_found": [n_found],
        }
    )
    metrics_df.write_parquet(metrics_out)

    # PR curve (exclude the final fence-post point added by sklearn)
    pr_df = pl.DataFrame(
        {
            "tool": tool_name,
            "species": species,
            "precision": precision_arr[:-1].tolist(),
            "recall": recall_arr[:-1].tolist(),
            "threshold": thresholds_arr.tolist(),
        }
    )
    pr_df.write_parquet(pr_curve_out)

    print(
        f"{tool_name:25s} vs {species:12s}  "
        f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
        f"found={n_found}/{len(gt)}"
    )


def _write_empty_metrics(tool_name, species, n_pos, n_neg, metrics_out, pr_curve_out):
    pl.DataFrame(
        {
            "tool": [tool_name],
            "species": [species],
            "auc_roc": [float("nan")],
            "auc_pr": [float("nan")],
            "sens_at_99prec": [float("nan")],
            "prec_at_50rec": [float("nan")],
            "n_positives": [n_pos],
            "n_negatives": [n_neg],
            "n_found": [0],
        }
    ).write_parquet(metrics_out)
    pl.DataFrame(
        {"tool": [], "species": [], "precision": [], "recall": [], "threshold": []}
    ).write_parquet(pr_curve_out)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)

    run_evaluation(
        tool_results_gz=sys.argv[1],
        species=sys.argv[2],
        ground_truth_parquet=sys.argv[3],
        tool_name=sys.argv[4],
        metrics_out=sys.argv[5],
        pr_curve_out=sys.argv[6],
    )
