#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""Score one (encoding, ksize) combo under every composite metric — one Nextflow task per combo.

This is notebook 200 section 2b's sweep body, pulled out of the notebook's own for-loop.
That loop scored ~100 combos sequentially inside the notebook, checkpointing by rewriting a CSV
after each combo — resumable only in the sense that a restarted notebook could skip already-done
combos, with no parallelism and no protection against losing an in-progress combo: on 2026-08-05,
hp_lehninger/hp_lehninger_plus_c k18 (364M/429M rows) thrashed this machine's 128GB RAM for 5+
days near-zero-progress before an OOM-kill with nothing saved, since the checkpoint only flushed
BETWEEN combos, not mid-combo. One task per combo here instead: real parallelism across combos,
and Nextflow's storeDir gives free per-combo resumability without a notebook having to track a
"done" set itself.

Usage:
    compute_metric_leaderboard_combo.py \\
        --dash-encoding hp-thomas-dill --display-encoding hp_thomas_dill --ksize 26 \\
        --data-dir /Users/olga/data/gencode/results-human-mouse-orthologs \\
        --output 200_metric_leaderboard.hp-thomas-dill.k26.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "/Users/olga/code/2024-kmerseek-analysis/notebooks")
import ortholog_analysis_utils as ou  # noqa: E402

COMPOSITE_COLUMNS = ["query_name", "target_name", "jaccard", "containment",
                     "n_intersecting_hashes", "expected_shared_kmers",
                     "poisson_pvalue", "enrichment", "query_tfidf", "mean_matched_kmer_freq"]

# Same threshold/target as the notebook version — see its 2026-08-05 update for the incident
# that motivated this. AUC/AUPRC on tens of millions of negatives is statistically
# indistinguishable from the full set; every true positive is always kept.
NEGATIVE_SAMPLE_ROW_THRESHOLD = 100_000_000
NEGATIVE_SAMPLE_TARGET = 20_000_000

OUTPUT_SCHEMA = {
    "encoding": pl.Utf8, "dash_encoding": pl.Utf8, "ksize": pl.Int64, "metric": pl.Utf8,
    "roc_auc": pl.Float64, "pr_auc": pl.Float64, "n_pairs": pl.Int64,
    "n_pairs_full": pl.Int64, "subsampled": pl.Boolean,
}


def empty_output() -> pl.DataFrame:
    return pl.DataFrame(schema=OUTPUT_SCHEMA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dash-encoding", required=True)
    ap.add_argument("--display-encoding", required=True)
    ap.add_argument("--ksize", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    dash_enc, disp_enc, k = args.dash_encoding, args.display_encoding, args.ksize

    try:
        base_lazy = (
            ou.scan_available_columns(dash_enc, k, args.data_dir, COMPOSITE_COLUMNS)
            .with_columns([
                pl.col("query_name").str.split("|").list.get(6).str.to_uppercase().alias("human_gene"),
                pl.col("target_name").str.split("|").list.get(6).str.to_uppercase().alias("mouse_gene"),
            ])
        )
        # Row count only (parquet metadata / cheap streaming count) -- decides subsampling
        # without ever materializing the full table.
        n_rows_total = base_lazy.select(pl.len()).collect(engine="streaming").item()
    except FileNotFoundError:
        print(f"MISSING: {disp_enc} k={k}", flush=True)
        empty_output().write_csv(args.output)
        return
    except pl.exceptions.NoDataError:
        print(f"EMPTY: {disp_enc} k={k}", flush=True)
        empty_output().write_csv(args.output)
        return
    except pl.exceptions.ComputeError as e:
        print(f"CORRUPT FILE, skipping: {disp_enc} k={k}  ({type(e).__name__}: {e})", flush=True)
        empty_output().write_csv(args.output)
        return

    subsampled = n_rows_total > NEGATIVE_SAMPLE_ROW_THRESHOLD
    print(f"starting: {disp_enc} k={k} ({n_rows_total:,} rows)"
          + (f" -- subsampling negatives to ~{NEGATIVE_SAMPLE_TARGET:,}" if subsampled else "")
          + "...", flush=True)

    mgi_pairs_ou, _ = ou.load_mgi_orthologs()
    mgi_pairs_labels = (mgi_pairs_ou.select(["human_upper", "mouse_upper"])
                         .rename({"human_upper": "human_gene", "mouse_upper": "mouse_gene"})
                         .unique().with_columns(pl.lit(1).alias("label")))

    # n_human/n_mouse (the Bonferroni/BH denominator) must reflect the FULL file's protein-ID
    # cardinality, not a subsample -- cheap streaming distinct count, independent of the giant
    # per-row table below.
    n_human, n_mouse = (
        base_lazy.select([pl.col("query_name").n_unique(), pl.col("target_name").n_unique()])
        .collect(engine="streaming").row(0)
    )
    n_total = n_human * n_mouse

    lazy = (
        base_lazy.join(mgi_pairs_labels.lazy(), on=["human_gene", "mouse_gene"], how="left")
        .with_columns(pl.col("label").fill_null(0))
    )
    if subsampled:
        keep_frac = min(1.0, NEGATIVE_SAMPLE_TARGET / n_rows_total)
        cutoff = int(keep_frac * 1_000_000)
        lazy = lazy.filter(
            (pl.col("label") == 1)
            | (pl.concat_str(["query_name", "target_name"], separator="|").hash(seed=0) % 1_000_000 < cutoff)
        )
    # query_name/target_name (full protein IDs) are the biggest single memory cost of this scan
    # and aren't used past this point -- human_gene/mouse_gene already carry what downstream code
    # needs.
    df = lazy.drop(["query_name", "target_name"]).collect(engine="streaming")

    # A handful of hp-lehninger re-runs (k24, k26-30) shipped `prob_overlap` instead of
    # `poisson_pvalue` (older/newer kmerseek version) -- ensure_poisson_pvalue recomputes it
    # from n_intersecting_hashes/expected_shared_kmers when the file doesn't have it, same as
    # ou.load_kmerseek_data does for pre-poisson_pvalue TSVs.
    df = ou.ensure_poisson_pvalue(df)

    # Same conservative-correction logic as ou.load_kmerseek_data, generalized to any combo:
    # N_TOTAL is this file's own unique-query x unique-target count, not a fixed constant.
    poisson_p = df["poisson_pvalue"].fill_null(1.0).to_numpy()
    df = df.with_columns([
        pl.Series("poisson_p_bonf_conservative", np.clip(poisson_p * n_total, 0, 1)),
        pl.Series("poisson_p_bh_conservative", ou.bh_conservative(poisson_p, n_total)),
    ])

    df = ou.add_composite_scores(df)
    aucs = ou.compute_aucs(df)

    rows = [
        dict(encoding=disp_enc, dash_encoding=dash_enc, ksize=k, metric=metric,
             roc_auc=vals["roc_auc"], pr_auc=vals["pr_auc"], n_pairs=df.height,
             n_pairs_full=n_rows_total, subsampled=subsampled)
        for metric, vals in aucs.items()
    ]
    out = pl.DataFrame(rows, schema=OUTPUT_SCHEMA) if rows else empty_output()
    out.write_csv(args.output)
    print(f"{disp_enc} k={k}: {df.height:,}" + (f" of {n_rows_total:,}" if subsampled else "")
          + f" pairs, {len(rows)} metric rows written", flush=True)


if __name__ == "__main__":
    main()
