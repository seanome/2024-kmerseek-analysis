#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""Standalone runner for notebook 200's section 2b metric-leaderboard sweep.

Exists because running this inside the Jupyter kernel let the giant k=18 HP-Lehninger combos
(364M/429M rows) silently thrash the machine for 5+ days before the kernel died with nothing
saved (checkpoint only flushes *between* combos). Run this headless with nohup instead so
progress survives an editor/kernel restart and is watchable via `tail -f`:

    nohup /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \\
        scripts/run_200_section2b_leaderboard.py > /tmp/200_section2b.log 2>&1 &
    tail -f /tmp/200_section2b.log

Writes/appends to the same 200_metric_leaderboard_all_combos.csv the notebook cell reads, so
re-running the notebook cell afterward just sees everything as already cached.
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import ortholog_analysis_utils as ou

DATA_DIR = Path("/Users/olga/data/gencode/results-human-mouse-orthologs")

COMPOSITE_COLUMNS = ["query_name", "target_name", "jaccard", "containment",
                     "n_intersecting_hashes", "expected_shared_kmers",
                     "poisson_pvalue", "enrichment", "query_tfidf", "mean_matched_kmer_freq"]

NEGATIVE_SAMPLE_ROW_THRESHOLD = 100_000_000
NEGATIVE_SAMPLE_TARGET = 20_000_000

METRIC_LEADERBOARD_CSV = DATA_DIR / "200_metric_leaderboard_all_combos.csv"


def _file_size(enc: str, k: int) -> int:
    try:
        return ou.genome_wide_results_file(enc, k, DATA_DIR).stat().st_size
    except FileNotFoundError:
        return 0


def main() -> None:
    mgi_pairs_ou, _ = ou.load_mgi_orthologs()
    mgi_pairs_labels = (mgi_pairs_ou.select(["human_upper", "mouse_upper"])
                         .rename({"human_upper": "human_gene", "mouse_upper": "mouse_gene"})
                         .unique().with_columns(pl.lit(1).alias("label")))

    combos_2b_df = ou.load_all_alphabet_ksize_combos()
    combos_2b = list(zip(
        combos_2b_df["dash_encoding"].to_list(),
        combos_2b_df["display_encoding"].to_list(),
        combos_2b_df["ksize"].to_list(),
    ))
    combos_2b.sort(key=lambda c: _file_size(c[0], c[2]))

    leaderboard_cached = pl.read_csv(METRIC_LEADERBOARD_CSV) if METRIC_LEADERBOARD_CSV.exists() else pl.DataFrame()
    done = (set(zip(leaderboard_cached["dash_encoding"].to_list(), leaderboard_cached["ksize"].to_list()))
            if leaderboard_cached.height else set())
    rows = leaderboard_cached.to_dicts()
    print(f"{len(done)}/{len(combos_2b)} combos already cached", flush=True)

    for dash_enc, disp_enc, k in combos_2b:
        if (dash_enc, k) in done:
            continue
        size_gb = _file_size(dash_enc, k) / 1e9
        t0 = time.time()
        try:
            base_lazy = (
                ou.scan_available_columns(dash_enc, k, DATA_DIR, COMPOSITE_COLUMNS)
                .with_columns([
                    pl.col("query_name").str.split("|").list.get(6).str.to_uppercase().alias("human_gene"),
                    pl.col("target_name").str.split("|").list.get(6).str.to_uppercase().alias("mouse_gene"),
                ])
            )
            n_rows_total = base_lazy.select(pl.len()).collect(engine="streaming").item()
        except FileNotFoundError:
            print(f"  MISSING: {disp_enc} k={k}", flush=True)
            continue
        except pl.exceptions.NoDataError:
            print(f"  EMPTY: {disp_enc} k={k}", flush=True)
            continue
        except pl.exceptions.ComputeError as e:
            print(f"  CORRUPT FILE, skipping: {disp_enc} k={k}  ({type(e).__name__}: {e})", flush=True)
            continue

        subsampled = n_rows_total > NEGATIVE_SAMPLE_ROW_THRESHOLD
        print(f"  starting: {disp_enc} k={k} ({size_gb:.2f} GB compressed, {n_rows_total:,} rows)"
              + (f" -- subsampling negatives to ~{NEGATIVE_SAMPLE_TARGET:,}" if subsampled else "")
              + "...", flush=True)

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
        df = lazy.drop(["query_name", "target_name"]).collect(engine="streaming")

        df = ou.ensure_poisson_pvalue(df)

        poisson_p = df["poisson_pvalue"].fill_null(1.0).to_numpy()
        df = df.with_columns([
            pl.Series("poisson_p_bonf_conservative", np.clip(poisson_p * n_total, 0, 1)),
            pl.Series("poisson_p_bh_conservative", ou.bh_conservative(poisson_p, n_total)),
        ])

        df = ou.add_composite_scores(df)
        aucs = ou.compute_aucs(df)
        elapsed = time.time() - t0
        for metric, vals in aucs.items():
            rows.append(dict(encoding=disp_enc, dash_encoding=dash_enc, ksize=k, metric=metric,
                              roc_auc=vals["roc_auc"], pr_auc=vals["pr_auc"], n_pairs=df.height,
                              n_pairs_full=n_rows_total, subsampled=subsampled))
        pl.DataFrame(rows).write_csv(METRIC_LEADERBOARD_CSV)
        print(f"  {disp_enc} k={k}: {df.height:,}" + (f" of {n_rows_total:,}" if subsampled else "")
              + f" pairs  ({elapsed:.1f}s)", flush=True)

    print("\nAll combos done.", flush=True)


if __name__ == "__main__":
    main()
