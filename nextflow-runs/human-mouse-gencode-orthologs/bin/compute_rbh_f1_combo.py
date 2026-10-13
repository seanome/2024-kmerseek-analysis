#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""Score one (encoding, ksize) combo's RBH-F1 vs MGI, OrthoFinder (matched scope), continuous
ranking AUC, and BH-per-pair-F1 -- one Nextflow task per combo.

This is notebook 200 section 1's `rbh_vs_matched_orthofinder`/`bh_per_pair_f1` sweep, pulled out
of the notebook's own sequential for-loop -- same reasoning as `compute_metric_leaderboard_combo.py`
(section 2b's equivalent extraction): real parallelism across combos instead of a notebook loop
that has to be babysat, and storeDir gives free per-combo resumability. Also generalizes the
sweep from protein/dayhoff/hp only to every combo with real genome-wide results (all 6 dash-named
HP variants across their full k range) -- notebook 206 section 1 hand-rolled a second, partial
version of this exact sweep for just the 6 new variants (only 2/6 ever finished), which this
replaces too.

Usage:
    compute_rbh_f1_combo.py \\
        --dash-encoding hp-thomas-dill --display-encoding hp_thomas_dill --ksize 26 \\
        --data-dir /Users/olga/data/gencode/results-human-mouse-orthologs \\
        --of-tsv /Users/olga/data/gencode/data-for-orthofinder/.../gencode.v49...tsv \\
        --output 200_rbh_f1.hp-thomas-dill.k26.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/Users/olga/code/2024-kmerseek-analysis/notebooks")
import ortholog_analysis_utils as ou  # noqa: E402

RBH_SCHEMA = {
    "encoding": pl.Utf8, "dash_encoding": pl.Utf8, "ksize": pl.Int64, "n_genes_scope": pl.Int64,
    "kmerseek_precision": pl.Float64, "kmerseek_recall": pl.Float64, "kmerseek_F1": pl.Float64,
    "orthofinder_precision": pl.Float64, "orthofinder_recall": pl.Float64,
    "orthofinder_F1_matched": pl.Float64, "gap_to_orthofinder": pl.Float64,
    "roc_auc": pl.Float64, "pr_auc": pl.Float64,
    "bh_per_pair_precision": pl.Float64, "bh_per_pair_recall": pl.Float64, "bh_per_pair_F1": pl.Float64,
}


def empty_output() -> pl.DataFrame:
    return pl.DataFrame(schema=RBH_SCHEMA)


def gene_from_protein_id(pid: str) -> str:
    parts = pid.split("|")
    return parts[-2] if len(parts) >= 2 else pid


def load_orthofinder_pairs(of_tsv: Path) -> set[tuple[str, str]]:
    """Full (not scope-restricted) OrthoFinder human/mouse gene pair set -- re-scoped per combo
    by the caller, same as notebook 200's `of_set_all` (lesson 3: OrthoFinder must be scored on
    the SAME gene universe as the kmerseek combo it's compared to, not its own full workload)."""
    of_raw = pl.read_csv(str(of_tsv), separator="\t", null_values=["", "nan"])
    h_col, m_col = "gencode.v49.pc_translations", "gencode.vM38.pc_translations"
    pairs = set()
    for row in of_raw.iter_rows(named=True):
        for h_pid in (row[h_col] or "").split(","):
            h_pid = h_pid.strip()
            if not h_pid:
                continue
            h_gene = gene_from_protein_id(h_pid).upper()
            for m_pid in (row[m_col] or "").split(","):
                m_pid = m_pid.strip()
                if m_pid:
                    pairs.add((h_gene, gene_from_protein_id(m_pid).upper()))
    return pairs


def prf1(called: set, truth: set) -> tuple[float, float, float]:
    tp, fp, fn = len(called & truth), len(called - truth), len(truth - called)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dash-encoding", required=True)
    ap.add_argument("--display-encoding", required=True)
    ap.add_argument("--ksize", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--of-tsv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    dash_enc, disp_enc, k = args.dash_encoding, args.display_encoding, args.ksize

    try:
        pair = ou.load_pair_table(dash_enc, k, args.data_dir)
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

    mgi_pairs_ou, _ = ou.load_mgi_orthologs()
    mgi_pairs_df = (mgi_pairs_ou.select(["human_upper", "mouse_upper"])
                     .rename({"human_upper": "human_gene", "mouse_upper": "mouse_gene"}).unique())
    of_set_all = load_orthofinder_pairs(args.of_tsv)

    human_genes_in_scope = set(pair["human_gene"].unique().to_list())

    # Reciprocal-best-hit: human gene's top mouse hit AND that mouse gene's top human hit agree.
    best_h2m = (pair.sort("jaccard", descending=True)
                .group_by("human_gene").agg(pl.col("mouse_gene").first().alias("best_mouse")))
    best_m2h = (pair.sort("jaccard", descending=True)
                .group_by("mouse_gene").agg(pl.col("human_gene").first().alias("best_human")))
    rbh = (best_h2m.join(best_m2h, left_on=["human_gene", "best_mouse"], right_on=["best_human", "mouse_gene"])
           .select(["human_gene", "best_mouse"]).rename({"best_mouse": "mouse_gene"}))
    rbh_set = set(zip(rbh["human_gene"].to_list(), rbh["mouse_gene"].to_list()))

    mgi_scope = mgi_pairs_df.filter(pl.col("human_gene").is_in(list(human_genes_in_scope)))
    mgi_scope_set = set(zip(mgi_scope["human_gene"].to_list(), mgi_scope["mouse_gene"].to_list()))

    ks_p, ks_r, ks_f1 = prf1(rbh_set, mgi_scope_set)

    # Lesson 3: OrthoFinder re-scored on the SAME gene universe this combo covers, not its own
    # full workload -- otherwise the comparison isn't valid.
    of_set_scoped = {(hg, mg) for (hg, mg) in of_set_all if hg in human_genes_in_scope}
    of_p, of_r, of_f1 = prf1(of_set_scoped, mgi_scope_set)

    # Continuous ranking quality: label EVERY candidate pair, not just the RBH-selected best hit.
    mgi_scope_labels = mgi_scope.select(["human_gene", "mouse_gene"]).unique().with_columns(pl.lit(1).alias("label"))
    pair_labeled = pair.join(mgi_scope_labels, on=["human_gene", "mouse_gene"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )
    y = pair_labeled["label"].to_numpy()
    if y.sum() > 0 and (1 - y).sum() > 0:
        scores = pair_labeled["jaccard"].to_numpy()
        roc_auc = roc_auc_score(y, scores)
        pr_auc = average_precision_score(y, scores)
    else:
        roc_auc = pr_auc = float("nan")

    row = dict(
        encoding=disp_enc, dash_encoding=dash_enc, ksize=k, n_genes_scope=len(human_genes_in_scope),
        kmerseek_precision=ks_p, kmerseek_recall=ks_r, kmerseek_F1=ks_f1,
        orthofinder_precision=of_p, orthofinder_recall=of_r, orthofinder_F1_matched=of_f1,
        gap_to_orthofinder=of_f1 - ks_f1, roc_auc=roc_auc, pr_auc=pr_auc,
    )

    # BH-per-pair F1 ("lesson 2", shown to underperform RBH): needs n_intersecting_hashes /
    # expected_shared_kmers, not just jaccard, so this is a second raw scan rather than reusing
    # load_pair_table's cache. Best-effort -- a failure here doesn't invalidate the RBH result
    # above, which is the methodology actually used everywhere downstream.
    try:
        raw = (
            ou.scan_genome_wide_results(
                dash_enc, k, args.data_dir,
                columns=["query_name", "target_name", "n_intersecting_hashes", "expected_shared_kmers"],
            )
            .with_columns([
                pl.col("query_name").str.split("|").list.get(6).str.to_uppercase().alias("human_gene"),
                pl.col("target_name").str.split("|").list.get(6).str.to_uppercase().alias("mouse_gene"),
            ])
            .collect(engine="streaming")
        )
        k_arr = raw["n_intersecting_hashes"].fill_null(0).to_numpy()
        lam_arr = raw["expected_shared_kmers"].fill_null(0).to_numpy()
        p_raw = poisson.sf(k_arr - 1, np.maximum(lam_arr, 1e-300)).clip(1e-300, 1.0)
        raw = raw.with_columns(pl.Series("p_raw", p_raw))

        bh_pair = raw.group_by(["human_gene", "mouse_gene"]).agg(pl.col("p_raw").min().alias("p_min"))
        n = bh_pair.height
        p_min = bh_pair["p_min"].to_numpy()
        order = np.argsort(p_min)
        p_bh = np.empty(n)
        p_bh[order] = (p_min[order] * n / (np.arange(n) + 1)).clip(max=1)
        p_bh[order] = np.minimum.accumulate(p_bh[order[::-1]])[::-1]
        bh_pair = bh_pair.with_columns(pl.Series("bh", p_bh))

        called = bh_pair.filter(pl.col("bh") <= 0.05)
        called_set = set(zip(called["human_gene"].to_list(), called["mouse_gene"].to_list()))
        bh_p, bh_r, bh_f1 = prf1(called_set, mgi_scope_set)
        row.update(bh_per_pair_precision=bh_p, bh_per_pair_recall=bh_r, bh_per_pair_F1=bh_f1)
    except (FileNotFoundError, pl.exceptions.NoDataError, pl.exceptions.ComputeError) as e:
        print(f"  (bh_per_pair skipped: {type(e).__name__})", flush=True)
        row.update(bh_per_pair_precision=None, bh_per_pair_recall=None, bh_per_pair_F1=None)

    pl.DataFrame([row], schema=RBH_SCHEMA).write_csv(args.output)
    print(f"{disp_enc} k={k}: kmerseek_F1={ks_f1:.4f} orthofinder_F1={of_f1:.4f} "
          f"n_scope={len(human_genes_in_scope):,}", flush=True)


if __name__ == "__main__":
    main()
