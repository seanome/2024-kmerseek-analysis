#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""Score one (encoding, ksize) combo's per-HGNC-family discriminative AUC -- one Nextflow task
per combo, generalizing notebook 206 section 4's hand-picked 5-family list (Olfactory receptors,
CYP2/CYP3, antiviral restriction factors, sperm/testis) to every HGNC `gene_group` with >=15
protein-coding genes (~279 families, `ou.load_hgnc_family_gene_sets`'s default).

Same "sweep moved out of the notebook" pattern as `compute_rbh_f1_combo.py` /
`compute_metric_leaderboard_combo.py`: one task per combo instead of a sequential in-notebook
loop, storeDir gives free per-combo resumability. Takes one (encoding, ksize) combo per
invocation -- the caller decides scope. Notebook 206 section 4's plots use a fixed 9/10-combo
subset (protein/dayhoff at their established best-k, all 6 HP variants at k=30, plus
hp-pbotc-1st-ed's own k=19 point); section 9's all-HGNC-family generalization instead runs this
over the full alphabet x ksize sweep (same combo_tuples as computeRbhF1/computeMetricLeaderboard).
Family COUNT (5 -> ~279) and combo COUNT (9/10 -> ~100) are independent scaling dimensions --
`ou.load_families_kmerseek_scores` scans each raw file once regardless of family count, so
neither affects the other's cost.

Usage:
    compute_family_auc_combo.py \\
        --dash-encoding hp-pbotc-1st-ed --display-encoding hp_pbotc_1st_ed --ksize 19 \\
        --data-dir /Users/olga/data/gencode/results-human-mouse-orthologs \\
        --output 206_family_auc.hp-pbotc-1st-ed.k19.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/Users/olga/code/2024-kmerseek-analysis/notebooks")
import polars as pl  # noqa: E402

import ortholog_analysis_utils as ou  # noqa: E402

N_FLOOR = 15  # scored pairs required before a family's AUC is reported -- notebook 206's own floor
N_BOOT = 1000  # bootstrap_auc_ci resamples -- same as notebook 206 section 4

OUTPUT_SCHEMA = {
    "family": pl.Utf8, "n_family_genes": pl.Int64,
    "encoding": pl.Utf8, "dash_encoding": pl.Utf8, "ksize": pl.Int64, "n": pl.Int64,
    "auc_score_bonf": pl.Float64, "auc_score_bonf_lo": pl.Float64, "auc_score_bonf_hi": pl.Float64,
    "auc_jaccard": pl.Float64, "auc_jaccard_lo": pl.Float64, "auc_jaccard_hi": pl.Float64,
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

    family_gene_sets = ou.load_hgnc_family_gene_sets()  # default: min 15 genes, zinc fingers excluded
    _, mgi_set = ou.load_mgi_orthologs()

    try:
        family_dfs = ou.load_families_kmerseek_scores(dash_enc, k, family_gene_sets, mgi_ortholog_set=mgi_set,
                                                       data_dir=args.data_dir)
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

    rows = []
    n_skipped = 0
    for fam_name, df in family_dfs.items():
        if df.height == 0 or "label" not in df.columns:
            n_skipped += 1
            continue
        y = df["label"].to_numpy()
        if len(y) < N_FLOOR or y.sum() == 0 or y.sum() == len(y):
            n_skipped += 1
            continue
        scores_bonf = df["score_bonf_neglogp_cont"].fill_null(0.0).fill_nan(0.0).to_numpy()
        scores_jaccard = df["jaccard"].fill_null(0.0).fill_nan(0.0).to_numpy()
        res_bonf = ou.bootstrap_auc_ci(y, scores_bonf, n_boot=N_BOOT)
        res_jacc = ou.bootstrap_auc_ci(y, scores_jaccard, n_boot=N_BOOT)
        rows.append(dict(
            family=fam_name, n_family_genes=len(family_gene_sets[fam_name]),
            encoding=disp_enc, dash_encoding=dash_enc, ksize=k, n=len(y),
            auc_score_bonf=res_bonf["point"], auc_score_bonf_lo=res_bonf["lo"], auc_score_bonf_hi=res_bonf["hi"],
            auc_jaccard=res_jacc["point"], auc_jaccard_lo=res_jacc["lo"], auc_jaccard_hi=res_jacc["hi"],
        ))

    out = pl.DataFrame(rows, schema=OUTPUT_SCHEMA) if rows else empty_output()
    out.write_csv(args.output)
    print(f"{disp_enc} k={k}: {len(rows)}/{len(family_gene_sets)} families scored "
          f"({n_skipped} below floor or single-class)", flush=True)


if __name__ == "__main__":
    main()
