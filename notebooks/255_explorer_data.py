#!/usr/bin/env python3
"""Data for the notebook 255 explorer page: the per alphabet-ksize pair table of
255_ranking_metrics_per_arm.py, plus precision-recall and ROC curves sampled at fixed
grids, so the page never ships raw matches.

Each alphabet-ksize pair's regions are labelled and reduced to one row per pair of spans
exactly as 255_ranking_metrics_per_arm.py does. For every overlap rule and metric, the
curves are drawn over the matches that metric can score, for the metric and for region
length on those same matches. AP and AUC are recomputed from the full (unsampled) curves
and compared with per_arm_metrics.parquet; the largest difference is printed.

Curves are stored as base64 little-endian uint16 (value * 65535), 100 points each.

Usage: 255_explorer_data.py [--workers 4] [--out PATH]
Per-pair results are cached in <out dir>/explorer_parts/ so a rerun resumes.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_metrics_utils as rm  # noqa: E402

COLS = ["region_evalue", "region_ka_bits", "region_mean_idf", "region_tfidf", "region_enrichment",
        "region_tail_probability", "region_n_shared_kmers"]
READ = ["query_name", "target_name", "region_start", "region_end", "target_start", "target_end",
        "region_ka_lambda", "region_length"] + COLS
RECALL_GRID = np.round(np.arange(1, 101) / 100, 2)  # 0.01 .. 1.00
# ROC: dense at small false-positive rates, where a low base rate puts the useful cutoffs
FPR_GRID = np.unique(np.round(np.r_[np.logspace(-5, -1, 40), np.linspace(0, 1, 61)], 6))
OUT = rm.PER_ARM.parent / "explorer_data.json"


def enc(v: np.ndarray) -> str:
    q = np.round(np.clip(v, 0, 1) * 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode()


def curves(s: np.ndarray, y: np.ndarray):
    """Sampled PR and ROC curves plus AP and AUC from the full curves (ties as one step)."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    o = np.argsort(-s, kind="stable")
    ss, yy = s[o], y[o]
    last = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(yy)[last].astype(float); fp = np.cumsum(~yy)[last].astype(float)
    rec, prec = tp / n1, tp / (tp + fp)
    fpr, tpr = np.r_[0.0, fp / n0], np.r_[0.0, tp / n1]
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    auc = float(np.trapezoid(tpr, fpr))
    pr = prec[np.minimum(np.searchsorted(rec, RECALL_GRID - 1e-12, side="left"), len(rec) - 1)]
    roc = np.interp(FPR_GRID, fpr, tpr)
    return dict(pr=enc(pr), roc=enc(roc)), ap, auc


def one_pair(path: Path) -> dict:
    alphabet, k = path.stem.rsplit(".k", 1)
    truth = pl.read_parquet(rm.TRUTH)
    df = rm.independent_matches(rm.label_regions(pl.read_parquet(path, columns=READ), truth))
    L = rm.score(df, "region_length")
    scores = {c: rm.score(df, c) for c in COLS}
    out = {}
    for rule, rname in rm.RULES:
        y = df[rule].to_numpy()
        for c in COLS:
            ok = np.isfinite(scores[c])
            m = curves(scores[c][ok], y[ok]) if ok.any() else None
            l = curves(L[ok], y[ok]) if ok.any() else None
            if m is None or l is None:
                continue
            out[f"{rname}|{rm.NAME[c]}"] = dict(pr=m[0]["pr"], roc=m[0]["roc"], lpr=l[0]["pr"], lroc=l[0]["roc"],
                                               ap=m[1], auc=m[2], lap=l[1], lauc=l[2])
    return dict(alphabet=alphabet, ksize=int(k), n=df.height, curves=out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    parts = args.out.parent / "explorer_parts"
    parts.mkdir(exist_ok=True)
    paths = sorted((rm.PF998 / "regions").glob("*.parquet"), key=lambda p: p.stat().st_size)
    todo = [p for p in paths if not (parts / f"{p.stem}.json").exists()]
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one_pair, p): p for p in todo}
        for f in as_completed(futs):
            r = f.result()
            (parts / f"{futs[f].stem}.json").write_text(json.dumps(r))
            print(f"{futs[f].stem}: {r['n']:_} matches", file=sys.stderr, flush=True)

    arm = pl.read_parquet(rm.PER_ARM)
    got = [json.loads((parts / f"{p.stem}.json").read_text()) for p in paths]
    # recomputed AP / AUC against the stored table
    ref = {(r["alphabet"], r["ksize"], r["rule"], r["metric"]): r for r in arm.iter_rows(named=True)}
    worst = 0.0
    for g in got:
        for key, c in g["curves"].items():
            rule, metric = key.split("|")
            r = ref[(g["alphabet"], g["ksize"], rule, metric)]
            worst = max(worst, abs(c["ap"] - r["ap"]), abs(c["auc"] - r["auc"]),
                        abs(c["lap"] - r["length_ap"]), abs(c["lauc"] - r["length_auc"]))
    print(f"largest |recomputed - stored| over AP, AUC and length's: {worst:.2e}", file=sys.stderr)

    keep = ["alphabet", "ksize", "bits", "metric", "rule", "n_matches", "n_scored", "n_correct", "n_correct_all",
            "base_rate", "auc", "ap", "length_auc", "length_ap", "precision_at_1", "precision_at_1_random",
            "n_queries_with_correct"]
    rows = arm.select(keep).sort("alphabet", "ksize", "rule", "metric")
    curves_out = {f"{g['alphabet']}|{g['ksize']}|{key}": {k: v for k, v in c.items() if k in ("pr", "roc", "lpr", "lroc")}
                  for g in got for key, c in g["curves"].items()}
    data = dict(columns=keep, rows=[[None if isinstance(v, float) and not np.isfinite(v) else
                                     (round(v, 5) if isinstance(v, float) else v) for v in r] for r in rows.iter_rows()],
                rules=[n for _, n in rm.RULES], recall_grid=RECALL_GRID.tolist(), fpr_grid=FPR_GRID.tolist(),
                max_abs_diff=worst, curves=curves_out)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
