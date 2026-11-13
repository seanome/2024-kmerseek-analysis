#!/usr/bin/env python3
"""Per contrast: how many sequences each method keeps, and whether the k-mer tests lack power.

Usage: power_check.py <contrasts_dir> <out.tsv> <out_hist.tsv>

For each contrast it reads the p-values each test already wrote and reports:
- the number of tests, the smallest p-value and the smallest Benjamini-Hochberg corrected p;
- pi0, the fraction of tests with no change estimated from the p-values (Storey, lambda 0.5:
  2 x the fraction of p-values above 0.5). A value below 1 means there are more small
  p-values than chance gives, so some change is present even when nothing passes correction;
- the number of p-values at or below 0.05 and 0.001, next to the number chance gives (m x p);
- for SPLASH, the anchors kept at its corrected p <= 0.05 (rows of scores.tsv).

It also takes the control's DE-kupl p-values and corrects them as if there had been as many
tests as each real contrast had, to show how many of the control's kept k-mers would survive
that many tests.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

contrasts_dir, out_tsv, out_hist = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
BINS = np.linspace(0, 1, 21)  # p-value histogram, 20 bins of width 0.05
hist_rows = []
CONTRASTS = ["control", "young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]


def bh_min_and_kept(p, m=None):
    """Smallest BH-corrected p and the number with corrected p <= 0.05, for m tests in all."""
    p = np.sort(p[~np.isnan(p)])
    m = len(p) if m is None else m
    q = p * m / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    return float(min(q.min(), 1.0)), int((q <= 0.05).sum())


def describe(method, contrast, p, extra=None):
    p = p[~np.isnan(p)]
    m = len(p)
    qmin, kept = bh_min_and_kept(p)
    row = {
        "contrast": contrast, "method": method, "tests": m,
        "min_p": float(p.min()), "min_corrected_p": qmin, "kept_corrected_p_le_0.05": kept,
        "pi0": float(min(1.0, 2 * (p > 0.5).mean())),
        "p_le_0.05": int((p <= 0.05).sum()), "chance_p_le_0.05": round(0.05 * m),
        "p_le_0.001": int((p <= 0.001).sum()), "chance_p_le_0.001": round(0.001 * m),
    }
    row.update(extra or {})
    counts, _ = np.histogram(p, bins=BINS)
    hist_rows.extend({"contrast": contrast, "method": method, "bin_low": float(lo), "n": int(n)}
                     for lo, n in zip(BINS[:-1], counts))
    print(row, flush=True)
    return row


rows, control_p = [], None
for c in CONTRASTS:
    d = contrasts_dir / c
    raw = d / "dekupl_limma/DEkupl_result/control_vs_PEC_kmer_counts/raw_pvals.txt.gz"
    p = pl.read_csv(raw, separator="\t", has_header=False, columns=[1]).to_series(0).cast(pl.Float64).to_numpy()
    if c == "control":
        control_p = p
    rows.append(describe("DE-kupl limma, k-mers", c, p))
    for design in ("pair_condition", "condition"):
        f = d / f"deseq/deseq_{design}.parquet"
        if f.exists():
            p = pl.read_parquet(f, columns=["pvalue"])["pvalue"].to_numpy()
            rows.append(describe(f"anchor counts DESeq2 ~ {design.replace('_', ' + ')}", c, p))
    scores = d / "splash_out/result.after_correction.scores.tsv"
    summary = d / "deseq/deseq_summary.json"
    tested = json.loads(summary.read_text())["anchors_tested_by_splash"] if summary.exists() else None
    kept = sum(1 for _ in open(scores)) - 1
    rows.append({"contrast": c, "method": "SPLASH, anchors", "tests": tested,
                 "kept_corrected_p_le_0.05": kept})
    print(rows[-1], flush=True)

# The control's planted changes, corrected for each real contrast's number of k-mer tests.
for r in rows:
    if r["method"].startswith("DE-kupl") and r["contrast"] != "control":
        _, kept = bh_min_and_kept(control_p, m=r["tests"])
        r["control_kept_if_this_many_tests"] = kept

pl.DataFrame(rows).write_csv(out_tsv, separator="\t")
pl.DataFrame(hist_rows).write_csv(out_hist, separator="\t")
print("wrote", out_tsv)
