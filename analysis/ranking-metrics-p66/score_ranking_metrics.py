#!/usr/bin/env python3
"""Score kmerseek's ranking values against the labeled benchmark.

The point of the script is the control: region length on its own is a strong
ranker, so any value has to beat length before it counts for anything. Two
comparisons are made for each value:

  * against every match at once, where length is the control;
  * within one region length at a time, which is where a value either shows it
    carries its own signal or shows it was length in disguise.

Reads labeled_pairs.parquet from build_labeled_benchmark.py.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ["region_evalue", "region_tfidf", "region_mean_idf", "region_enrichment",
           "region_ka_bits", "region_length"]
LENGTH_BINS = [(24, 30), (30, 40), (40, 60), (60, 120), (120, 10_000)]


def as_score(d: pl.DataFrame, col: str) -> np.ndarray:
    """Turn a column into a score where bigger is better.

    A region past the composition boundary has no E-value (infinity); it ranks
    last rather than being dropped, so every match stays in the comparison.
    """
    v = d[col].to_numpy().astype(float)
    if col == "region_evalue":
        s = -np.log10(np.clip(v, 1e-300, None))
        return np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1)
    return np.nan_to_num(v, nan=0.0)


def bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int, rng) -> tuple:
    """AUC with a percentile interval. Returns NaNs when a class is too small."""
    if y.sum() < 2 or (~y.astype(bool)).sum() < 2:
        return (np.nan, np.nan, np.nan)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 2 <= y[i].sum() < len(i):
            draws.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return roc_auc_score(y, s), lo, hi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="labeled_pairs.parquet", type=Path)
    p.add_argument("--rule", default="correct_frac20_of_region",
                   help="which correctness column to score against")
    p.add_argument("--n-boot", type=int, default=1500)
    args = p.parse_args()

    d = pl.read_parquet(args.pairs)
    y = d[args.rule].to_numpy()
    L = d["region_length"].to_numpy().astype(float)
    rng = np.random.default_rng(0)
    print(f"{len(d):_} independent matches, {int(y.sum()):_} correct "
          f"({100 * y.mean():.1f}%), rule {args.rule}\n")

    print("Ranking every match together")
    print(f"{'metric':<24}{'ROC AUC':>10}{'avg precision':>16}")
    for c in METRICS:
        s = as_score(d, c)
        tag = "  (control)" if c == "region_length" else ""
        print(f"{c:<24}{roc_auc_score(y, s):>10.4f}{average_precision_score(y, s):>16.4f}{tag}")

    print("\nWithin one region length at a time")
    hdr = f"{'length':<10}{'n':>7}{'correct':>9}"
    for c in METRICS[:4]:
        hdr += f"{c.replace('region_', ''):>13}"
    print(hdr)
    for lo, hi in LENGTH_BINS:
        m = (L >= lo) & (L < hi)
        if m.sum() < 40 or not (2 <= y[m].sum() < m.sum()):
            print(f"{f'{lo}-{hi - 1}':<10}{int(m.sum()):>7_}{int(y[m].sum()):>9}   too few to score")
            continue
        line = f"{f'{lo}-{hi - 1}':<10}{int(m.sum()):>7_}{int(y[m].sum()):>9}"
        for c in METRICS[:4]:
            line += f"{roc_auc_score(y[m], as_score(d, c)[m]):>13.3f}"
        print(line)

    print("\nShort matches only, with a 95% bootstrap interval, because this is where the"
          "\nbenchmark runs out of correct matches and an AUC stops meaning much")
    print(f"{'cut':>5}{'pairs':>8}{'correct':>9}   {'mean IDF':>24}   {'E-value':>24}")
    mi, ev = as_score(d, "region_mean_idf"), as_score(d, "region_evalue")
    for cut in (28, 30, 32, 35, 40, 45, 50):
        m = L < cut
        a1, l1, h1 = bootstrap_auc(y[m], mi[m], args.n_boot, rng)
        a2, l2, h2 = bootstrap_auc(y[m], ev[m], args.n_boot, rng)
        fmt = lambda a, l, h: f"{a:.3f} [{l:.3f}, {h:.3f}]" if np.isfinite(a) else "too few"
        print(f"{cut:>5}{int(m.sum()):>8_}{int(y[m].sum()):>9}   "
              f"{fmt(a1, l1, h1):>24}   {fmt(a2, l2, h2):>24}")


if __name__ == "__main__":
    main()
