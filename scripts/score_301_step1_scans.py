#!/usr/bin/env python3
"""Notebook 301, step 1: score two scans that need no search on notebook 244's 87 cases.

The 87 cases are the rows of notebook 244's candidate table (PR 78, commit 9e6434f5)
admitted under "all criteria". A case is one human Swiss-Prot feature in one target
species. Both scans read only the human protein, so they give the same answer for every
species of a feature: the 87 cases are 50 distinct human features on 42 proteins.

Scans, both over a 19-residue window (notebook 231's window,
`swissprot_control_utils.kd_transmem_calls`):

* hydropathy: mean Kyte-Doolittle hydropathy of the window.
* period: the protein as a two-letter string, hydrophobic +1 (hp_pbotc_1st_ed2 class 0,
  ACFILMPVWY) and polar -1, as in notebook 295's period scan
  (nextflow-runs/convergence-afp/bin/scans.py, `hp_autocorr`). The window's score is its
  largest autocorrelation over lags 1 to 9: the mean of x[i] * x[i + lag] over the pairs
  inside the window. Lags stop at 9, about half the window, so every lag has at least 10
  pairs. Notebook 295 used lags 1 to 30 on whole proteins; a 19-residue window has no lag
  30.

A scan's call at threshold t: the windows scoring above t, merged where they touch or
overlap, each merged run reported as the residues its windows cover (1-based, inclusive).

Decoys (task step 3): for each feature, two copies of its parent protein with the feature's
residues replaced by
* a di-residue shuffle: the window cut into adjacent pairs (1-2, 3-4, ...), the pairs put in
  random order; an odd last residue stays last;
* the window reversed.
A decoy is hit when a call overlaps the replaced residues. The threshold for each scan is
the lowest one at which fewer than 1.0% of decoy windows are hit.

Recall: a feature is recalled when a call overlaps it by at least one residue. Boundary
error of a recalled feature: |call start - feature start| + |call end - feature end|, using
the overlapping call with the most residues in common with the feature (ties: the smaller
error).

Outputs in tables/: 301_step1_scan_curve_nb244_swissprot.csv (recall and decoy rate at every
threshold), 301_step1_scan_features_nb244_swissprot.csv (per feature at the chosen
threshold), 301_step1_scan_summary_nb244_swissprot.csv, and the decoy sequences in
data/301_short_feature_boundaries/step1_decoys_nb244_swissprot.fasta.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

WINDOW = 19
MAX_LAG = 9
DECOY_RATE_LIMIT = 0.01
#: Kyte and Doolittle 1982's cutoff for a membrane-spanning 19-residue window, the one
#: notebook 231 used. Reported for context only: the pass rule's threshold is set on decoys.
KD_PUBLISHED_CUTOFF = 1.6
SHUFFLE_SEED = 301
HYDROPHOBIC_HP = set("ACFILMPVWY")
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}  # fmt: skip


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    acc = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            acc = line[1:].split()[0]
            seqs[acc] = []
        elif acc is not None:
            seqs[acc].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def hydropathy_windows(seq: str) -> np.ndarray:
    """Mean Kyte-Doolittle hydropathy of each window; index i covers residues i..i+18."""
    h = np.array([KYTE_DOOLITTLE.get(c, 0.0) for c in seq])
    if h.size < WINDOW:
        return np.empty(0)
    return np.convolve(h, np.ones(WINDOW) / WINDOW, mode="valid")


def period_windows(seq: str) -> np.ndarray:
    """Largest H/P autocorrelation over lags 1..MAX_LAG inside each window."""
    x = np.array([1.0 if c in HYDROPHOBIC_HP else -1.0 for c in seq])
    n_win = x.size - WINDOW + 1
    if n_win <= 0:
        return np.empty(0)
    best = np.full(n_win, -np.inf)
    for lag in range(1, MAX_LAG + 1):
        prod = x[:-lag] * x[lag:]  # prod[j] pairs residue j with j + lag
        n_pairs = WINDOW - lag
        mean = np.convolve(prod, np.ones(n_pairs) / n_pairs, mode="valid")[:n_win]
        best = np.maximum(best, mean)
    return best


SCANS = {"hydropathy": hydropathy_windows, "period": period_windows}


def calls_at(scores: np.ndarray, t: float) -> list[tuple[int, int]]:
    """Merged calls (1-based inclusive) from windows scoring above t."""
    hot = scores > t
    if not hot.any():
        return []
    padded = np.concatenate(([False], hot, [False])).astype(np.int8)
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    # windows a..b-1 cover residues a .. b-1+WINDOW-1 (0-based); runs closer than one
    # window apart overlap and are merged
    out: list[list[int]] = []
    for a, b in zip(starts, ends):
        s, e = a + 1, b - 1 + WINDOW
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def best_window_over(scores: np.ndarray, start: int, end: int) -> float:
    """Highest window score among windows sharing a residue with start..end (1-based)."""
    lo = max(0, start - WINDOW)  # window i covers i+1..i+WINDOW (1-based)
    hi = min(scores.size, end)
    return float(scores[lo:hi].max()) if hi > lo else -np.inf


def di_residue_shuffle(window: str, rng: np.random.Generator) -> str:
    pairs = [window[i : i + 2] for i in range(0, len(window) - 1, 2)]
    tail = window[-1] if len(window) % 2 else ""
    order = rng.permutation(len(pairs))
    return "".join(pairs[i] for i in order) + tail


def best_call(calls, s, e):
    """Overlapping call with the most residues in common; ties to the smaller error."""
    best = None
    for cs, ce in calls:
        ov = min(ce, e) - max(cs, s) + 1
        if ov <= 0:
            continue
        err = abs(cs - s) + abs(ce - e)
        key = (-ov, err)
        if best is None or key < best[0]:
            best = (key, cs, ce, ov, err)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--decoy-fasta", type=Path, required=True)
    a = ap.parse_args()

    cases = pl.read_csv(a.cases, separator="\t")
    seqs = read_fasta(a.fasta)
    feats = (
        cases.group_by("query", "feature_type", "feature_start", "feature_end")
        .agg(
            pl.col("gene").first(),
            n_cases=pl.len(),
            species=pl.col("species").sort().str.join(","),
        )
        .sort("query", "feature_start")
        .with_row_index("feature_id")
    )
    rng = np.random.default_rng(SHUFFLE_SEED)
    decoys = []  # (feature_id, kind, sequence)
    with a.decoy_fasta.open("w") as fh:
        for r in feats.iter_rows(named=True):
            p = seqs[r["query"]]
            s, e = r["feature_start"], r["feature_end"]
            w = p[s - 1 : e]
            for kind, new in (
                ("dishuffle", di_residue_shuffle(w, rng)),
                ("reversed", w[::-1]),
            ):
                assert sorted(new) == sorted(w) and len(new) == len(w)
                d = p[: s - 1] + new + p[e:]
                decoys.append((r["feature_id"], kind, d))
                fh.write(f">{r['query']}|{s}-{e}|{kind}\n{d}\n")

    curve_rows, feat_rows, summary_rows = [], [], []
    for scan, fn in SCANS.items():
        t0 = time.perf_counter()
        real_scores = {acc: fn(seqs[acc]) for acc in feats["query"].unique()}
        ms_per_protein = (time.perf_counter() - t0) * 1000 / len(real_scores)
        f_best = np.array(
            [
                best_window_over(
                    real_scores[r["query"]], r["feature_start"], r["feature_end"]
                )
                for r in feats.iter_rows(named=True)
            ]
        )
        d_best = np.array(
            [
                best_window_over(
                    fn(dseq), feats["feature_start"][fid], feats["feature_end"][fid]
                )
                for fid, _, dseq in decoys
            ]
        )
        weights = feats["n_cases"].to_numpy()
        # threshold grid: every distinct best score, plus one below all
        grid = np.unique(
            np.concatenate([f_best, d_best, [min(f_best.min(), d_best.min()) - 1]])
        )
        for t in grid:
            curve_rows.append(
                {
                    "scan": scan,
                    "threshold": float(t),
                    "decoy_hit_rate": float((d_best > t).mean()),
                    "recall_features": float((f_best > t).mean()),
                    "recall_cases": float(
                        ((f_best > t) * weights).sum() / weights.sum()
                    ),
                }
            )
        ok = grid[[(d_best > t).mean() < DECOY_RATE_LIMIT for t in grid]]
        rules = {"decoy hit rate below 1%": float(ok.min())}
        if scan == "hydropathy":
            rules["Kyte-Doolittle 1982 cutoff 1.6 (context, not the pass rule)"] = (
                KD_PUBLISHED_CUTOFF
            )
        for rule, t_star in rules.items():
            for r, fb in zip(feats.iter_rows(named=True), f_best):
                calls = calls_at(real_scores[r["query"]], t_star)
                b = best_call(calls, r["feature_start"], r["feature_end"])
                L = r["feature_end"] - r["feature_start"] + 1
                feat_rows.append(
                    {
                        "scan": scan,
                        "threshold_rule": rule,
                        **{
                            k: r[k]
                            for k in (
                                "feature_id",
                                "gene",
                                "query",
                                "feature_type",
                                "feature_start",
                                "feature_end",
                                "n_cases",
                                "species",
                            )
                        },
                        "feature_length_aa": L,
                        "best_window_score": fb,
                        "recalled": b is not None,
                        "call_start": None if b is None else b[1],
                        "call_end": None if b is None else b[2],
                        "overlap_aa": None if b is None else b[3],
                        "overlap_fraction_of_feature": None if b is None else b[3] / L,
                        "boundary_error_aa": None if b is None else b[4],
                    }
                )
            fr = pl.DataFrame(feat_rows).filter(
                (pl.col("scan") == scan) & (pl.col("threshold_rule") == rule)
            )
            rec = fr.filter("recalled")
            summary_rows.append(
                {
                    "scan": scan,
                    "threshold_rule": rule,
                    "threshold": t_star,
                    "n_decoy_windows": len(decoys),
                    "n_decoys_hit": int((d_best > t_star).sum()),
                    "decoy_hit_rate": float((d_best > t_star).mean()),
                    "n_features": feats.height,
                    "n_cases": int(weights.sum()),
                    "recall_features": rec.height / feats.height,
                    "recall_cases": float(rec["n_cases"].sum() / weights.sum()),
                    "median_boundary_error_aa": rec["boundary_error_aa"].median(),
                    "median_overlap_fraction": rec[
                        "overlap_fraction_of_feature"
                    ].median(),
                    "ms_per_protein_warm": ms_per_protein,
                }
            )

    a.tables.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(curve_rows).write_csv(
        a.tables / "301_step1_scan_curve_nb244_swissprot.csv"
    )
    pl.DataFrame(feat_rows).write_csv(
        a.tables / "301_step1_scan_features_nb244_swissprot.csv"
    )
    s = pl.DataFrame(summary_rows)
    s.write_csv(a.tables / "301_step1_scan_summary_nb244_swissprot.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=250):
        print(s)


if __name__ == "__main__":
    main()
