"""Seed-and-extend in a reduced alphabet, evaluated on known alignments (notebook 232).

For an aligned pair and an alphabet, every aligned column scores +1 when the two residues
share a class and -c when they do not; a gap column is a wall no segment may cross. A seed
is an exact run of at least `s` matching columns. The pair's score is the best-scoring
ungapped segment that contains a seed, which is what an X-drop extension converges to when
X is large. With s = 0 the seed requirement is dropped and the score is the best local
ungapped segment (Kadane), an upper bound on any seeded scheme.

The prefix-sum trick makes every seed length free once the column scores are known:
best segment containing [a, b) = score(a, b) + best extension leftward from a + best
extension rightward from b, and both extensions come from cumulative extrema of the
prefix sum.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hp_conservation_utils import GAP, TABLES, longest_true_run

WALL = -10_000.0


def column_scores(
    qa: np.ndarray, ta: np.ndarray, penalty: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column score and the match mask, with gap columns set to WALL and unmatched."""
    both = (qa != GAP) & (ta != GAP)
    same = (qa == ta) & both
    sc = np.where(same, 1.0, -penalty)
    sc[~both] = WALL
    return sc, same


def best_segment_containing_runs(
    sc: np.ndarray, same: np.ndarray, min_run: int
) -> float:
    """Best ungapped segment score among segments that contain a run of >= min_run matches.
    min_run = 0 returns the best local segment overall (0 if nothing positive)."""
    n = sc.size
    if n == 0:
        return 0.0
    P = np.concatenate(([0.0], np.cumsum(sc)))  # P[i] = sum(sc[:i])
    # Best extension leftward from column a: max(0, P[a] - min_{i<=a} P[i]).
    left = P - np.minimum.accumulate(P)
    # Best extension rightward from column b: max(0, max_{j>=b} P[j] - P[b]).
    right = np.maximum.accumulate(P[::-1])[::-1] - P
    if min_run == 0:
        return float(max(0.0, left.max()))
    padded = np.concatenate(([False], same, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_run
    if not keep.any():
        return 0.0
    a, b = starts[keep], ends[keep]
    seed = (b - a).astype(float)
    return float((seed + left[a] + right[b]).max())


def pair_seed_extend(
    qaln: str,
    taln: str,
    alphabet: str,
    penalties: list[float],
    seeds: list[int],
    n_shuffle: int,
    rng: np.random.Generator,
) -> list[dict]:
    tab = TABLES[alphabet]
    q = np.frombuffer(qaln.encode(), dtype=np.uint8)
    t = np.frombuffer(taln.encode(), dtype=np.uint8)
    qa, ta = tab[q], tab[t]
    t_res_idx = np.flatnonzero(ta != GAP)
    targets = [("real", ta)]
    for i in range(n_shuffle):
        ts = ta.copy()
        ts[t_res_idx] = ta[rng.permutation(t_res_idx)]
        targets.append((f"null{i}", ts))
    rows = []
    for c in penalties:
        for kind, tt in targets:
            sc, same = column_scores(qa, tt, c)
            for s in seeds:
                rows.append(
                    {
                        "alphabet": alphabet,
                        "penalty": c,
                        "seed": s,
                        "kind": kind,
                        "score": best_segment_containing_runs(sc, same, s),
                    }
                )
    return rows


def recall_at_null_quantile(
    df: pl.DataFrame, by: list[str], fpr: float
) -> pl.DataFrame:
    """Per group in `by`: threshold = (1-fpr) quantile of null scores (pooled over identity
    bins), recall = share of real pairs scoring strictly above it, per identity bin."""
    thr = (
        df.filter(pl.col("kind") != "real")
        .group_by(by)
        .agg(
            pl.col("score").quantile(1 - fpr, interpolation="higher").alias("threshold")
        )
    )
    real = df.filter(pl.col("kind") == "real").join(thr, on=by)
    return (
        real.group_by(by + ["identity_bin"])
        .agg(
            pl.len().alias("n_pairs"),
            (pl.col("score") > pl.col("threshold")).mean().alias("recall"),
            pl.col("threshold").first(),
        )
        .with_columns(pl.lit(fpr).alias("fpr"))
    )
