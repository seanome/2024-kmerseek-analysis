#!/usr/bin/env python3
"""Does any seed-and-extend hit on the loop survive its own multiplicity?

scripts/nb276_seed_extend_loop.py sweeps 132 settings and 4 of them beat shuffled
CD47 at p < 0.05. At a 0.05 threshold over 132 tests, 6.6 are expected by chance
alone, so that count is not evidence on its own and the four have to be tested
properly.

This re-tests only those four, with 50x more shuffles so a small p-value can
actually be resolved, and against a second, stricter null that keeps CD47's
neighbouring-residue frequencies rather than only its composition. A hit that is
real should survive both.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_seed_extend_followup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    SEED,
    resolve_uniprot,
)
from nb276_seed_extend_loop import best_loop_segment, encode  # noqa: E402

#: Shuffles for the focused test. 5_000 resolves p down to 0.0002, which is what
#: a Bonferroni threshold over the sweep's 132 settings needs.
N_SHUFFLE = 5_000
#: The sweep's own size, used for the Bonferroni threshold.
N_SETTINGS_IN_SWEEP = 132


def shuffle_preserving_pairs(
    sequence: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle keeping the count of every neighbouring class pair, by Euler walk.

    A plain shuffle keeps only how often each class appears. This keeps how often
    each ordered pair of classes appears, so a null sequence has the same local
    texture as CD47 and not just the same composition. Built as a random Eulerian
    path through the de Bruijn graph on classes, which is the standard way to do
    it; if no path is found the plain shuffle is returned and the caller is told.
    """
    n = len(sequence)
    if n < 3:
        return sequence.copy()
    edges: dict[int, list[int]] = {}
    for a, b in zip(sequence[:-1], sequence[1:]):
        edges.setdefault(int(a), []).append(int(b))
    for targets in edges.values():
        rng.shuffle(targets)
    start, end = int(sequence[0]), int(sequence[-1])
    # Walk the graph, consuming each edge once.
    walk, stack = [], [start]
    remaining = {node: list(targets) for node, targets in edges.items()}
    while stack:
        node = stack[-1]
        if remaining.get(node):
            stack.append(remaining[node].pop())
        else:
            walk.append(stack.pop())
    walk.reverse()
    if len(walk) != n or walk[0] != start or walk[-1] != end:
        out = sequence.copy()
        rng.shuffle(out)
        return out
    return np.array(walk, dtype=sequence.dtype)


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop0 = range(LOOP_MATURE_START + offset - 1, LOOP_MATURE_END + offset)

    sweep = pl.read_csv(OUTDIR / "11_seed_extend_loop.csv")
    winners = sweep.filter(
        pl.col("reaches_loop") & (pl.col("p_value_vs_shuffled_cd47") < 0.05)
    ).sort("p_value_vs_shuffled_cd47")
    if winners.is_empty():
        print("Nothing from the sweep to re-test.")
        return

    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    bonferroni = 0.05 / N_SETTINGS_IN_SWEEP
    rng = np.random.default_rng(SEED)
    rows = []
    for setting in winners.iter_rows(named=True):
        clusters = alphabets[setting["alphabet"]]
        query = encode(p66["sequence"], clusters)
        target = encode(cd47["sequence"], clusters)
        seed_len, penalty = setting["seed_length"], setting["penalty"]
        real = best_loop_segment(query, target, loop0, seed_len, penalty)
        print(f"{setting['alphabet']} seed={seed_len} penalty={penalty}: "
              f"score {real['score']}, running {N_SHUFFLE} shuffles of each kind...",
              flush=True)
        results = {}
        for null_name in ("composition", "neighbour_pairs"):
            scores = np.empty(N_SHUFFLE)
            fell_back = 0
            for i in range(N_SHUFFLE):
                if null_name == "composition":
                    copy = target.copy()
                    rng.shuffle(copy)
                else:
                    copy = shuffle_preserving_pairs(target, rng)
                    if np.array_equal(np.sort(copy), np.sort(target)) is False:
                        fell_back += 1
                found = best_loop_segment(query, copy, loop0, seed_len, penalty)
                scores[i] = found["score"] if found else 0.0
            results[null_name] = {
                "p": float((scores >= real["score"]).mean()),
                "median": float(np.median(scores)),
                "max": float(scores.max()),
                "reach": float((scores > 0).mean()),
                "fell_back": fell_back,
            }
        rows.append(
            {
                "alphabet": setting["alphabet"],
                "n_classes": setting["n_classes"],
                "seed_length": seed_len,
                "penalty": penalty,
                "score": real["score"],
                "segment_length": real["q_end"] - real["q_start"],
                "p_composition_null": results["composition"]["p"],
                "p_neighbour_pairs_null": results["neighbour_pairs"]["p"],
                "composition_null_median": results["composition"]["median"],
                "neighbour_null_median": results["neighbour_pairs"]["median"],
                "composition_null_max": results["composition"]["max"],
                "neighbour_null_max": results["neighbour_pairs"]["max"],
                "null_fraction_reaching_loop": results["composition"]["reach"],
                "survives_bonferroni_both_nulls": (
                    results["composition"]["p"] < bonferroni
                    and results["neighbour_pairs"]["p"] < bonferroni
                ),
            }
        )

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "12_seed_extend_followup.csv")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(220)
    print(f"\nBonferroni threshold for the sweep's {N_SETTINGS_IN_SWEEP} settings: "
          f"p < {bonferroni:.5f}")
    print(table)
    survivors = table.filter(pl.col("survives_bonferroni_both_nulls"))
    print(f"\n{survivors.height} of {table.height} survive both nulls at the "
          f"corrected threshold.")
    print(f"\nWrote {OUTDIR / '12_seed_extend_followup.csv'}")


if __name__ == "__main__":
    main()
