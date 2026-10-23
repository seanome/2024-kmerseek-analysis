#!/usr/bin/env python3
"""Seed-and-extend on the P66 / CD47 pair, asking whether it reaches the loop.

Exact k-mer matching cannot reach the claimed loop: step 9 of
scripts/nb276_p66_cd47_structure.py shows no k-mer covering P66 202-208 occurs in
CD47 at any setting kmerseek was run at. Seed-and-extend can in principle do
better, because a seed that lands beside the loop extends across it, and the
extension tolerates mismatches that an exact k-mer cannot.

So this asks the question the exact-match test cannot: with seeds anywhere in the
two proteins and ungapped extension in a reduced alphabet, does any segment cover
the loop, and in particular D205 and D207, better than chance?

notebooks/seed_extend_utils.py scores an alignment that already exists (notebook
232). P66 and CD47 have no alignment, which is the whole difficulty, so seeds are
found here by exact match in the reduced alphabet and each one is extended along
its own diagonal. The column scoring is the same as that module's: +1 when the two
residues share a class, -penalty when they do not.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_seed_extend_loop.py
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
    LOOP_CRITICAL_MATURE,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    SEED,
    resolve_uniprot,
)

#: Seed lengths tried. 4 is shorter than any exact k kmerseek is run at, so if
#: seed-and-extend can reach the loop at all it will be somewhere in this range.
SEED_LENGTHS = (4, 5, 6, 7)
#: Mismatch penalties, the same values notebook 232 sweeps.
PENALTIES = (0.5, 1.0, 2.0)
#: X-drop: stop extending once the score falls this far below the best so far.
XDROP = 10.0
#: Shuffles of the target used as the null. Each one keeps CD47's composition.
N_SHUFFLE = 100

#: Alphabets where this question is live. The eight that write the loop as one
#: repeated class are left out: there a seed of any length matches every polar
#: stretch in CD47, so a segment covering the loop is guaranteed and says nothing.
#: Step 9 already shows what those alphabets do, and leaving them in makes the
#: null 200 times slower for an answer that is true by construction.
UNIFORM_ON_THE_LOOP = {
    "hp_kyte_doolittle2",
    "hp_lehninger2",
    "hp_lehninger_c_nonpolar2",
    "hp_pbotc_1st_ed2",
    "hp_thomas_dill2",
    "hp_thomas_dill_no_c2",
    "hp_lehninger_hpc3",
    "gbmr4",
}


def encode(sequence: str, clusters: list[str]) -> np.ndarray:
    table = {residue: i for i, cl in enumerate(clusters) for residue in cl}
    return np.array([table.get(residue, -1) for residue in sequence], dtype=np.int16)


def extend_from_seed(
    query: np.ndarray,
    target: np.ndarray,
    q_seed: int,
    t_seed: int,
    seed_len: int,
    penalty: float,
    xdrop: float,
) -> tuple[int, int, int, float]:
    """Extend one seed ungapped along its diagonal, X-drop in both directions.

    Returns the segment as (query start, query end, target start, score), all
    0-based half-open on the query, and the score under the +1 / -penalty scheme.
    """

    def run(step: int) -> tuple[int, float]:
        """Walk one direction, returning how far the best score reached."""
        best_score, best_offset, score = 0.0, 0, 0.0
        offset = 0
        while True:
            offset += 1
            qi = q_seed + (seed_len - 1 + offset if step > 0 else -offset)
            ti = t_seed + (seed_len - 1 + offset if step > 0 else -offset)
            if not (0 <= qi < len(query) and 0 <= ti < len(target)):
                break
            score += 1.0 if query[qi] == target[ti] and query[qi] >= 0 else -penalty
            if score > best_score:
                best_score, best_offset = score, offset
            if best_score - score > xdrop:
                break
        return best_offset, best_score

    left_offset, left_score = run(-1)
    right_offset, right_score = run(+1)
    q_start = q_seed - left_offset
    q_end = q_seed + seed_len + right_offset  # half-open
    return q_start, q_end, t_seed - left_offset, seed_len + left_score + right_score


def best_loop_segment(
    query: np.ndarray,
    target: np.ndarray,
    loop0: range,
    seed_len: int,
    penalty: float,
) -> dict | None:
    """The best-scoring extended segment that covers the loop, or None.

    Seeds are every exact match of `seed_len` classes between the two sequences.
    Only segments whose query span overlaps the loop count.
    """
    index: dict[bytes, list[int]] = {}
    for i in range(len(target) - seed_len + 1):
        window = target[i : i + seed_len]
        if window.min() < 0:
            continue
        index.setdefault(window.tobytes(), []).append(i)

    best = None
    for qi in range(len(query) - seed_len + 1):
        window = query[qi : qi + seed_len]
        if window.min() < 0:
            continue
        for ti in index.get(window.tobytes(), ()):
            q_start, q_end, t_start, score = extend_from_seed(
                query, target, qi, ti, seed_len, penalty, XDROP
            )
            covered = sorted(set(range(q_start, q_end)) & set(loop0))
            if not covered:
                continue
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "q_start": q_start,
                    "q_end": q_end,
                    "t_start": t_start,
                    "seed_q": qi,
                    "seed_t": ti,
                    "n_loop_covered": len(covered),
                    "loop_covered": covered,
                }
    return best


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop_start, loop_end = LOOP_MATURE_START + offset, LOOP_MATURE_END + offset
    loop0 = range(loop_start - 1, loop_end)  # 0-based
    critical0 = {pos + offset - 1 for pos in LOOP_CRITICAL_MATURE}
    p66_seq, cd47_seq = p66["sequence"], cd47["sequence"]

    alphabets = {
        name: clusters
        for name, clusters in {
            **hc.ALPHABET_CLUSTERS,
            **EXTRA_ALPHABET_CLUSTERS,
        }.items()
        if name not in UNIFORM_ON_THE_LOOP
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for name, clusters in sorted(alphabets.items()):
        print(f"  {name} ({len(clusters)} classes)...", flush=True)
        query = encode(p66_seq, clusters)
        target = encode(cd47_seq, clusters)
        shuffled = []
        for _ in range(N_SHUFFLE):
            copy = target.copy()
            rng.shuffle(copy)
            shuffled.append(copy)
        for seed_len in SEED_LENGTHS:
            for penalty in PENALTIES:
                real = best_loop_segment(query, target, loop0, seed_len, penalty)
                null_scores = []
                for copy in shuffled:
                    found = best_loop_segment(query, copy, loop0, seed_len, penalty)
                    null_scores.append(found["score"] if found else 0.0)
                null_scores = np.array(null_scores)
                row = {
                    "alphabet": name,
                    "n_classes": len(clusters),
                    "seed_length": seed_len,
                    "penalty": penalty,
                    "reaches_loop": real is not None,
                    "score": real["score"] if real else None,
                    "null_median": float(np.median(null_scores)),
                    "null_max": float(null_scores.max()),
                    "null_fraction_reaching_loop": float((null_scores > 0).mean()),
                }
                if real:
                    row |= {
                        "p66_span": f"{real['q_start'] + 1}-{real['q_end']}",
                        "cd47_span": f"{real['t_start'] + 1}-"
                                     f"{real['t_start'] + (real['q_end'] - real['q_start'])}",
                        "p66_residues": p66_seq[real["q_start"] : real["q_end"]],
                        "cd47_residues": cd47_seq[
                            real["t_start"] : real["t_start"]
                            + (real["q_end"] - real["q_start"])
                        ],
                        "n_loop_residues_covered": real["n_loop_covered"],
                        "covers_both_aspartates": set(real["loop_covered"])
                        >= critical0,
                        "p_value_vs_shuffled_cd47": float(
                            (null_scores >= real["score"]).mean()
                        ),
                    }
                rows.append(row)

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "11_seed_extend_loop.csv")

    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(220)
    print(f"Loop: P66 {loop_start}-{loop_end} = {p66_seq[loop_start - 1:loop_end]}, "
          f"aspartates at {sorted(p + 1 for p in critical0)}")
    print(f"{len(alphabets)} alphabets tested; {len(UNIFORM_ON_THE_LOOP)} left out "
          f"because they write the loop as one repeated class.\n")
    reached = table.filter(pl.col("reaches_loop"))
    print(f"{reached.height} of {table.height} settings produce an extended segment "
          f"covering the loop at all.")
    if reached.is_empty():
        print("Seed-and-extend does not reach the loop in any alphabet.")
        return
    beats = reached.filter(pl.col("p_value_vs_shuffled_cd47") < 0.05)
    print(f"{beats.height} of those beat shuffled CD47 at p < 0.05.")
    print(f"{reached.filter(pl.col('covers_both_aspartates')).height} cover both "
          f"D{sorted(LOOP_CRITICAL_MATURE)[0] + offset} and "
          f"D{sorted(LOOP_CRITICAL_MATURE)[1] + offset}.\n")
    print(reached.select(
        "alphabet", "seed_length", "penalty", "score", "null_median", "null_max",
        "p_value_vs_shuffled_cd47", "n_loop_residues_covered",
        "covers_both_aspartates", "p66_span", "cd47_span",
    ).sort("p_value_vs_shuffled_cd47", "alphabet"))
    print(f"\nWrote {OUTDIR / '11_seed_extend_loop.csv'}")


if __name__ == "__main__":
    main()
