#!/usr/bin/env python3
"""Does combining alphabets do better on the loop than any single one?

Every test so far used one alphabet at a time. Two alphabets can be combined two
ways, and they pull in opposite directions:

  * **Intersection, as a product alphabet.** Two residues are in the same class
    only if they are in the same class in both alphabets. This is a finer
    alphabet, so exact matches get rarer and more specific. polarity4 and
    funcgroups8 both separate acidic from basic but cut the hydrophobic residues
    differently, so their product is finer than either.
  * **Agreement, as a graded score.** Instead of same class or not, a column
    scores by how many of the chosen alphabets put the two residues together.
    This is softer than any single alphabet and amounts to a cheap substitution
    matrix built out of alphabets rather than out of aligned counts.

The scheme tested here uses both at once, with no free penalty parameter:

  column score   2 * (share of alphabets where the two residues share a class) - 1,
                 so +1 when all agree and -1 when none do
  seed           `seed_length` consecutive columns where every alphabet agrees,
                 which is an exact match in the product alphabet
  extension      X-drop along the diagonal, as before

Two combinations are fixed in advance, and only two, so the correction is small:

  1. polarity4 + funcgroups8, the pair Olga suggested
  2. all 19 alphabets in the repo's table, the whole panel

Two seeding rules are tried, and the first one fails for a reason worth keeping:

  **product seeding** (prespecified first): a seed must match in every alphabet
  of the combination at once. For the full panel the product alphabet turns out
  to have 20 classes over the 20 residues, which is to say requiring every
  alphabet to agree is requiring identity. Neither combination reaches the loop
  under this rule, and it is strictly worse than any single alphabet. Combining
  by intersection defeats the purpose.

  **coarse seeding** (the repair): the seed comes from polarity4 alone, which is
  a member of both combinations and coarse enough that seeds exist, and the
  agreement score of the whole combination judges the extension. This is what
  "combine a few alphabets" should mean: a coarse alphabet proposes, the
  combination decides. It is reported as a repair of the rule above, not as a
  third prespecified test.

Each is judged the way the single-alphabet lead was: against the same 437 human
proteins matched to CD47 on length and acidic fraction, with the score split into
the loop's own columns and the flanking ones.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_combined_alphabets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_hsdm17_prespecified_test import (  # noqa: E402
    ACIDIC_TOLERANCE_PP,
    HUMAN_PROTEOME,
    LENGTH_TOLERANCE,
    N_BACKGROUND,
    acidic_fraction,
    read_fasta,
)
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    SEED,
    resolve_uniprot,
)

#: The combinations, fixed before the test. Two, so the corrected threshold is
#: 0.05 / 2 rather than anything like the 132-setting sweep's.
COMBINATIONS: dict[str, list[str] | None] = {
    "polarity4 + funcgroups8": ["polarity4", "funcgroups8"],
    "all 19 alphabets": None,  # None means every alphabet in the table
}
#: Seed lengths tried per combination. Prespecified, and reported for all of them
#: rather than picking the best afterwards.
SEED_LENGTHS = (4, 5, 6)
#: X-drop, as everywhere else in this notebook.
XDROP = 10.0
ALPHA = 0.05
#: The alphabet that proposes seeds under the coarse-seeding rule. It belongs to
#: both combinations and has four classes, so seeds exist where the product
#: alphabet has none.
SEEDING_ALPHABET = "polarity4"
#: Which seeding rules to run.
SEEDING_RULES = ("product", "coarse")


def class_tables(names: list[str]) -> list[dict[str, int]]:
    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    return [
        {residue: i for i, cl in enumerate(alphabets[name]) for residue in cl}
        for name in names
    ]


def agreement_matrix(tables: list[dict[str, int]]) -> np.ndarray:
    """A 26x26 share-of-alphabets-agreeing matrix over the 20 standard residues."""
    residues = "ACDEFGHIKLMNPQRSTVWY"
    index = {r: i for i, r in enumerate(residues)}
    matrix = np.full((26, 26), -1.0)
    for a in residues:
        for b in residues:
            agree = sum(
                1
                for table in tables
                if a in table and b in table and table[a] == table[b]
            )
            matrix[index[a], index[b]] = 2.0 * agree / len(tables) - 1.0
    return matrix


def to_index(sequence: str) -> np.ndarray:
    residues = "ACDEFGHIKLMNPQRSTVWY"
    index = {r: i for i, r in enumerate(residues)}
    return np.array([index.get(r, -1) for r in sequence], dtype=np.int16)


def product_classes(sequence: str, tables: list[dict[str, int]]) -> list[tuple]:
    """Each residue's class in the product alphabet, as a tuple of its classes."""
    return [
        tuple(table.get(r, -1 - i) for i, table in enumerate(tables))
        for r in sequence
    ]


def best_loop_segment_combined(
    query_idx: np.ndarray,
    target_idx: np.ndarray,
    query_product: list[tuple],
    target_product: list[tuple],
    score_matrix: np.ndarray,
    loop0: range,
    seed_len: int,
) -> dict | None:
    """Best agreement-scored segment covering the loop.

    `query_product` and `target_product` are whatever the seeding rule uses: the
    product alphabet's class tuples, or one coarse alphabet's classes. The score
    is always the combination's agreement score.
    """
    index: dict[tuple, list[int]] = {}
    for i in range(len(target_product) - seed_len + 1):
        key = tuple(target_product[i : i + seed_len])
        index.setdefault(key, []).append(i)

    def column(qi, ti):
        a, b = query_idx[qi], target_idx[ti]
        return score_matrix[a, b] if a >= 0 and b >= 0 else -1.0

    best = None
    for qi in range(len(query_product) - seed_len + 1):
        key = tuple(query_product[qi : qi + seed_len])
        for ti in index.get(key, ()):
            seed_score = sum(column(qi + j, ti + j) for j in range(seed_len))

            def walk(step):
                best_score, best_offset, score = 0.0, 0, 0.0
                offset = 0
                while True:
                    offset += 1
                    q = qi + (seed_len - 1 + offset if step > 0 else -offset)
                    t = ti + (seed_len - 1 + offset if step > 0 else -offset)
                    if not (0 <= q < len(query_idx) and 0 <= t < len(target_idx)):
                        break
                    score += column(q, t)
                    if score > best_score:
                        best_score, best_offset = score, offset
                    if best_score - score > XDROP:
                        break
                return best_offset, best_score

            left_offset, left_score = walk(-1)
            right_offset, right_score = walk(+1)
            q_start = qi - left_offset
            q_end = qi + seed_len + right_offset
            total = seed_score + left_score + right_score
            if not set(range(q_start, q_end)) & set(loop0):
                continue
            if best is None or total > best["score"]:
                best = {
                    "score": float(total),
                    "q_start": q_start,
                    "q_end": q_end,
                    "t_start": ti - left_offset,
                }
    return best


def split_by_loop(
    best: dict,
    query_idx: np.ndarray,
    target_idx: np.ndarray,
    score_matrix: np.ndarray,
    loop: tuple[int, int],
) -> tuple[float, float]:
    loop_score = flank_score = 0.0
    for i in range(best["q_end"] - best["q_start"]):
        qi, ti = best["q_start"] + i, best["t_start"] + i
        if not (0 <= qi < len(query_idx) and 0 <= ti < len(target_idx)):
            continue
        a, b = query_idx[qi], target_idx[ti]
        value = score_matrix[a, b] if a >= 0 and b >= 0 else -1.0
        if loop[0] - 1 <= qi <= loop[1] - 1:
            loop_score += value
        else:
            flank_score += value
    return loop_score, flank_score


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)
    loop0 = range(loop[0] - 1, loop[1])
    all_names = sorted({**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS})

    cd47_acidic = acidic_fraction(cd47["sequence"])
    human = read_fasta(HUMAN_PROTEOME)
    low = int(len(cd47["sequence"]) * (1 - LENGTH_TOLERANCE))
    high = int(len(cd47["sequence"]) * (1 + LENGTH_TOLERANCE))
    eligible = [
        (name, seq)
        for name, seq in human.items()
        if low <= len(seq) <= high
        and seq.isalpha()
        and abs(acidic_fraction(seq) - cd47_acidic) * 100 <= ACIDIC_TOLERANCE_PP
        and name != "CD47"
        and seq != cd47["sequence"]
    ]
    rng = np.random.default_rng(SEED)
    picked = [
        eligible[i]
        for i in sorted(
            rng.choice(
                len(eligible), size=min(N_BACKGROUND, len(eligible)), replace=False
            )
        )
    ]
    print(f"{len(picked)} matched human proteins, the same set as the single-alphabet "
          f"test.\n")

    query_idx = to_index(p66["sequence"])
    rows = []
    for label, names in COMBINATIONS.items():
        names = names or all_names
        tables = class_tables(names)
        matrix = agreement_matrix(tables)
        query_product = product_classes(p66["sequence"], tables)
        cd47_product = product_classes(cd47["sequence"], tables)
        cd47_idx = to_index(cd47["sequence"])

        distinct = len({tuple(t) for t in product_classes("ACDEFGHIKLMNPQRSTVWY", tables)})
        loop_string = " ".join(
            "".join(str(c) for c in t)
            for t in query_product[loop[0] - 1 : loop[1]]
        )
        print(f"=== {label} ===")
        print(f"  {len(names)} alphabets, product alphabet has {distinct} classes "
              f"over the 20 residues")
        print(f"  the loop in the product alphabet: {loop_string}")

        seeding_table = class_tables([SEEDING_ALPHABET])
        query_coarse = product_classes(p66["sequence"], seeding_table)
        cd47_coarse = product_classes(cd47["sequence"], seeding_table)

        for rule in SEEDING_RULES:
          q_seed_src = query_product if rule == "product" else query_coarse
          t_seed_src = cd47_product if rule == "product" else cd47_coarse
          for seed_len in SEED_LENGTHS:
            observed = best_loop_segment_combined(
                query_idx, cd47_idx, q_seed_src, t_seed_src, matrix,
                loop0, seed_len,
            )
            if observed is None:
                print(f"  {rule} seeding, seed {seed_len}: does not reach the loop")
                rows.append({"combination": label, "n_alphabets": len(names),
                             "product_classes": distinct, "seeding_rule": rule,
                             "seed_length": seed_len, "reaches_loop": False})
                continue
            obs_loop, obs_flank = split_by_loop(
                observed, query_idx, cd47_idx, matrix, loop
            )
            scores, loop_scores, flank_scores, reached = [], [], [], 0
            for name, sequence in picked:
                t_idx = to_index(sequence)
                t_seed = product_classes(
                    sequence, tables if rule == "product" else seeding_table
                )
                found = best_loop_segment_combined(
                    query_idx, t_idx, q_seed_src, t_seed, matrix,
                    loop0, seed_len,
                )
                if found is None:
                    scores.append(-np.inf)
                    continue
                reached += 1
                scores.append(found["score"])
                lo, fl = split_by_loop(found, query_idx, t_idx, matrix, loop)
                loop_scores.append(lo)
                flank_scores.append(fl)
            finite = np.array([s for s in scores if np.isfinite(s)])
            n_ge = int((finite >= observed["score"]).sum())
            n_loop_ge = int((np.array(loop_scores) >= obs_loop).sum()) if loop_scores else 0
            rows.append(
                {
                    "combination": label,
                    "n_alphabets": len(names),
                    "product_classes": distinct,
                    "seeding_rule": rule,
                    "seed_length": seed_len,
                    "reaches_loop": True,
                    "score": round(observed["score"], 3),
                    "segment_length": observed["q_end"] - observed["q_start"],
                    "p66_span": f"{observed['q_start'] + 1}-{observed['q_end']}",
                    "cd47_span": f"{observed['t_start'] + 1}-"
                                 f"{observed['t_start'] + observed['q_end'] - observed['q_start']}",
                    "loop_score": round(obs_loop, 3),
                    "flank_score": round(obs_flank, 3),
                    "n_background": len(picked),
                    "background_reaching_loop": reached,
                    "background_median_score": round(float(np.median(finite)), 3)
                    if finite.size
                    else None,
                    "background_max_score": round(float(finite.max()), 3)
                    if finite.size
                    else None,
                    "n_background_ge_total": n_ge,
                    "p_total": n_ge / len(picked),
                    "n_background_ge_on_loop": n_loop_ge,
                    "p_loop_among_reaching": n_loop_ge / reached if reached else None,
                    "background_loop_median": round(float(np.median(loop_scores)), 3)
                    if loop_scores
                    else None,
                }
            )
            print(f"  {rule} seeding, seed {seed_len}: score {observed['score']:.2f} over "
                  f"{observed['q_end'] - observed['q_start']} residues, "
                  f"loop {obs_loop:.2f} flanks {obs_flank:.2f}; "
                  f"{n_ge}/{len(picked)} background reach it (p = {n_ge / len(picked):.3f}); "
                  f"on the loop {n_loop_ge}/{reached} beat it")
        print()

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "25_combined_alphabets.csv")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(240)
    print(table.select([c for c in table.columns if c not in ("p66_span", "cd47_span")]))
    print(f"\nTwo combinations were fixed in advance, so the corrected threshold is "
          f"p < {ALPHA / len(COMBINATIONS):.3f}.")
    reached = table.filter(pl.col("reaches_loop"))
    if not reached.is_empty():
        best = reached.sort("p_total").row(0, named=True)
        print(f"Best combination: {best['combination']} seed {best['seed_length']}, "
              f"p = {best['p_total']:.3f} on the whole segment.")
        print(f"  On the loop's own columns, {best['n_background_ge_on_loop']} of "
              f"{best['background_reaching_loop']} matched proteins beat it.")
    print(f"\nWrote {OUTDIR / '25_combined_alphabets.csv'}")


if __name__ == "__main__":
    main()
