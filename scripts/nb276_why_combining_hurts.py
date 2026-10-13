#!/usr/bin/env python3
"""Why does averaging alphabets hurt, when each one should corroborate the other?

The intuition is sound and the implementation in
scripts/nb276_combined_alphabets.py is what breaks it. That script scored a
column by the share of alphabets putting the two residues in the same class. An
average is not corroboration: it is dilution, and three measurements show why.

  1. **Alphabets disagree by construction, because they differ in size.** For a
     random pair of residues, a 2-class alphabet puts them together about half
     the time and protein20 almost never. The share-agreeing score for any pair
     is therefore driven by how many coarse alphabets are in the panel, not by
     whether the two residues resemble each other.
  2. **A real match stops being worth +1.** Column by column over the segment
     hsdm17 actually found, this prints what each column earns under hsdm17 and
     what the same column earns under the average. A column hsdm17 calls a match
     can fall to near zero once the other alphabets vote.
  3. **So the extension dies.** With columns worth a fraction of what they were,
     X-drop stops after a few residues, which is exactly what the combination
     runs showed: 5 to 7 residues instead of 36.

Then the test that does what corroboration should mean. Rather than averaging
per column, use the alphabets as separate tests of the same segment: hsdm17
proposes P66 197-232 against CD47 109-144, and every other alphabet is asked
whether that same segment scores above its own matched-protein background. That
is a conjunction of independent checks, not an average, and it is the version of
the idea worth running.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_why_combining_hurts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_combined_alphabets import (  # noqa: E402
    agreement_matrix,
    class_tables,
    to_index,
)
from nb276_hsdm17_prespecified_test import (  # noqa: E402
    ACIDIC_TOLERANCE_PP,
    HUMAN_PROTEOME,
    LENGTH_TOLERANCE,
    N_BACKGROUND,
    PENALTY,
    acidic_fraction,
    read_fasta,
)
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    OUTDIR,
    SEED,
    resolve_uniprot,
)
from nb276_seed_extend_loop import encode  # noqa: E402

#: The segment hsdm17 found, in each protein's UniProt numbering.
P66_SEGMENT = (197, 232)
CD47_SEGMENT = (109, 144)
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def collision_probability(table: dict[str, int]) -> float:
    """Chance two residues drawn uniformly at random share a class."""
    pairs = [(a, b) for a in RESIDUES for b in RESIDUES]
    return sum(1 for a, b in pairs if table[a] == table[b]) / len(pairs)


def segment_score(
    p66_seq: str, target: str, t_start: int, clusters: list[str], penalty: float
) -> float:
    """Ungapped score of the fixed segment against a target at a fixed offset."""
    query = encode(p66_seq, clusters)
    code = encode(target, clusters)
    total = 0.0
    for i in range(P66_SEGMENT[1] - P66_SEGMENT[0] + 1):
        qi = P66_SEGMENT[0] - 1 + i
        ti = t_start - 1 + i
        if not (0 <= qi < len(query) and 0 <= ti < len(code)):
            return total
        total += 1.0 if query[qi] == code[ti] and query[qi] >= 0 else -penalty
    return total


def best_fixed_length_score(
    p66_seq: str, target: str, clusters: list[str], penalty: float
) -> float:
    """Best score of the fixed P66 segment against any offset in the target."""
    query = encode(p66_seq, clusters)
    code = encode(target, clusters)
    length = P66_SEGMENT[1] - P66_SEGMENT[0] + 1
    q = query[P66_SEGMENT[0] - 1 : P66_SEGMENT[1]]
    best = -np.inf
    for start in range(len(code) - length + 1):
        window = code[start : start + length]
        same = sum(
            1 for a, b in zip(q, window) if a == b and a >= 0
        )
        best = max(best, same - penalty * (length - same))
    return float(best)


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    p66_seq, cd47_seq = p66["sequence"], cd47["sequence"]
    all_alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    names = sorted(all_alphabets)

    # --- 1. alphabets disagree because they differ in size ---------------
    coll = pl.DataFrame(
        [
            {
                "alphabet": name,
                "n_classes": len(all_alphabets[name]),
                "chance_two_residues_share_a_class": round(
                    collision_probability(class_tables([name])[0]), 3
                ),
            }
            for name in names
        ]
    ).sort("n_classes", "alphabet")
    coll.write_csv(OUTDIR / "26_alphabet_collision_rates.csv")
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    print("1. HOW OFTEN EACH ALPHABET CALLS A RANDOM PAIR OF RESIDUES A MATCH")
    print(coll)
    mean_chance = float(coll["chance_two_residues_share_a_class"].mean())
    print(f"\nMean over the panel: {mean_chance:.3f}. The share-agreeing score for a "
          f"random pair\nis about this, so the average is set by how many coarse "
          f"alphabets are in the panel.")

    # --- 2. the same columns under one alphabet and under the average ----
    tables = class_tables(names)
    matrix = agreement_matrix(tables)
    idx = to_index(p66_seq)
    cd47_idx = to_index(cd47_seq)
    hsdm17 = all_alphabets["hsdm17"]
    hs_q = encode(p66_seq, hsdm17)
    hs_t = encode(cd47_seq, hsdm17)

    rows = []
    for i in range(P66_SEGMENT[1] - P66_SEGMENT[0] + 1):
        qi = P66_SEGMENT[0] - 1 + i
        ti = CD47_SEGMENT[0] - 1 + i
        hsdm17_col = 1.0 if hs_q[qi] == hs_t[ti] else -PENALTY
        agree = matrix[idx[qi], cd47_idx[ti]]
        rows.append(
            {
                "p66_position": qi + 1,
                "p66_residue": p66_seq[qi],
                "cd47_position": ti + 1,
                "cd47_residue": cd47_seq[ti],
                "identical": p66_seq[qi] == cd47_seq[ti],
                "hsdm17_column_score": hsdm17_col,
                "share_of_19_alphabets_agreeing": round((agree + 1) / 2, 3),
                "averaged_column_score": round(float(agree), 3),
            }
        )
    columns = pl.DataFrame(rows)
    columns.write_csv(OUTDIR / "27_segment_columns_one_vs_average.csv")
    print("\n2. THE SAME 36 COLUMNS, UNDER hsdm17 AND UNDER THE AVERAGE")
    print(columns)
    matched = columns.filter(pl.col("hsdm17_column_score") > 0)
    print(f"\nColumns hsdm17 calls a match: {matched.height}, total "
          f"{matched['hsdm17_column_score'].sum():.1f} under hsdm17 and "
          f"{matched['averaged_column_score'].sum():.2f} under the average.")
    print(f"Whole segment: {columns['hsdm17_column_score'].sum():.1f} under hsdm17, "
          f"{columns['averaged_column_score'].sum():.2f} under the average.")
    print("That is point 3: with the segment worth less than nothing under the "
          "average,\nno X-drop extension can build it, which is why the combination "
          "runs stopped at 5 to 7 residues.")

    # --- the conjunction test -------------------------------------------
    cd47_acidic = acidic_fraction(cd47_seq)
    human = read_fasta(HUMAN_PROTEOME)
    low = int(len(cd47_seq) * (1 - LENGTH_TOLERANCE))
    high = int(len(cd47_seq) * (1 + LENGTH_TOLERANCE))
    eligible = [
        (n, s)
        for n, s in human.items()
        if low <= len(s) <= high
        and s.isalpha()
        and abs(acidic_fraction(s) - cd47_acidic) * 100 <= ACIDIC_TOLERANCE_PP
        and n != "CD47"
        and s != cd47_seq
    ]
    rng = np.random.default_rng(SEED)
    picked = [
        eligible[i]
        for i in sorted(
            rng.choice(len(eligible), size=min(N_BACKGROUND, len(eligible)),
                       replace=False)
        )
    ]
    print(f"\n3. CORROBORATION AS A CONJUNCTION, not an average")
    print(f"The fixed segment P66 {P66_SEGMENT[0]}-{P66_SEGMENT[1]} against CD47 "
          f"{CD47_SEGMENT[0]}-{CD47_SEGMENT[1]}, scored in each alphabet on its own,")
    print(f"against the same {len(picked)} matched human proteins.\n")
    conj = []
    for name in names:
        clusters = all_alphabets[name]
        observed = segment_score(p66_seq, cd47_seq, CD47_SEGMENT[0], clusters, PENALTY)
        nulls = np.array(
            [best_fixed_length_score(p66_seq, s, clusters, PENALTY) for _, s in picked]
        )
        n_ge = int((nulls >= observed).sum())
        conj.append(
            {
                "alphabet": name,
                "n_classes": len(clusters),
                "segment_score_on_cd47": observed,
                "background_median": float(np.median(nulls)),
                "background_max": float(nulls.max()),
                "n_background_ge": n_ge,
                "n_background": len(picked),
                "p_value": n_ge / len(picked),
                "corroborates_at_0_05": n_ge / len(picked) < 0.05,
            }
        )
    conjunction = pl.DataFrame(conj).sort("p_value")
    conjunction.write_csv(OUTDIR / "28_segment_corroboration_by_alphabet.csv")
    print(conjunction)
    n_yes = int(conjunction["corroborates_at_0_05"].sum())
    print(f"\n{n_yes} of {conjunction.height} alphabets independently find this "
          f"segment better than their own matched background at p < 0.05.")
    print(f"Bonferroni over {conjunction.height} alphabets is p < "
          f"{0.05 / conjunction.height:.4f}; "
          f"{int((conjunction['p_value'] < 0.05 / conjunction.height).sum())} clear that.")
    print(f"\nWrote {OUTDIR / '28_segment_corroboration_by_alphabet.csv'}")


if __name__ == "__main__":
    main()
