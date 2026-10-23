#!/usr/bin/env python3
"""Why does funcgroups8 fail at 8 classes, when dayhoff6 and gbmr7 are coarser?

Class count is not what decides it. Three things are, and this measures all three.

  1. **How often an alphabet calls a random pair a match.** funcgroups8 splits the
     residues more evenly than its class count suggests, so its chance rate is
     0.145, lower than dayhoff6's 0.19 at 6 classes. A low chance rate is good,
     so this is not the reason either.
  2. **Signal against that chance rate.** What matters is how many of the
     segment's 36 columns an alphabet calls a match, against how many it would
     call by chance. This prints observed, expected and the excess per alphabet.
  3. **Which residue pairs do the work.** An alphabet only scores this segment if
     it happens to merge the particular residue pairs the segment contains. This
     lists, per alphabet, every matched column's residue pair, so the merges that
     carry the score can be read off.

Point 3 matters beyond funcgroups8. hsdm17 has a class `KE` and sdm12 has `KER`,
both merging a basic residue with an acidic one, which most alphabets do not. If
the two alphabets that cleared Bonferroni are both scoring this segment through
the same unusual merge, they are not two independent corroborations of it.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_which_merges_carry_the_segment.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    OUTDIR,
    resolve_uniprot,
)
from nb276_why_combining_hurts import (  # noqa: E402
    CD47_SEGMENT,
    P66_SEGMENT,
    RESIDUES,
    collision_probability,
)
from nb276_combined_alphabets import class_tables  # noqa: E402

#: Acidic and basic residues, for naming a merge that crosses charge.
ACIDIC, BASIC = set("DE"), set("KRH")


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    p66_part = p66["sequence"][P66_SEGMENT[0] - 1 : P66_SEGMENT[1]]
    cd47_part = cd47["sequence"][CD47_SEGMENT[0] - 1 : CD47_SEGMENT[1]]
    pairs = list(zip(p66_part, cd47_part))
    n_columns = len(pairs)

    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    corroboration = pl.read_csv(
        OUTDIR / "28_segment_corroboration_by_alphabet.csv"
    ).select("alphabet", "p_value", "corroborates_at_0_05", "segment_score_on_cd47")

    rows, pair_rows = [], []
    for name in sorted(alphabets):
        clusters = alphabets[name]
        table = class_tables([name])[0]
        chance = collision_probability(table)
        matched = [
            (a, b) for a, b in pairs if a in table and b in table and table[a] == table[b]
        ]
        identical = [(a, b) for a, b in matched if a == b]
        crossing = [
            (a, b)
            for a, b in matched
            if (a in ACIDIC and b in BASIC) or (a in BASIC and b in ACIDIC)
        ]
        expected = chance * n_columns
        rows.append(
            {
                "alphabet": name,
                "n_classes": len(clusters),
                "chance_a_random_pair_matches": round(chance, 3),
                "columns_called_a_match": len(matched),
                "expected_by_chance": round(expected, 1),
                "excess_over_chance": round(len(matched) - expected, 1),
                "of_those_identical": len(identical),
                "of_those_non_identical": len(matched) - len(identical),
                "of_those_acidic_against_basic": len(crossing),
                "largest_class_size": max(len(c) for c in clusters),
            }
        )
        for a, b in sorted(set(matched)):
            pair_rows.append(
                {
                    "alphabet": name,
                    "pair": f"{a}-{b}",
                    "identical": a == b,
                    "acidic_against_basic": (a in ACIDIC and b in BASIC)
                    or (a in BASIC and b in ACIDIC),
                    "times_in_segment": sum(
                        1 for x, y in matched if (x, y) == (a, b)
                    ),
                }
            )

    table = (
        pl.DataFrame(rows)
        .join(corroboration, on="alphabet", how="left")
        .sort("n_classes", "alphabet")
    )
    table.write_csv(OUTDIR / "30_segment_merges_by_alphabet.csv")
    pair_table = pl.DataFrame(pair_rows)
    pair_table.write_csv(OUTDIR / "31_segment_matched_pairs_by_alphabet.csv")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(230)
    print(f"The segment is {n_columns} columns:")
    print(f"  P66  {p66_part}")
    print(f"  CD47 {cd47_part}")
    print(f"  {sum(1 for a, b in pairs if a == b)} identical\n")
    print("Observed matched columns against what chance gives, per alphabet:")
    print(table.select(
        "alphabet", "n_classes", "chance_a_random_pair_matches",
        "columns_called_a_match", "expected_by_chance", "excess_over_chance",
        "of_those_non_identical", "of_those_acidic_against_basic",
        "p_value", "corroborates_at_0_05",
    ))

    print("\nfuncgroups8 against the two that cleared Bonferroni:")
    for name in ("funcgroups8", "sdm12", "hsdm17"):
        row = table.filter(pl.col("alphabet") == name).row(0, named=True)
        merges = pair_table.filter(
            (pl.col("alphabet") == name) & ~pl.col("identical")
        )
        print(f"\n  {name} ({row['n_classes']} classes), p = {row['p_value']:.3f}")
        print(f"    classes: {' '.join(alphabets[name])}")
        print(f"    matched {row['columns_called_a_match']} columns, "
              f"{row['expected_by_chance']} expected, excess "
              f"{row['excess_over_chance']}")
        print(f"    non-identical pairs it accepts: "
              f"{', '.join(merges['pair'].to_list()) or 'none'}")
        crossing = merges.filter(pl.col("acidic_against_basic"))["pair"].to_list()
        print(f"    of those, acidic against basic: "
              f"{', '.join(crossing) or 'none'}")

    print("\nAre the two Bonferroni-clearing alphabets independent?")
    sdm12_pairs = set(
        pair_table.filter(
            (pl.col("alphabet") == "sdm12") & ~pl.col("identical")
        )["pair"].to_list()
    )
    hsdm17_pairs = set(
        pair_table.filter(
            (pl.col("alphabet") == "hsdm17") & ~pl.col("identical")
        )["pair"].to_list()
    )
    shared = sdm12_pairs & hsdm17_pairs
    print(f"  sdm12 accepts {len(sdm12_pairs)} non-identical pairs, hsdm17 "
          f"{len(hsdm17_pairs)}, shared {len(shared)}: "
          f"{', '.join(sorted(shared)) or 'none'}")
    print(f"\nWrote {OUTDIR / '30_segment_merges_by_alphabet.csv'}")


if __name__ == "__main__":
    main()
