#!/usr/bin/env python3
"""Print the P66 197-232 / CD47 109-144 alignment with every annotation on it.

The segment has been described in prose over and over in this notebook. This
prints it, with one track per thing that has been claimed about it, so each claim
can be checked against the residues instead of taken on trust:

  the loop the claim names, under the P66 row
  which columns are identical
  which columns hsdm17 calls a match, and which of those are the K-against-E pair
  which CD47 residues touch SIRPalpha, over the CD47 row
  where CD47's ectodomain ends

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_print_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    resolve_uniprot,
)
from nb276_why_combining_hurts import CD47_SEGMENT, P66_SEGMENT  # noqa: E402

ALPHABET = "hsdm17"
#: CD47's ectodomain, from the UniProt entry's Chain and Transmembrane features.
ECTODOMAIN_END = 141


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)
    face = set(
        pl.read_csv(OUTDIR / "04_cd47_sirpalpha_face.csv")[
            "cd47_uniprot_position"
        ].to_list()
    )

    p66_part = p66["sequence"][P66_SEGMENT[0] - 1 : P66_SEGMENT[1]]
    cd47_part = cd47["sequence"][CD47_SEGMENT[0] - 1 : CD47_SEGMENT[1]]
    n = len(p66_part)

    clusters = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}[ALPHABET]
    table = {r: i for i, cl in enumerate(clusters) for r in cl}

    face_track, ecto_track = [], []
    identity, class_track, loop_track, ruler = [], [], [], []
    for i in range(n):
        p66_pos, cd47_pos = P66_SEGMENT[0] + i, CD47_SEGMENT[0] + i
        a, b = p66_part[i], cd47_part[i]
        face_track.append("*" if cd47_pos in face else " ")
        ecto_track.append(" " if cd47_pos <= ECTODOMAIN_END else "T")
        identity.append("|" if a == b else " ")
        same_class = table.get(a, -1) == table.get(b, -2)
        class_track.append(
            "K" if same_class and a != b and {a, b} == {"K", "E"}
            else ":" if same_class and a != b
            else " "
        )
        loop_track.append("^" if loop[0] <= p66_pos <= loop[1] else " ")
        ruler.append("|" if p66_pos % 10 == 0 else " ")

    pad = 14
    def row(label, content):
        print(f"{label:>{pad}}  {content}")

    print(f"P66 {P66_SEGMENT[0]}-{P66_SEGMENT[1]} against CD47 "
          f"{CD47_SEGMENT[0]}-{CD47_SEGMENT[1]}, the segment seed-and-extend found "
          f"in {ALPHABET}.\n")
    row("P66 every 10", "".join(ruler))
    row(f"P66 {P66_SEGMENT[0]}", p66_part)
    row("the loop", "".join(loop_track))
    row("identical", "".join(identity))
    row(f"{ALPHABET} also", "".join(class_track))
    row(f"CD47 {CD47_SEGMENT[0]}", cd47_part)
    row("touches SIRPa", "".join(face_track))
    row("past ectodomain", "".join(ecto_track))

    n_identical = sum(1 for c in identity if c == "|")
    n_class = sum(1 for c in class_track if c != " ")
    n_ke = sum(1 for c in class_track if c == "K")
    n_face = sum(1 for c in face_track if c == "*")
    n_ecto = sum(1 for c in ecto_track if c == " ")
    loop_face = sum(
        1
        for i in range(n)
        if loop[0] <= P66_SEGMENT[0] + i <= loop[1]
        and CD47_SEGMENT[0] + i in face
    )
    print()
    print(f"  ^  the loop in the claim, P66 {loop[0]}-{loop[1]}: "
          f"{p66_part[loop[0] - P66_SEGMENT[0]:loop[1] - P66_SEGMENT[0] + 1]}")
    print(f"  |  identical residue: {n_identical} of {n}")
    print(f"  :  not identical but the same {ALPHABET} class: {n_class}, "
          f"of which {n_ke} are a lysine against a glutamate (marked K)")
    print(f"  *  CD47 residue within 4.5 A of SIRPalpha: {n_face} of the 21 in the "
          f"whole face")
    print(f"  T  past the end of CD47's ectodomain at {ECTODOMAIN_END}: "
          f"{n - n_ecto} columns, so {100 * n_ecto / n:.0f}% of the segment is "
          f"inside it")
    print(f"\n  Opposite the loop sits CD47 "
          f"{CD47_SEGMENT[0] + loop[0] - P66_SEGMENT[0]}-"
          f"{CD47_SEGMENT[0] + loop[1] - P66_SEGMENT[0]}: "
          f"{cd47_part[loop[0] - P66_SEGMENT[0]:loop[1] - P66_SEGMENT[0] + 1]}, "
          f"{loop_face} of those 7 touching SIRPalpha.")
    print(f"  The loop's own columns contribute {sum(1 for i in range(n) if loop[0] <= P66_SEGMENT[0] + i <= loop[1] and identity[i] == '|')} "
          f"identical residues and "
          f"{sum(1 for i in range(n) if loop[0] <= P66_SEGMENT[0] + i <= loop[1] and class_track[i] != ' ')} "
          f"same-class ones, out of its 7.")


if __name__ == "__main__":
    main()
