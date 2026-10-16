#!/usr/bin/env python3
"""Does the loop earn the seed-and-extend segment's score, or do the flanks?

A segment can cover the loop and owe none of its score to it. The claim is about
the loop, so a segment whose score comes from the residues beside it is not
evidence for the claim however good the segment is.

For the best candidate from scripts/nb276_seed_extend_followup.py this splits the
score column by column and reports what the loop's own columns contribute, with
the rest of the segment as the comparison.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_seed_extend_loop_contribution.py
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
    resolve_uniprot,
)
from nb276_seed_extend_loop import encode  # noqa: E402


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)

    follow = pl.read_csv(OUTDIR / "12_seed_extend_followup.csv").sort(
        "p_composition_null"
    )
    sweep = pl.read_csv(OUTDIR / "11_seed_extend_loop.csv").filter(
        pl.col("reaches_loop")
    )
    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    face = set(
        pl.read_csv(OUTDIR / "04_cd47_sirpalpha_face.csv")["cd47_uniprot_position"]
        .to_list()
    )

    rows = []
    for candidate in follow.iter_rows(named=True):
        row = sweep.filter(
            (pl.col("alphabet") == candidate["alphabet"])
            & (pl.col("seed_length") == candidate["seed_length"])
            & (pl.col("penalty") == candidate["penalty"])
        ).row(0, named=True)
        clusters = alphabets[candidate["alphabet"]]
        query = encode(p66["sequence"], clusters)
        target = encode(cd47["sequence"], clusters)
        p66_start = int(row["p66_span"].split("-")[0])
        cd47_start = int(row["cd47_span"].split("-")[0])
        length = len(row["p66_residues"])
        penalty = candidate["penalty"]

        loop_score = other_score = 0.0
        loop_cols = other_cols = 0
        for i in range(length):
            qi, ti = p66_start - 1 + i, cd47_start - 1 + i
            if not (0 <= qi < len(query) and 0 <= ti < len(target)):
                continue
            column = 1.0 if query[qi] == target[ti] and query[qi] >= 0 else -penalty
            if loop[0] <= p66_start + i <= loop[1]:
                loop_score += column
                loop_cols += 1
            else:
                other_score += column
                other_cols += 1
        rows.append(
            {
                "alphabet": candidate["alphabet"],
                "seed_length": candidate["seed_length"],
                "penalty": penalty,
                "total_score": candidate["score"],
                "segment_length": length,
                "loop_columns": loop_cols,
                "loop_score": round(loop_score, 2),
                "other_columns": other_cols,
                "other_score": round(other_score, 2),
                "loop_score_per_column": round(loop_score / loop_cols, 3)
                if loop_cols
                else None,
                "other_score_per_column": round(other_score / other_cols, 3)
                if other_cols
                else None,
                "loop_share_of_score": round(loop_score / candidate["score"], 3)
                if candidate["score"]
                else None,
                "p66_loop_residues": p66["sequence"][loop[0] - 1 : loop[1]],
                "cd47_residues_opposite_loop": cd47["sequence"][
                    cd47_start - 1 + (loop[0] - p66_start) :
                    cd47_start - 1 + (loop[1] - p66_start) + 1
                ],
                "n_face_residues_opposite_loop": sum(
                    1
                    for i in range(loop[0] - p66_start, loop[1] - p66_start + 1)
                    if cd47_start + i in face
                ),
            }
        )

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "14_seed_extend_loop_contribution.csv")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    print("Where each candidate segment's score comes from:\n")
    print(table.select(
        "alphabet", "seed_length", "total_score", "segment_length",
        "loop_columns", "loop_score", "loop_score_per_column",
        "other_score_per_column", "loop_share_of_score",
    ))
    print("\nWhat sits opposite the loop:\n")
    print(table.select(
        "alphabet", "p66_loop_residues", "cd47_residues_opposite_loop",
        "n_face_residues_opposite_loop",
    ))
    worse = table.filter(
        pl.col("loop_score_per_column") < pl.col("other_score_per_column")
    )
    print(f"\nCandidates where the loop scores worse per column than the rest of "
          f"its own segment: {worse.height} of {table.height}")
    print(f"\nWrote {OUTDIR / '14_seed_extend_loop_contribution.csv'}")


if __name__ == "__main__":
    main()
