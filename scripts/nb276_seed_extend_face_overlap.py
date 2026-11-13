#!/usr/bin/env python3
"""Do the seed-and-extend segments land on CD47's SIRPalpha face, or elsewhere?

A segment that covers the P66 loop is only interesting if the CD47 side of it sits
on the face CD47 uses on SIRPalpha. A segment aligning the loop onto CD47's
transmembrane or cytoplasmic region says nothing about the mimicry claim, however
good its score.

Reads the sweep (11_seed_extend_loop.csv) and the measured face
(04_cd47_sirpalpha_face.csv) and writes 13_seed_extend_face_overlap.csv.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_seed_extend_face_overlap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import OUTDIR  # noqa: E402

#: CD47's ectodomain in UniProt Q08722 numbering: the mature chain starts at 19 and
#: the immunoglobulin domain ends at 141, where the first transmembrane helix
#: begins. Taken from the entry's own Chain and Transmembrane features.
ECTODOMAIN = (19, 141)


def main() -> None:
    face_positions = set(
        pl.read_csv(OUTDIR / "04_cd47_sirpalpha_face.csv")["cd47_uniprot_position"]
        .to_list()
    )
    sweep = pl.read_csv(OUTDIR / "11_seed_extend_loop.csv").filter(
        pl.col("reaches_loop")
    )

    rows = []
    for row in sweep.iter_rows(named=True):
        start, end = (int(x) for x in row["cd47_span"].split("-"))
        span = set(range(start, end + 1))
        touched = sorted(face_positions & span)
        inside = len(span & set(range(ECTODOMAIN[0], ECTODOMAIN[1] + 1)))
        rows.append(
            {
                "alphabet": row["alphabet"],
                "seed_length": row["seed_length"],
                "penalty": row["penalty"],
                "score": row["score"],
                "p_value_vs_shuffled_cd47": row["p_value_vs_shuffled_cd47"],
                "cd47_span": row["cd47_span"],
                "segment_length": end - start + 1,
                "fraction_in_ectodomain": round(inside / (end - start + 1), 3),
                "n_face_residues_touched": len(touched),
                "face_residues_touched": ",".join(str(p) for p in touched) or "none",
                "covers_both_aspartates": row["covers_both_aspartates"],
            }
        )

    table = pl.DataFrame(rows).sort(
        "n_face_residues_touched", "score", descending=[True, True]
    )
    table.write_csv(OUTDIR / "13_seed_extend_face_overlap.csv")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    print(f"CD47 ectodomain taken as UniProt {ECTODOMAIN[0]}-{ECTODOMAIN[1]}; "
          f"{len(face_positions)} face residues.\n")
    print(table)
    print(f"\nsegments touching at least one face residue: "
          f"{(table['n_face_residues_touched'] > 0).sum()} of {table.height}")
    print(f"segments entirely outside the ectodomain: "
          f"{(table['fraction_in_ectodomain'] == 0).sum()}")
    print(f"\nWrote {OUTDIR / '13_seed_extend_face_overlap.csv'}")


if __name__ == "__main__":
    main()
