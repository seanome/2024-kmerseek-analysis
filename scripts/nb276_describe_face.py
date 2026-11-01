#!/usr/bin/env python3
"""What shape is CD47's SIRPalpha-binding face: one patch, or 21 separate places?

The face has been carried through this notebook as a count, "21 residues within
4.5 A of SIRPalpha", which says nothing about how they sit along the chain. This
prints them as runs of consecutive residues, with the sequence and the closest
contact in each, so the face can be read rather than counted.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_describe_face.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import OUTDIR  # noqa: E402

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
#: CD47's immunoglobulin ectodomain in UniProt Q08722 numbering, from the entry's
#: own Chain and Transmembrane features.
ECTODOMAIN = (19, 141)


def main() -> None:
    face = pl.read_csv(OUTDIR / "04_cd47_sirpalpha_face.csv").sort(
        "cd47_uniprot_position"
    )
    positions = face["cd47_uniprot_position"].to_list()
    residue = dict(zip(positions, face["residue"].to_list()))
    distance = dict(zip(positions, face["min_distance_angstrom"].to_list()))

    runs: list[list[int]] = []
    current = [positions[0]]
    for previous, position in zip(positions[:-1], positions[1:]):
        if position == previous + 1:
            current.append(position)
        else:
            runs.append(current)
            current = [position]
    runs.append(current)

    print(f"{len(positions)} CD47 residues lie within 4.5 A of SIRPalpha in PDB "
          f"2JJS, and they fall into {len(runs)} runs of consecutive residues.")
    print(f"UniProt Q08722 numbering; the ectodomain is "
          f"{ECTODOMAIN[0]}-{ECTODOMAIN[1]}.\n")
    rows = []
    for run in runs:
        sequence = "".join(THREE_TO_ONE[residue[p]] for p in run)
        closest = min(distance[p] for p in run)
        span = str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}"
        print(f"  {span:>9}  {len(run):>2} residue{'s' if len(run) > 1 else ' '}  "
              f"{sequence:<10} closest contact {closest:.2f} A")
        rows.append(
            {
                "span": span,
                "n_residues": len(run),
                "sequence": sequence,
                "closest_contact_angstrom": round(closest, 2),
            }
        )

    singletons = sum(1 for run in runs if len(run) == 1)
    longest = max(runs, key=len)
    first, last = positions[0], positions[-1]
    print(f"\nLongest run: {len(longest)} residues, "
          f"{longest[0]}-{longest[-1]}.")
    print(f"Single residues on their own: {singletons} of the {len(runs)} runs.")
    print(f"First to last: {first}-{last}, which is {last - first + 1} residues of "
          f"chain holding the {len(positions)} that touch.")
    print(f"All of it inside the ectodomain: "
          f"{all(ECTODOMAIN[0] <= p <= ECTODOMAIN[1] for p in positions)}.")
    print("\nSo it is one face made of several loops, not 21 separate sites: the "
          "\nN-terminal end plus loops at one end of the immunoglobulin domain, "
          "\nwhich is what the Hatherley 2008 paper describes as a convoluted "
          "\ninterface rather than a flat patch.")

    pl.DataFrame(rows).write_csv(OUTDIR / "39_cd47_face_runs.csv")
    print(f"\nWrote {OUTDIR / '39_cd47_face_runs.csv'}")


if __name__ == "__main__":
    main()
