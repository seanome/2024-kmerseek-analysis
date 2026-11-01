#!/usr/bin/env python3
"""Draw ten partitions of the 20 amino acids into two classes of 10, at random.

Experiment A (notebook 246) asks whether the 2-letter hydrophobic-polar alphabets win on
short Swiss-Prot features because of hydrophobicity or because any 2-letter alphabet
would. These partitions are the comparison: same class count, same k, residues grouped
without regard to chemistry.

A draw is rejected and drawn again when it is close to one of the six 2-letter H/P
alphabets in kmerseek's README table: when either of its classes differs from that
alphabet's hydrophobic set by fewer than 4 residues (residues in one set and not the
other). Either class, because which class is called 1 is arbitrary: a draw whose class 2
is nearly hydrophobic is the same partition as one whose class 1 is. A draw identical to
an earlier accepted one is also drawn again. Both counts go in the manifest header.

Writes, under --out-dir:
  random2_<01..10>.tsv    residue, class (1 or 2)
  encoded_hp_*.tsv, hp2.manifest.tsv
                          the six H/P alphabets as partitions, encoded the same way
                          (see write_hp)
  random2.manifest.tsv    one row per partition: name, class-1 and class-2 residues, the
                          mean Kyte-Doolittle hydropathy of each class, and the distance
                          to the nearest H/P alphabet; the seed and rejection counts are
                          in the '#' header lines.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 20260929
N_PARTITIONS = 10
MIN_DISTANCE = 4

#: Hydrophobic class of the six 2-letter H/P alphabets, from kmerseek's README table.
HP_HYDROPHOBIC = {
    "hp_lehninger2": "AFGILMPVWY",
    "hp_thomas_dill2": "ACFILMVWY",
    "hp_kyte_doolittle2": "ACFILMV",
    "hp_thomas_dill_no_c2": "AFILMVWY",
    "hp_lehninger_c_nonpolar2": "ACFGILMPVWY",
    "hp_pbotc_1st_ed2": "ACFILMPVWY",
}

#: Kyte, J. & Doolittle, R. F. (1982) J Mol Biol 157:105-132, Table 2.
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}  # fmt: skip


def distance_to_hp(class1: frozenset[str]) -> tuple[int, str]:
    """Smallest number of residues by which either class differs from an H/P hydrophobic set."""
    class2 = frozenset(AMINO_ACIDS) - class1
    best = min(
        (len(side ^ frozenset(h)), name)
        for name, h in HP_HYDROPHOBIC.items()
        for side in (class1, class2)
    )
    return best


def draw_partitions(
    seed: int = SEED,
    n: int = N_PARTITIONS,
    min_distance: int = MIN_DISTANCE,
    max_draws: int = 100_000,
) -> tuple[list[frozenset[str]], int, int]:
    """Ten accepted class-1 sets, and how many draws were rejected as near-H/P or repeats.

    At the default distance of 4 about 1% of draws are rejected (measured over 20_000).
    No partition is ever more than 9 residues from all six H/P sets, so a distance above
    9 can never be met; max_draws stops the loop instead of running forever.
    """
    rng = random.Random(seed)
    accepted: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    n_near_hp = n_repeat = 0
    while len(accepted) < n:
        if n_near_hp + n_repeat + len(accepted) >= max_draws:
            raise RuntimeError(
                f"{max_draws} draws gave only {len(accepted)} partitions at distance >= {min_distance}"
            )
        residues = list(AMINO_ACIDS)
        rng.shuffle(residues)
        class1 = frozenset(residues[:10])
        if distance_to_hp(class1)[0] < min_distance:
            n_near_hp += 1
            continue
        # The same partition with the classes swapped counts as a repeat.
        key = min(class1, frozenset(AMINO_ACIDS) - class1, key=sorted)
        if key in seen:
            n_repeat += 1
            continue
        seen.add(key)
        accepted.append(class1)
    return accepted, n_near_hp, n_repeat


def kd_mean(residues) -> float:
    return sum(KYTE_DOOLITTLE[r] for r in residues) / len(residues)


def write(out_dir: Path, seed: int = SEED) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, n_near_hp, n_repeat = draw_partitions(seed)
    rows = []
    for i, class1 in enumerate(parts, start=1):
        name = f"random2_{i:02d}"
        class2 = frozenset(AMINO_ACIDS) - class1
        with open(out_dir / f"{name}.tsv", "w") as fh:
            fh.write("residue\tclass\n")
            for r in AMINO_ACIDS:
                fh.write(f"{r}\t{1 if r in class1 else 2}\n")
        dist, nearest = distance_to_hp(class1)
        rows.append(
            [
                name,
                f"{name}.tsv",
                "".join(sorted(class1)),
                "".join(sorted(class2)),
                f"{kd_mean(class1):.2f}",
                f"{kd_mean(class2):.2f}",
                str(dist),
                nearest,
            ]
        )
    manifest = out_dir / "random2.manifest.tsv"
    with open(manifest, "w") as fh:
        fh.write(f"# seed\t{seed}\n")
        fh.write(
            f"# draws rejected, within {MIN_DISTANCE - 1} residues of an H/P alphabet\t{n_near_hp}\n"
        )
        fh.write(f"# draws rejected, repeating an accepted partition\t{n_repeat}\n")
        h_means = [kd_mean(h) for h in HP_HYDROPHOBIC.values()]
        p_means = [kd_mean(set(AMINO_ACIDS) - set(h)) for h in HP_HYDROPHOBIC.values()]
        fh.write(
            "# kd_mean: mean Kyte-Doolittle hydropathy of the class. For comparison, the six "
            f"H/P alphabets have {min(h_means):+.2f} to {max(h_means):+.2f} in the hydrophobic "
            f"class and {min(p_means):+.2f} to {max(p_means):+.2f} in the polar one\n"
        )
        fh.write(
            "name\tpartition_tsv\tclass1\tclass2\tkd_mean_class1\tkd_mean_class2\t"
            "residues_from_nearest_hp\tnearest_hp\n"
        )
        for r in rows:
            fh.write("\t".join(r) + "\n")
    return manifest


def write_hp(out_dir: Path) -> Path:
    """The six H/P alphabets as partitions, so they run through the same A/D encoding.

    Encoded like the random partitions, the H/P arms get their E-value fit the same way:
    kmerseek's fit shuffles the sequence it is given, which is amino acids for a built-in
    alphabet and A/D for an encoded one, and on the notebook 246 positive control the
    A/D-shuffled fit let 11-13% more rows through E <= 0.01. Positions, n_shared and mean
    IDF are identical either way. Class 1 is the hydrophobic class. Named
    encoded_<alphabet>, never hp_*, which names kmerseek's own alphabets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for alphabet, h in HP_HYDROPHOBIC.items():
        name = f"encoded_{alphabet}"
        with open(out_dir / f"{name}.tsv", "w") as fh:
            fh.write("residue\tclass\n")
            for r in AMINO_ACIDS:
                fh.write(f"{r}\t{1 if r in h else 2}\n")
        rows.append(f"{name}\t{name}.tsv\tprotein20\n")
    manifest = out_dir / "hp2.manifest.tsv"
    with open(manifest, "w") as fh:
        fh.write(
            "# the six 2-letter H/P alphabets in kmerseek's README table, encoded as A/D "
            "(hydrophobic -> A) and run as protein20, like the random partitions\n"
        )
        fh.write("name\tpartition_tsv\talphabet\n")
        fh.writelines(rows)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=Path("data/random_alphabets"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    manifest = write(args.out_dir, args.seed)
    write_hp(args.out_dir)
    print(manifest.read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
