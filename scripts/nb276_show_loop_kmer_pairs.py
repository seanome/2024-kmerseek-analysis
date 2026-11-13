#!/usr/bin/env python3
"""Print the actual k-mer pairs P66 and CD47 share on the claimed loop.

Step 9 of scripts/nb276_p66_cd47_structure.py counts them. A count is not enough
to judge a match: this prints the two spans residue by residue with the class
string they encode to, so a match can be read rather than trusted.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_show_loop_kmer_pairs.py
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

#: Printable class letters. Only alphabets up to 20 classes are used here.
CLASS_LETTERS = "ABCDEFGHIJKLMNOPQRST"


def encode(sequence: str, clusters: list[str]) -> list[int]:
    table = {residue: i for i, cl in enumerate(clusters) for residue in cl}
    return [table.get(residue, -1) for residue in sequence]


def loop_kmer_pairs(
    p66_seq: str,
    cd47_seq: str,
    loop: tuple[int, int],
    alphabets: dict[str, list[str]],
    ksizes: range,
) -> pl.DataFrame:
    """One row per shared k-mer that covers the loop, with both spans written out."""
    rows = []
    for name, clusters in alphabets.items():
        p66_code = encode(p66_seq, clusters)
        cd47_code = encode(cd47_seq, clusters)
        for k in ksizes:
            cd47_positions: dict[tuple, list[int]] = {}
            for i in range(len(cd47_code) - k + 1):
                cd47_positions.setdefault(tuple(cd47_code[i : i + k]), []).append(i + 1)
            first = max(0, loop[0] - k)
            last = min(len(p66_code) - k, loop[1] - 1)
            for i in range(first, last + 1):
                key = tuple(p66_code[i : i + k])
                for cd47_start in cd47_positions.get(key, []):
                    p66_start, p66_end = i + 1, i + k
                    overlap = sorted(
                        set(range(p66_start, p66_end + 1))
                        & set(range(loop[0], loop[1] + 1))
                    )
                    rows.append(
                        {
                            "alphabet": name,
                            "n_classes": len(clusters),
                            "ksize": k,
                            "p66_start": p66_start,
                            "p66_end": p66_end,
                            "p66_residues": p66_seq[p66_start - 1 : p66_end],
                            "cd47_start": cd47_start,
                            "cd47_end": cd47_start + k - 1,
                            "cd47_residues": cd47_seq[cd47_start - 1 : cd47_start + k - 1],
                            "class_string": "".join(CLASS_LETTERS[c] for c in key),
                            "n_loop_residues_covered": len(overlap),
                            "loop_residues_covered": ",".join(str(p) for p in overlap),
                            "n_identical_residues": sum(
                                a == b
                                for a, b in zip(
                                    p66_seq[p66_start - 1 : p66_end],
                                    cd47_seq[cd47_start - 1 : cd47_start + k - 1],
                                )
                            ),
                        }
                    )
    return pl.DataFrame(rows)


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)

    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    pairs = loop_kmer_pairs(
        p66["sequence"], cd47["sequence"], loop, alphabets, range(5, 26)
    )
    pairs.write_csv(OUTDIR / "10_loop_kmer_pairs.csv")

    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(200)
    print(f"Loop is P66 {loop[0]}-{loop[1]} = {p66['sequence'][loop[0] - 1:loop[1]]}\n")
    if pairs.is_empty():
        print("No shared k-mer covers the loop in any alphabet at any k from 5 to 25.")
        return
    print(f"{pairs.height} shared k-mers cover the loop, in "
          f"{pairs['alphabet'].n_unique()} alphabets:\n")
    for name in sorted(pairs["alphabet"].unique().to_list()):
        sub = pairs.filter(pl.col("alphabet") == name).sort("ksize", "p66_start")
        print(f"=== {name} ({sub['n_classes'][0]} classes): {sub.height} k-mers, "
              f"k {sub['ksize'].min()}-{sub['ksize'].max()} ===")
        for row in sub.iter_rows(named=True):
            print(f"  k={row['ksize']}  covers {row['n_loop_residues_covered']} loop "
                  f"residue(s): {row['loop_residues_covered']}")
            print(f"    P66  {row['p66_start']:>4}-{row['p66_end']:<4} {row['p66_residues']}")
            print(f"    CD47 {row['cd47_start']:>4}-{row['cd47_end']:<4} {row['cd47_residues']}")
            print(f"    class{'':<9} {row['class_string']}   identical residues: "
                  f"{row['n_identical_residues']}/{row['ksize']}")
        print()
    print(f"Wrote {OUTDIR / '10_loop_kmer_pairs.csv'}")


if __name__ == "__main__":
    main()
