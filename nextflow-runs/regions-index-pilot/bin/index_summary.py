#!/usr/bin/env python3
"""Residues in each index, and the score a hit needs to reach E = 1 against it.

For a query of m residues against an index of n residues, a hit reaches E-value E when its
score is at least log2(K * m * n / E) bits (Karlin and Altschul 1990). This prints
log2(K * m * n) (E = 1) for the median query length, for each index, counting n two ways:
the target entries alone, and target plus decoys (what is searched).

K = 0.0180 is BORROWED, not fitted on these indexes: it is the kmerseek docs/evalue.md value
for a 15_000-sequence Swiss-Prot sample at hp_thomas_dill2 k = 12, C = 2, X = 8. The
difference between the two indexes does not depend on K: it is log2(n_whole / n_regions).

The last columns divide the bits by a per-position information value supplied with the
pilot's design (0.157 bits per aligned position, the H/P alphabet at 20-30% identity). It
is an assumed value, not measured here, and the column is named for that.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

K_BORROWED = 0.0180
BITS_PER_POSITION_ASSUMED = 0.157


def fasta_stats(path: Path) -> tuple[int, int, int]:
    """(entries, residues in target entries, residues in DECOY_ entries)."""
    n = res_t = res_d = 0
    decoy = False
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
                decoy = line.startswith(">DECOY_")
            elif decoy:
                res_d += len(line.strip())
            else:
                res_t += len(line.strip())
    return n, res_t, res_d


def lengths(path: Path) -> list[int]:
    out, cur = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    out.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
    if cur is not None:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--index", nargs=2, action="append", metavar=("NAME", "TARGET_DECOY_FASTA"),
                    required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = statistics.median(lengths(args.queries))
    rows = []
    for name, path in args.index:
        n_entries, res_t, res_d = fasta_stats(Path(path))
        for label, n in [("target", res_t), ("target_plus_decoy", res_t + res_d)]:
            bits = math.log2(K_BORROWED * m * n)
            rows.append({
                "index": name, "residues_counted": label, "entries": n_entries,
                "residues": n, "median_query_aa": m, "K_borrowed": K_BORROWED,
                "bits_needed_E1": round(bits, 2),
                "positions_needed_at_0157_assumed": round(bits / BITS_PER_POSITION_ASSUMED),
            })
    by = {(r["index"], r["residues_counted"]): r for r in rows}
    names = [n for n, _ in args.index]
    if len(names) == 2:
        a, b = names
        for label in ("target", "target_plus_decoy"):
            d = by[(a, label)]["bits_needed_E1"] - by[(b, label)]["bits_needed_E1"]
            ratio = by[(a, label)]["residues"] / by[(b, label)]["residues"]
            print(f"[bits] {label}: {a} needs {d:.2f} more bits than {b} "
                  f"(residue ratio {ratio:.2f}, log2 = {math.log2(ratio):.2f})", file=sys.stderr)
    args.out.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print("[bits] " + "  ".join(f"{k}={v}" for k, v in r.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
