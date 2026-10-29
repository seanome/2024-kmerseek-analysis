#!/usr/bin/env python3
"""Notebook 301, step 3: two decoys for every feature window of every query.

Standard library only (runs on Sherlock with `module load python/3.12.1`).

For each feature in truth.tsv (step 2), two copies of its query protein in which only the
feature's residues are replaced:
  dishuffle  the window cut into adjacent pairs (1-2, 3-4, ...), the pairs put in a random
             order; an odd last residue stays last. Keeps the residue and H/P composition,
             destroys the order. Same recipe as step 1 (scripts/score_301_step1_scans.py).
  reversed   the window read backwards.
Everything outside the window, including the protein's other features, is unchanged. A
hit on the replaced residues counts as a false positive.

A decoy identical to the real window is not a decoy: a window of 3 residues or fewer has at
most one pair to move, and a palindrome reads the same reversed. The shuffle is redrawn up
to MAX_REDRAWS times; a decoy still identical after that is written with `unchanged=True`
and left out of every decoy count.

Outputs in --out
  decoys.fasta   one protein per (feature, kind); id <decoy_id>
  decoys.tsv     decoy_id, query_id, feature_row (0-based row of truth.tsv), kind, start,
                 end (1-based, inclusive, the replaced residues), real_residues,
                 decoy_residues, unchanged
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SEED = 301
MAX_REDRAWS = 20


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def di_residue_shuffle(window: str, rng: random.Random) -> str:
    pairs = [window[i : i + 2] for i in range(0, len(window) - 1, 2)]
    tail = window[-1] if len(window) % 2 else ""
    order = list(range(len(pairs)))
    rng.shuffle(order)
    return "".join(pairs[i] for i in order) + tail


def decoy_window(window: str, kind: str, rng: random.Random) -> str:
    if kind == "reversed":
        return window[::-1]
    new = di_residue_shuffle(window, rng)
    for _ in range(MAX_REDRAWS):
        if new != window:
            break
        new = di_residue_shuffle(window, rng)
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    seqs = read_fasta(a.queries)
    with open(a.truth) as fh:
        truth = list(csv.DictReader(fh, delimiter="\t"))

    rows = []
    with open(a.out / "decoys.fasta", "w") as fa:
        for i, r in enumerate(truth):
            p = seqs[r["query_id"]]
            s, e = int(r["start"]), int(r["end"])
            w = p[s - 1 : e]
            assert w == r["residues"], (i, r["query_id"])
            for kind in ("dishuffle", "reversed"):
                new = decoy_window(w, kind, rng)
                assert sorted(new) == sorted(w) and len(new) == len(w)
                d = p[: s - 1] + new + p[e:]
                assert len(d) == len(p) and d[s - 1 : e] == new
                did = f"decoy_{i:05d}_{kind}"
                fa.write(f">{did}\n")
                for j in range(0, len(d), 60):
                    fa.write(d[j : j + 60] + "\n")
                rows.append(
                    {
                        "decoy_id": did,
                        "query_id": r["query_id"],
                        "feature_row": i,
                        "feature_type": r["feature_type"],
                        "kind": kind,
                        "start": s,
                        "end": e,
                        "real_residues": w,
                        "decoy_residues": new,
                        "unchanged": new == w,
                    }
                )
    with open(a.out / "decoys.tsv", "w", newline="") as fh:
        wr = csv.DictWriter(
            fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        wr.writeheader()
        wr.writerows(rows)
    n_unchanged = sum(r["unchanged"] for r in rows)
    print(f"features\t{len(truth)}")
    print(f"decoys\t{len(rows)}")
    for kind in ("dishuffle", "reversed"):
        k = [r for r in rows if r["kind"] == kind]
        print(f"{kind} unchanged (left out)\t{sum(r['unchanged'] for r in k)}")
    print(f"decoys used\t{len(rows) - n_unchanged}")


if __name__ == "__main__":
    main()
