#!/usr/bin/env python3
"""Notebook 301, step 5: the query subset the kmerseek run searches (Olga, 2026-10-21).

Standard library only. From the step 2 query set, draw with seed 301:
  150 human and 150 microbial real queries,
  30 chimeras on a human host and 30 on a microbial host,
and write them, followed by every usable decoy of their features (step 3, `unchanged` is
False), to --out-fasta. --out-ids lists the chosen query ids.

The full query set (2,000 real + 500 chimeras + 9,097 decoys) needs about 1,450 job-hours
for every setting with a Karlin-Altschul fit, measured in the pilot at about 6.5 s per
sequence. This subset with 19 settings needs about 55.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SEED = 301
N_REAL_PER_GROUP = 150
N_CHIMERA_PER_GROUP = 30


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


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step2", type=Path, required=True)
    ap.add_argument("--step3", type=Path, required=True)
    ap.add_argument("--out-fasta", type=Path, required=True)
    ap.add_argument("--out-ids", type=Path, required=True)
    a = ap.parse_args()
    rng = random.Random(SEED)
    truth = read_tsv(a.step2 / "truth.tsv")
    group = {r["query_id"]: (r["query_kind"], r["organism_group"]) for r in truth}
    chosen = []
    for kind, n in (("real", N_REAL_PER_GROUP), ("chimera", N_CHIMERA_PER_GROUP)):
        for g in ("human", "microbial"):
            pool = sorted(q for q, kg in group.items() if kg == (kind, g))
            chosen += sorted(rng.sample(pool, n))
    chosen_set = set(chosen)
    queries = read_fasta(a.step2 / "queries.fasta")
    decoy_seqs = read_fasta(a.step3 / "decoys.fasta")
    decoys = [
        d
        for d in read_tsv(a.step3 / "decoys.tsv")
        if d["query_id"] in chosen_set and d["unchanged"] == "False"
    ]
    with open(a.out_fasta, "w") as fh:
        for q in chosen:
            fh.write(f">{q}\n{queries[q]}\n")
        for d in decoys:
            fh.write(f">{d['decoy_id']}\n{decoy_seqs[d['decoy_id']]}\n")
    a.out_ids.write_text("\n".join(chosen) + "\n")
    n_feat = sum(r["query_id"] in chosen_set for r in truth)
    print(f"queries {len(chosen)}, features {n_feat}, decoys {len(decoys)}")
    print(f"sequences to search {len(chosen) + len(decoys)}")


if __name__ == "__main__":
    main()
