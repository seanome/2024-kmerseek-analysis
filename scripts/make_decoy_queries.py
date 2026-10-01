#!/usr/bin/env python3
"""Stage a QfO tree whose human proteome is dipeptide-shuffled decoys of the real one.

Notebook 270 counts how many calls kmerseek makes on queries that have no homolog, at
each --scaled. The decoys come from the mini set's 200 human queries: one shuffled copy
of each, keeping every adjacent residue pair (Altschul & Erickson 1985). The target
proteomes are symlinked from the real tree, so the decoy run searches the same targets.

Each decoy keeps its source's FASTA header with the accession prefixed DECOY_, so no
decoy joins to the Swiss-Prot or Pfam truth of the protein it came from. The seed is
fixed and the script refuses to overwrite a different decoy file, so every scaled value
is searched with the same decoys. The md5 of the decoy FASTA is written next to it.

    python3 scripts/make_decoy_queries.py \
        --qfo-dir nextflow-runs/qfo-pfam-region-benchmark/data/mini/qfo \
        --out-dir nextflow-runs/qfo-pfam-region-benchmark/data/mini-decoy/qfo
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HUMAN = "Eukaryota/UP000005640_9606.fasta"
SEED = 270


def dipeptide_shuffle(seq: str, rng: random.Random) -> str:
    """A random sequence with the same first residue, last residue and count of every
    adjacent residue pair as `seq` (Altschul & Erickson 1985, Mol Biol Evol 2:526).

    Copied from notebooks/256_bhf_dipeptide_shuffle_null.py on
    olgabot/bhf-dipeptide-shuffle-null, unchanged, so notebook 256 and 270 use one null.
    """
    if len(seq) < 3:
        return seq
    out = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        out[a].append(b)
    last = seq[-1]
    while True:
        final = {v: rng.choice(ends) for v, ends in out.items() if v != last}
        ok = True
        for v in final:
            seen, u = set(), v
            while u != last:
                if u in seen:
                    ok = False
                    break
                seen.add(u)
                u = final[u]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, ends in out.items():
        rest = list(ends)
        if v in final:
            rest.remove(final[v])
        rng.shuffle(rest)
        order[v] = rest + ([final[v]] if v in final else [])
    walk, u = [seq[0]], seq[0]
    for _ in range(len(seq) - 1):
        u = order[u].pop(0)
        walk.append(u)
    return "".join(walk)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records, header, seq = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header, seq = line[1:], []
        elif line.strip():
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def decoy_header(header: str) -> str:
    # sp|P12345|NAME_HUMAN ... -> sp|DECOY_P12345|NAME_HUMAN ...
    parts = header.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"not a UniProt header: {header[:60]}")
    return f"{parts[0]}|DECOY_{parts[1]}|{parts[2]}"


def pair_counts(seq: str) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for a, b in zip(seq, seq[1:]):
        c[a + b] += 1
    return c


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--qfo-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    rng = random.Random(SEED)
    lines = []
    for header, seq in read_fasta(args.qfo_dir / HUMAN):
        shuffled = dipeptide_shuffle(seq, rng)
        assert pair_counts(shuffled) == pair_counts(seq), header
        lines.append(f">{decoy_header(header)}\n{shuffled}\n")
    text = "".join(lines)
    md5 = hashlib.md5(text.encode()).hexdigest()

    dest = args.out_dir / HUMAN
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() != md5:
        print(f"{dest} exists and differs from what seed {SEED} gives; not overwriting",
              file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    (args.out_dir / "decoy_queries.md5").write_text(f"{md5}  {HUMAN}\n")

    # Targets: every other proteome in the real tree, linked rather than copied.
    n_links = 0
    for src in sorted(args.qfo_dir.rglob("*.fasta")):
        rel = src.relative_to(args.qfo_dir)
        if str(rel) == HUMAN:
            continue
        link = args.out_dir / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(src.resolve(), link)
        n_links += 1
    print(f"{len(lines)} decoy queries -> {dest} (md5 {md5}); {n_links} targets linked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
