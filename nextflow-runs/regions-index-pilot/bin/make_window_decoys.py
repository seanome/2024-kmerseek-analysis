#!/usr/bin/env python3
"""A decoy copy of an index: each entry's residues shuffled within 10-residue windows.

The sequence is split into non-overlapping windows of --window residues from its first
residue (the last window may be shorter) and the residues inside each window are shuffled.
Composition is kept window by window, so a decoy has the same local amino-acid make-up as
its source (a transmembrane helix stays hydrophobic, a low-complexity stretch stays
low-complexity) but no residue order beyond 10 positions. A call on a decoy is a call no
homology can explain.

Each decoy is named DECOY_<source name>. The shuffle for an entry is seeded by --seed and
the entry's name, so the decoy of an entry does not depend on file order. The script writes
the decoys alone and the target + decoy file that is searched, and refuses to overwrite a
decoy file with different content, so every search reads the same decoys. Same conventions
as scripts/make_decoy_queries.py on olgabot/midi-plus-0.4-decoy (DECOY_ prefix, fixed seed,
md5 written next to the file); that script is a dipeptide shuffle of queries, so it is not
reused here.
"""

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path


def read_fasta(path: Path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].split()[0], []
            elif line:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def window_shuffle(seq: str, rng: random.Random, window: int) -> str:
    out = []
    for i in range(0, len(seq), window):
        w = list(seq[i:i + window])
        rng.shuffle(w)
        out.extend(w)
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--decoy-out", type=Path, required=True)
    ap.add_argument("--combined-out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    lines_t, lines_d = [], []
    n = same = 0
    for name, seq in read_fasta(args.fasta):
        if name.startswith("DECOY_"):
            sys.exit(f"input already holds a decoy: {name}")
        h = hashlib.sha256(f"{args.seed}:{name}".encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        dec = window_shuffle(seq, rng, args.window)
        for i in range(0, len(seq), args.window):
            if Counter(seq[i:i + args.window]) != Counter(dec[i:i + args.window]):
                sys.exit(f"window composition changed in {name} at {i}")
        same += dec == seq
        lines_t.append(f">{name}\n{seq}\n")
        lines_d.append(f">DECOY_{name}\n{dec}\n")
        n += 1

    decoy_text = "".join(lines_d)
    md5 = hashlib.md5(decoy_text.encode()).hexdigest()
    if args.decoy_out.exists() and \
            hashlib.md5(args.decoy_out.read_bytes()).hexdigest() != md5:
        sys.exit(f"{args.decoy_out} exists with different content; refusing to overwrite")
    args.decoy_out.write_text(decoy_text)
    Path(str(args.decoy_out) + ".md5").write_text(f"{md5}  {args.decoy_out.name}\n")
    args.combined_out.write_text("".join(lines_t) + decoy_text)
    print(f"[decoys] entries={n} window={args.window} seed={args.seed} "
          f"decoys_identical_to_source={same} md5={md5}", file=sys.stderr)


if __name__ == "__main__":
    main()
