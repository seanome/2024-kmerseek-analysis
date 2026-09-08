#!/usr/bin/env python3
"""Split the query proteome into fixed-size chunks, one SLURM task each.

45_339 proteins against 572_700 reference sequences does not fit one walltime for any arm,
and a scheduler kill reports exitStatus Integer.MAX_VALUE rather than a signal code, so the
run would retry into the same wall rather than failing usefully.

Content-stable: a chunk is only rewritten when its bytes would change. An identical rewrite
still moves the mtime, and every task reading that chunk would re-run on the next -resume
for no reason.
"""

import argparse
from pathlib import Path


def read_fasta(path: Path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, seq
                name, seq = line.rstrip("\n"), []
            else:
                seq.append(line.rstrip("\n"))
    if name is not None:
        yield name, seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=2000)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = kept = idx = 0
    buf = []

    def flush(records, i):
        nonlocal written, kept
        body = "".join(f"{n}\n" + "\n".join(s) + "\n" for n, s in records)
        out = args.outdir / f"chunk_{i:04d}.fasta"
        if out.exists() and out.read_text() == body:
            kept += 1
        else:
            out.write_text(body)
            written += 1

    for rec in read_fasta(args.inp):
        buf.append(rec)
        if len(buf) == args.chunk_size:
            flush(buf, idx); buf = []; idx += 1
    if buf:
        flush(buf, idx); idx += 1

    if idx == 0:
        raise SystemExit(f"no FASTA records in {args.inp} -- nothing to search")
    print(f"{idx} chunks of <= {args.chunk_size} ({written} written, {kept} unchanged)")


if __name__ == "__main__":
    main()
