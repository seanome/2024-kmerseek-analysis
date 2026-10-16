#!/usr/bin/env python3
"""Split a proteome FASTA into fixed-size chunks for a SLURM array.

Chunks are written only when their content would differ from what is already on disk.
InterProScan chunks are inputs to a job array whose tasks are keyed by chunk index, and a
rewrite that changes nothing but the mtime makes every downstream reader look stale --
the same failure that bites Nextflow's cache on regenerated inputs.
"""

import argparse
from pathlib import Path


def read_fasta(path: Path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line.rstrip("\n"), []
            else:
                seq.append(line.rstrip("\n"))
    if name is not None:
        yield name, "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="stop after this many chunks (the mini run takes 1)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = kept = 0
    buf: list[tuple[str, str]] = []
    idx = 0

    def flush(records) -> bool:
        nonlocal written, kept
        out = args.outdir / f"chunk_{idx:04d}.fasta"
        body = "".join(f"{n}\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n"
                       for n, s in records)
        if out.exists() and out.read_text() == body:
            kept += 1
            return False
        out.write_text(body)
        written += 1
        return True

    for rec in read_fasta(args.inp):
        buf.append(rec)
        if len(buf) == args.chunk_size:
            flush(buf)
            buf = []
            idx += 1
            if args.max_chunks is not None and idx >= args.max_chunks:
                break
    if buf and (args.max_chunks is None or idx < args.max_chunks):
        flush(buf)
        idx += 1

    print(f"{idx} chunks in {args.outdir} ({written} written, {kept} unchanged)")


if __name__ == "__main__":
    main()
