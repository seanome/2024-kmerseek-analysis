#!/usr/bin/env python3
"""Stage the query proteome: one accession per header, then fixed-size chunks.

45_339 proteins against 572_700 reference sequences does not fit one walltime for any arm,
and a scheduler kill reports exitStatus Integer.MAX_VALUE rather than a signal code, so the
run would retry into the same wall rather than failing usefully.

HEADERS ARE REWRITTEN TO THE BARE ACCESSION, and every downstream step reads the rewritten
proteome (--proteome-out), never the original. The three sequence arms do not agree on
what a UniProt-style header means. A QfO proteome names its records

    >tr|A0A075B5I0|A0A075B5I0_MOUSE Immunoglobulin heavy variable V1-13 OS=...

phmmer and jackhmmer report the first whitespace token, tr|A0A075B5I0|A0A075B5I0_MOUSE.
mmseqs createdb parses the UniProt form and reports A0A075B5I0 -- checked against the
pinned mmseqs2 image, not assumed. compute_dark_set.py keys the proteome on the first
token, so the mmseqs2 arm would have joined NOTHING for every QfO species and the dark set
would have been "what phmmer and jackhmmer miss" while claiming three arms. Botryllus's
FUN000001_FUN000001 ids carry no '|' and agree under every parser, which is exactly what
let the pipeline run clean on it. Rewriting once, here, gives every arm the same key.

The rewrite is the UniProt one and nothing else: sp|ACC|NAME and tr|ACC|NAME become ACC;
any other header keeps its first whitespace token. Any two records collapsing onto one
accession is an error, not a warning -- a duplicate key double-counts in every downstream
table.

Content-stable: a chunk is only rewritten when its bytes would change. An identical rewrite
still moves the mtime, and every task reading that chunk would re-run on the next -resume
for no reason.
"""

import argparse
import re
from pathlib import Path

UNIPROT = re.compile(r"^(?:sp|tr)\|([^|\s]+)\|\S*")


def accession_of(header: str) -> str:
    """The join key every arm will report for this record."""
    token = header[1:].strip().split()[0] if header[1:].strip() else ""
    if not token:
        raise SystemExit(f"empty FASTA header: {header!r}")
    m = UNIPROT.match(token)
    return m.group(1) if m else token


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


def write_if_changed(out: Path, body: str) -> bool:
    if out.exists() and out.read_text() == body:
        return False
    out.write_text(body)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--proteome-out", type=Path, required=True,
                    help="the whole proteome with rewritten headers; what every "
                         "non-search step reads")
    ap.add_argument("--chunk-size", type=int, default=2000)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = kept = idx = 0
    buf = []
    seen: dict[str, str] = {}
    whole = []

    def flush(records, i):
        nonlocal written, kept
        body = "".join(f">{acc}\n" + "\n".join(s) + "\n" for acc, s in records)
        if write_if_changed(args.outdir / f"chunk_{i:04d}.fasta", body):
            written += 1
        else:
            kept += 1
        whole.append(body)

    for header, seq in read_fasta(args.inp):
        acc = accession_of(header)
        if acc in seen:
            raise SystemExit(
                f"two records share the accession {acc!r} after header rewriting:\n"
                f"  {seen[acc]}\n  {header}\n"
                f"A duplicate key double-counts in every downstream table; fix the input.")
        seen[acc] = header
        buf.append((acc, seq))
        if len(buf) == args.chunk_size:
            flush(buf, idx); buf = []; idx += 1
    if buf:
        flush(buf, idx); idx += 1

    if idx == 0:
        raise SystemExit(f"no FASTA records in {args.inp} -- nothing to search")
    write_if_changed(args.proteome_out, "".join(whole))
    n_rewritten = sum(1 for acc, h in seen.items() if h[1:].split()[0] != acc)
    print(f"{len(seen)} proteins, {n_rewritten} UniProt-style headers rewritten to the "
          f"accession; {idx} chunks of <= {args.chunk_size} "
          f"({written} written, {kept} unchanged)")


if __name__ == "__main__":
    main()
