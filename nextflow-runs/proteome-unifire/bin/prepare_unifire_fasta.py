#!/usr/bin/env python3
"""Rewrite a proteome FASTA into the header format UniFIRE requires.

UniFIRE's minimum header is `>{id}|{name} {flags}` with `OX={taxid}` mandatory, because
UniRule conditions are taxon-scoped: without a taxid most rules cannot fire and the run
completes with a near-empty prediction set that looks like a real biological result.

Botryllus is the case in hand. Its headers are bare gene-model ids -- `>FUN000001_FUN000001`
-- with no OX, no OS and no pipe, so it needs this. A UniProt-sourced proteome like
UP000005408 already carries OX= and passes through unchanged unless --force is given.

ORDERING. If UniFIRE is going to be run from precomputed InterProScan XML (the lite image),
InterProScan must be run on the OUTPUT of this script, not on the original FASTA. The taxid
reaches UniFIRE through the sequence identifier recorded in that XML; annotate afterwards
and there is nowhere for it to come from.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taxid", required=True)
    ap.add_argument("--organism", default=None, help="OS= value, optional but recommended")
    ap.add_argument("--force", action="store_true",
                    help="rewrite headers even if they already carry OX=")
    args = ap.parse_args()

    n = already = 0
    out_lines: list[str] = []
    with open(args.inp) as fh:
        for line in fh:
            if not line.startswith(">"):
                out_lines.append(line)
                continue
            n += 1
            header = line[1:].strip()
            if "OX=" in header and not args.force:
                already += 1
                out_lines.append(line)
                continue
            # Keep the original identifier verbatim as both id and name. The accession is
            # the join key back to the structural truth key and to every search result, so
            # a header rewrite that alters it would silently break every downstream join.
            ident = header.split()[0] if header.split() else header
            ident = ident.split("|")[-1] if "|" in ident else ident
            os_flag = f" OS={args.organism}" if args.organism else ""
            out_lines.append(f">{ident}|{ident}{os_flag} OX={args.taxid}\n")

    if n == 0:
        sys.exit(f"no FASTA records in {args.inp}")
    args.out.write_text("".join(out_lines))
    print(f"{n} records -> {args.out} "
          f"({n - already} rewritten, {already} already carried OX=)")


if __name__ == "__main__":
    main()
