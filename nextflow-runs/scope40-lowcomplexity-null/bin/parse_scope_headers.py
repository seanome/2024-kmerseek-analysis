#!/usr/bin/env python3
"""
Parse SCOPe ASTRAL FASTA headers into a domain -> SCOP hierarchy table.

Header format:
    >d1dlwa_ a.1.1.1 (A:) Protoglobin {Methanosarcina acetivorans [TaxId: 188937]}

Emits domain_id plus the class / fold / superfamily / family prefixes of the
SCOP sunid, which the hit tables join on to label pairs as same-fold etc.
Identical in behaviour to the parser in nextflow-runs/hp-alphabet-sweep/main.nf,
lifted out of the inline script so it can be unit-checked and reused.

Usage:
    parse_scope_headers.py --fasta scope40.fa --output scope_domains.tsv
"""

from __future__ import annotations

import argparse
import re
import sys

HEADER_RE = re.compile(r"^>(\S+)\s+(\S+)\s+\([^)]*\)\s+(.*?)\s+\{(.*?)\}\s*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    n = 0
    with open(args.fasta) as handle, open(args.output, "w") as out:
        out.write(
            "domain_id\tscop_id\tscop_class\tscop_fold\tscop_superfamily\t"
            "scop_family\tprotein_name\tspecies\n"
        )
        for line in handle:
            if not line.startswith(">"):
                continue
            match = HEADER_RE.match(line.rstrip())
            if match:
                domain_id, scop_id, pname, species = match.groups()
            else:
                parts = line[1:].rstrip().split(None, 2)
                domain_id = parts[0] if parts else ""
                scop_id = parts[1] if len(parts) > 1 else ""
                pname, species = "", ""

            levels = scop_id.split(".")
            scop_class = levels[0] if levels else scop_id
            scop_fold = ".".join(levels[:2]) if len(levels) >= 2 else scop_id
            scop_superfamily = ".".join(levels[:3]) if len(levels) >= 3 else scop_id

            out.write(
                f"{domain_id}\t{scop_id}\t{scop_class}\t{scop_fold}\t"
                f"{scop_superfamily}\t{scop_id}\t"
                f"{pname.replace(chr(9), ' ')}\t{species.replace(chr(9), ' ')}\n"
            )
            n += 1

    print(f"Parsed {n:,} SCOP domains", file=sys.stderr)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
