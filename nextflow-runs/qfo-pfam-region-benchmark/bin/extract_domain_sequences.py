#!/usr/bin/env python3
"""Write one FASTA record per annotated domain instance, for measuring pairwise identity.

The hypothesis under test is that HP patterning survives *after sequence identity
diverges*, so percent identity is the axis the whole claim lives on. It has to be measured
per domain pair: species divergence in MYA is a species-level average, and within a single
human-mouse comparison individual domain pairs run from ~20% to ~99% identity, which is
precisely the range the claim is about. Stratifying on MYA smears it away.

Identity is computed between DOMAIN REGIONS, not whole proteins. Two proteins sharing one
conserved domain in otherwise unrelated sequence would show low whole-protein identity
while the domain itself is highly conserved -- scoring that pair as "remote" would credit
every tool with a rescue it did not perform.

Record ids encode the instance so the alignment output can be joined straight back:
    <accession>|<pfam_id>|<start>-<end>
"""

import argparse
from pathlib import Path

import polars as pl


def read_fasta(path: Path) -> dict[str, str]:
    recs, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    recs[name] = "".join(buf)
                parts = line[1:].split("|")
                name = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name:
        recs[name] = "".join(buf)
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", required=True, type=Path,
                   help="domain table: accession, pfam_id, domain_start, domain_end")
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-length", type=int, default=10,
                   help="skip very short regions; identity over a handful of residues is "
                        "noise, and point features are 1-2 residues wide")
    args = p.parse_args()

    seqs = read_fasta(args.fasta)
    dom = pl.read_parquet(args.truth).select(
        "accession", "pfam_id", "domain_start", "domain_end"
    ).unique()

    written = skipped_missing = skipped_short = 0
    with open(args.out, "w") as out:
        for acc, pfam, start, end in dom.iter_rows():
            seq = seqs.get(acc)
            if seq is None:
                skipped_missing += 1
                continue
            # Annotations are 1-based inclusive; python slicing is 0-based half-open.
            sub = seq[max(0, start - 1):end]
            if len(sub) < args.min_length:
                skipped_short += 1
                continue
            out.write(f">{acc}|{pfam}|{start}-{end}\n{sub}\n")
            written += 1

    print(f"wrote {written} domain regions "
          f"({skipped_missing} accessions absent from fasta, {skipped_short} too short)")


if __name__ == "__main__":
    main()
