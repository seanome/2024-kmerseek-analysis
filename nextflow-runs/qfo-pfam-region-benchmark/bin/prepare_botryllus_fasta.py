#!/usr/bin/env python3
"""
Turn the Botryllus schlosseri genome protein FASTA into a proteome the pipeline can index.

Botryllus is a TARGET-ONLY species: it is searched by every sequence arm and by ProstT5,
and it is never scored, because it has no Pfam annotations. See the `botryllus` row in
assets/qfo_species.tsv and the searched-but-not-scored note in main.nf.

The download has two defects that break the tools downstream, and this fixes both:

1. Repeated names. 2019 names carry 2..6 records each -- 4613 records in all, of which only
   4 groups are exact sequence duplicates; the rest are genuinely different sequences filed
   under one name (isoforms). `mmseqs createdb`, hmmer and foldseek all key on the name, so
   a repeat silently collapses or mis-attributes hits.

   Every record is KEPT. A name that appears once is passed through byte-identical; a name
   that appears n>1 times becomes <name>.1 .. <name>.n, ordered by descending length then by
   the sequence string so a rerun assigns the same suffixes. Keeping the isoforms rather
   than collapsing to the longest costs 6% more sequence and no consistency: botryllus is
   never scored, so there is no per-protein denominator to keep aligned with the other nine
   proteomes, and the point of adding the species is to find a domain-bearing hit -- the
   isoform dropped could be the one carrying it.

2. One '.' character, in FUN029508_FUN029508. '.' is not an amino acid. It is replaced with
   X, the standard unknown-residue code.

FUN008084_FUN008084 is the Botryllus histocompatibility factor, the flagship case for this
whole benchmark. It is a single 252-aa record under a name that is not repeated, so it must
survive with its name unchanged. That is asserted rather than assumed.

Usage:
    python3 bin/prepare_botryllus_fasta.py \
        --in  ~/Downloads/botryllus_new_genome_protein.fasta \
        --out ~/data/quest-for-orthologs/<release>/Eukaryota/BOTSCH2026_30301.fasta \
        --summary-out botryllus_prepare_summary.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# The 20 proteinogenic amino acids, plus X for an unknown residue. Anything else in the
# input is replaced with X and counted.
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
UNKNOWN = "X"

# The Botryllus histocompatibility factor. Its name must reach the index unchanged.
BHF_NAME = "FUN008084_FUN008084"
BHF_LENGTH = 252


def read_records(path: Path) -> list[tuple[str, list[str]]]:
    """(name, raw sequence lines) in file order, keeping the original line wrapping.

    The lines are kept as they were read so a record this script does not touch can be
    written back byte-identical rather than re-wrapped at whatever width we would pick.
    """
    records: list[tuple[str, list[str]]] = []
    name: str | None = None
    lines: list[str] = []
    with open(path) as fh:
        for raw in fh:
            if raw.startswith(">"):
                if name is not None:
                    records.append((name, lines))
                name = raw[1:].rstrip("\n")
                lines = []
            else:
                stripped = raw.rstrip("\n")
                if stripped:
                    lines.append(stripped)
    if name is not None:
        records.append((name, lines))
    return records


def clean_lines(lines: list[str]) -> tuple[list[str], int]:
    """Replace non-standard residues with X. Returns the lines and how many were replaced."""
    n_replaced = 0
    out = []
    for line in lines:
        if all(c in STANDARD or c == UNKNOWN for c in line):
            out.append(line)
            continue
        chars = []
        for c in line:
            if c in STANDARD or c == UNKNOWN:
                chars.append(c)
            else:
                chars.append(UNKNOWN)
                n_replaced += 1
        out.append("".join(chars))
    return out, n_replaced


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="infile", required=True, type=Path,
                   help="the downloaded botryllus protein FASTA")
    p.add_argument("--out", dest="outfile", required=True, type=Path,
                   help="cleaned FASTA to write, e.g. <qfo>/Eukaryota/BOTSCH2026_30301.fasta")
    p.add_argument("--summary-out", type=Path,
                   help="JSON summary; printed to stdout regardless")
    args = p.parse_args()

    records = read_records(args.infile)
    if not records:
        raise SystemExit(f"no FASTA records in {args.infile}")

    counts = Counter(name for name, _ in records)
    dup_names = {n for n, c in counts.items() if c > 1}

    # Suffixes are assigned per name-group, ordered by descending length then by the
    # sequence itself, so the same input always produces the same <name>.<i> assignment.
    # Without a total order the suffixes would follow file order, which is fine here but
    # would move if the download were ever re-exported in a different order.
    groups: dict[str, list[str]] = {}
    for name, lines in records:
        if name in dup_names:
            groups.setdefault(name, []).append("".join(lines))
    suffix_of: dict[tuple[str, str], list[int]] = {}
    for name, seqs in groups.items():
        for i, seq in enumerate(sorted(seqs, key=lambda s: (-len(s), s)), start=1):
            suffix_of.setdefault((name, seq), []).append(i)

    n_replaced_total = 0
    n_renamed = 0
    chunks: list[str] = []
    for name, lines in records:
        cleaned, n_replaced = clean_lines(lines)
        n_replaced_total += n_replaced
        out_name = name
        if name in dup_names:
            # pop, not peek: the 4 groups that are exact sequence duplicates map several
            # records onto the same key and must still get distinct suffixes.
            out_name = f"{name}.{suffix_of[(name, ''.join(lines))].pop(0)}"
            n_renamed += 1
        chunks.append(">" + out_name + "\n")
        chunks.extend(line + "\n" for line in cleaned)

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile.write_text("".join(chunks))

    # ---- verify the OUTPUT, not the plan ----
    written = read_records(args.outfile)
    out_counts = Counter(name for name, _ in written)
    still_dup = sorted(n for n, c in out_counts.items() if c > 1)
    if still_dup:
        raise SystemExit(f"{args.outfile}: {len(still_dup)} names are still repeated, "
                         f"e.g. {still_dup[:5]}")
    bad = Counter()
    for _, lines in written:
        for line in lines:
            for c in line:
                if c not in STANDARD and c != UNKNOWN:
                    bad[c] += 1
    if bad:
        raise SystemExit(f"{args.outfile}: non-standard residues survived: {dict(bad)}")

    bhf = [lines for name, lines in written if name == BHF_NAME]
    bhf_ok = len(bhf) == 1 and len("".join(bhf[0])) == BHF_LENGTH
    if not bhf_ok:
        raise SystemExit(
            f"{BHF_NAME} (the Botryllus histocompatibility factor, the flagship case) is "
            f"not in {args.outfile} as a single {BHF_LENGTH}-aa record: found "
            f"{len(bhf)} record(s) of length {[len(''.join(l)) for l in bhf]}. It must "
            f"reach the index under its original name."
        )

    summary = {
        "input": str(args.infile),
        "output": str(args.outfile),
        "n_records_in": len(records),
        "n_records_out": len(written),
        "n_residues_in": sum(len("".join(l)) for _, l in records),
        "n_residues_out": sum(len("".join(l)) for _, l in written),
        "n_names_in": len(counts),
        "n_names_disambiguated": len(dup_names),
        "n_records_renamed": n_renamed,
        "n_non_standard_residues_replaced": n_replaced_total,
        "bhf_assertion": (
            f"{BHF_NAME} present, name unchanged, {BHF_LENGTH} aa: {bhf_ok}"
        ),
    }
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
