#!/usr/bin/env python3
"""Every annotated region of every reviewed Swiss-Prot entry, with its note and evidence.

One pass over uniprot_sprot.dat.gz. Writes two parquet files:

  <out>_features.parquet  accession, feature_type, start, end (1-based, inclusive),
                          length, description (the /note), evidence (the ECO codes,
                          ';'-joined), experimental (any ECO:0000269), is_mammal
  <out>_sequences.parquet accession, length, is_mammal, sequence

Feature types kept: DOMAIN, REGION, MOTIF, REPEAT, ZN_FING, COMPBIAS, COILED, TRANSMEM,
INTRAMEM. A feature whose start or end is fuzzy (<, >, ?) or points into another entry
(P12345:10..20) is dropped, because an uncertain boundary cannot be scored as truth or cut
out as an index entry. The count dropped per type is printed.

Mammal or not is read off the OC lineage lines, the same rule build_clade_excluded_reference.py
uses (`Mammalia` on an OC line is NCBI taxon 40674).

nextflow-runs/qfo-pfam-region-benchmark/bin/build_swissprot_truth.py parses FT lines too but
keeps only type and position: no /note, no /evidence, no COMPBIAS. This parser reads the
qualifier lines under each FT line, including notes and evidence lists that wrap.
"""

import argparse
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

import polars as pl

FEATURE_TYPES = {
    "DOMAIN",
    "REGION",
    "MOTIF",
    "REPEAT",
    "ZN_FING",
    "COMPBIAS",
    "COILED",
    "TRANSMEM",
    "INTRAMEM",
}
EXPERIMENTAL = "ECO:0000269"

# "FT   DOMAIN          23..120" or "FT   MOTIF           7" (a one-residue feature).
FT_START_RE = re.compile(r"^FT   (\S+)\s+(\S+)\s*$")
POS_RE = re.compile(r"^(\d+)(?:\.\.(\d+))?$")
ECO_RE = re.compile(r"ECO:\d{7}")


def parse_qualifiers(lines: list[str]) -> dict[str, str]:
    """Join the qualifier lines of one feature into {name: value}.

    A qualifier starts with /name="value and may wrap over several lines. A wrapped note
    joins with a space; a wrapped evidence list breaks after a comma, so it joins with no
    space and the ECO codes are pulled out with a regex anyway.
    """
    out: dict[str, str] = {}
    name = None
    for raw in lines:
        text = raw[21:].rstrip("\n")
        if text.startswith("/"):
            key, _, val = text[1:].partition("=")
            name = key
            out[name] = val
        elif name is not None:
            sep = "" if out[name].endswith(",") else " "
            out[name] += sep + text
    return {k: v.strip('"') for k, v in out.items()}


def stream(dat_path: Path):
    """Yield (accession, is_mammal, sequence, features) per reviewed entry."""
    acc = None
    reviewed = False
    lineage: list[str] = []
    seq: list[str] = []
    in_seq = False
    feats: list[tuple[str, str, list[str]]] = []  # (type, location, qualifier lines)

    with gzip.open(dat_path, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "ID":
                reviewed = "Reviewed;" in line
            elif tag == "AC" and acc is None:
                acc = line[5:].split(";")[0].strip()
            elif tag == "OC":
                lineage.extend(
                    t.strip().rstrip(".") for t in line[5:].split(";") if t.strip()
                )
            elif tag == "FT":
                m = FT_START_RE.match(line)
                if m and line[5] != " ":
                    feats.append((m.group(1), m.group(2), []))
                elif feats:
                    feats[-1][2].append(line)
            elif tag == "SQ":
                in_seq = True
            elif in_seq and line.startswith("     "):
                seq.append(line.strip().replace(" ", ""))
            elif line.startswith("//"):
                if acc and reviewed:
                    yield acc, "Mammalia" in lineage, "".join(seq), feats
                acc, reviewed, lineage, seq, in_seq, feats = (
                    None,
                    False,
                    [],
                    [],
                    False,
                    [],
                )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--swissprot-dat", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    feat_rows: list[tuple] = []
    seq_rows: list[tuple] = []
    dropped: Counter = Counter()
    for acc, is_mammal, sequence, feats in stream(args.swissprot_dat):
        seq_rows.append((acc, len(sequence), is_mammal, sequence))
        for ftype, loc, qual_lines in feats:
            if ftype not in FEATURE_TYPES:
                continue
            m = POS_RE.match(loc)
            if not m:
                dropped[ftype] += 1
                continue
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            if end > len(sequence) or start < 1 or end < start:
                dropped[ftype] += 1
                continue
            q = parse_qualifiers(qual_lines)
            eco = ECO_RE.findall(q.get("evidence", ""))
            feat_rows.append(
                (
                    acc,
                    ftype,
                    start,
                    end,
                    end - start + 1,
                    q.get("note", ""),
                    ";".join(eco),
                    EXPERIMENTAL in eco,
                    is_mammal,
                )
            )

    feats_df = pl.DataFrame(
        feat_rows,
        orient="row",
        schema={
            "accession": pl.Utf8,
            "feature_type": pl.Utf8,
            "start": pl.Int32,
            "end": pl.Int32,
            "length": pl.Int32,
            "description": pl.Utf8,
            "evidence": pl.Utf8,
            "experimental": pl.Boolean,
            "is_mammal": pl.Boolean,
        },
    )
    seqs_df = pl.DataFrame(
        seq_rows,
        orient="row",
        schema={
            "accession": pl.Utf8,
            "length": pl.Int32,
            "is_mammal": pl.Boolean,
            "sequence": pl.Utf8,
        },
    )
    if seqs_df["accession"].n_unique() != seqs_df.height:
        sys.exit("duplicate accessions in the flat file; the AC parse is wrong")

    feats_df.write_parquet(f"{args.out_prefix}_features.parquet", compression="zstd")
    seqs_df.write_parquet(f"{args.out_prefix}_sequences.parquet", compression="zstd")

    print(
        f"[features] entries={seqs_df.height} mammal={int(seqs_df['is_mammal'].sum())} "
        f"features={feats_df.height}",
        file=sys.stderr,
    )
    for ftype, n in sorted(feats_df["feature_type"].value_counts().iter_rows()):
        print(
            f"[features] {ftype:9s} kept={n:>8} dropped_fuzzy={dropped[ftype]}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
