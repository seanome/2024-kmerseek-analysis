#!/usr/bin/env python3
"""Put MMseqs2 or Foldseek alignments into the same columns as parse_phmmer.py writes.

MMseqs2 names a UniProt-style header by its accession alone (`P02144`, not
`sp|P02144|MYG_HUMAN`), and other headers by their first word;
Foldseek by the structure file name (`AF-P02144-F1`, with or without `.cif`). Both become
the bare accession. Only alignments between two different family proteins are kept.

    normalize_alignments.py tool alignments.m8 family.fasta out.tsv
"""

import re
import sys

import polars as pl

tool, m8, family_fasta, out = sys.argv[1:5]
family = {
    line[1:].split()[0].split("|")[1] for line in open(family_fasta) if line.startswith(">")
}

if tool == "mmseqs2":
    cols = ["query", "target", "fident", "qstart", "qend", "tstart", "tend", "evalue",
            "bitscore", "qaln", "taln"]
    to_acc = lambda s: s.split("|")[1] if "|" in s else s
else:
    cols = ["query", "target", "fident", "qstart", "qend", "tstart", "tend", "evalue",
            "bitscore", "prob", "qaln", "taln"]
    to_acc = lambda s: re.match(r"AF-([A-Z0-9]+)-F\d+", s).group(1)

df = pl.read_csv(m8, separator="\t", has_header=False, new_columns=cols,
                 infer_schema_length=0, quote_char=None)
df = (
    df.with_columns(
        query=pl.col("query").map_elements(to_acc, return_dtype=pl.String),
        target=pl.col("target").map_elements(to_acc, return_dtype=pl.String),
        tool=pl.lit(tool),
    )
    .filter(pl.col("target").is_in(family) & (pl.col("query") != pl.col("target")))
    .select("tool", "query", "target", "evalue", "bitscore", "qstart", "qend", "tstart",
            "tend", "qaln", "taln")
)
df.write_csv(out, separator="\t")
print(f"{tool}: {df.height} family alignments")
