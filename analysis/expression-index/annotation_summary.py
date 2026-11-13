"""One row per Botryllus protein: its length, its hit counts per tool, and its best MMseqs2 hit.

Reads the invertebrate-dark-set pipeline's hit files (MMseqs2, phmmer and jackhmmer against
Swiss-Prot with Ascidiacea removed; columns query, target, qstart, qend, bits, evalue) and
names each best MMseqs2 target from Swiss-Prot 2026_03's FASTA headers. pick_controls.py
chooses the six controls from this table.

    python annotation_summary.py HITS_DIR PROTEOME_FASTA DARK_PARQUET SPROT_HEADERS_GZ OUT_TSV_GZ
"""

import gzip
import re
import sys
from pathlib import Path

import polars as pl

COLS = ["query", "target", "qstart", "qend", "bits", "evalue"]
TOOLS = ["mmseqs2", "phmmer", "jackhmmer"]


def lengths(fasta):
    out, name, n = {}, None, 0
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    out[name] = n
                name, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    out[name] = n
    return pl.DataFrame({"protein": list(out), "length": list(out.values())})


def sprot_names(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            m = re.match(r">sp\|(\w+)\|\S+ (.*?) OS=(.*?) OX=", line)
            if m:
                rows.append(m.groups())
    return pl.DataFrame(rows, schema=["target", "description", "species"], orient="row")


def hits(hits_dir, tool):
    files = sorted(Path(hits_dir).glob(f"*.{tool}.tsv.gz"))
    return pl.concat([
        pl.read_csv(f, separator="\t", has_header=False, new_columns=COLS,
                    schema_overrides={"evalue": pl.Float64, "bits": pl.Float64})
        for f in files
    ])


def main(hits_dir, fasta, dark_parquet, sprot_headers, out):
    table = lengths(fasta)
    dark = pl.read_parquet(dark_parquet).select(pl.col("accession").alias("protein"))
    table = table.with_columns(pl.col("protein").is_in(dark["protein"].implode()).alias("dark"))
    for tool in TOOLS:
        h = hits(hits_dir, tool)
        per = h.group_by("query").agg(
            pl.len().alias(f"{tool}_rows"),
            pl.col("evalue").min().alias(f"{tool}_min_evalue"),
        ).rename({"query": "protein"})
        table = table.join(per, on="protein", how="left")
        if tool == "mmseqs2":
            # best hit = highest bit score; ties broken by target accession so the pick is fixed
            best = (h.sort(["query", "bits", "target"], descending=[False, True, False])
                    .group_by("query", maintain_order=True).first()
                    .select(pl.col("query").alias("protein"),
                            pl.col("target").alias("mmseqs2_best_target"),
                            pl.col("bits").alias("mmseqs2_best_bits"),
                            pl.col("evalue").alias("mmseqs2_best_evalue")))
            table = table.join(best, on="protein", how="left")
    names = sprot_names(sprot_headers).rename({
        "target": "mmseqs2_best_target", "description": "mmseqs2_best_description",
        "species": "mmseqs2_best_species"})
    table = table.join(names, on="mmseqs2_best_target", how="left")
    table = table.with_columns(pl.col(c).fill_null(0) for c in table.columns if c.endswith("_rows"))
    table.sort("protein").write_csv(out, separator="\t")
    named = table.filter(pl.col("mmseqs2_best_target").is_not_null())
    print(f"proteins {table.height}, dark {table['dark'].sum()}, "
          f"with an MMseqs2 row {named.height}, "
          f"best target not in the Swiss-Prot headers {named['mmseqs2_best_description'].null_count()}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
