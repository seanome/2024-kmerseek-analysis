#!/usr/bin/env python3
"""Step 1a of notebook 300: 10,000 real gut peptides, and the UHGP-90 proteins to hide.

No public human gut dataset on PRIDE ships a de novo peptide list (searched 2026-10-21: 477
gut microbiome projects, notebook 300 section 10). This is the task's fallback: peptides a
database search matched to a microbial protein, with every protein that holds the peptide
hidden from the database kmerseek searches.

Source: PXD069517 (fecal metaproteomes of celiac disease patients), file
MGDB_peptides_intensities_tax-funct_annotations.xlsx: Sequest-HT and Percolator against a
collection of human gut metagenomes ("MGDB").

Usage, in two calls around a grep:
  real_peptides.py select <xlsx> <human.fasta.gz>... --crap crap.fasta --out-dir DIR
  real_peptides.py hide <matched_lines.tsv> <spiked.tsv> --out-dir DIR

`select` keeps rows with Database MGDB, Confidence High (Percolator q-value below 0.01 in
Proteome Discoverer), 9 to 30 canonical residues, and no exact match (I and L read as one
letter) in the human reference proteome, its isoforms or cRAP. It draws 10,000 of them with a
fixed seed (12,000 drawn, so that 10,000 remain after `hide`) and writes their I/L-collapsed
strings for `grep -F -f`.

`hide` reads the UHGP-90 lines grep matched (accession, tab, I/L-collapsed sequence), maps each
real peptide to the proteins that contain it, and writes the proteins to hide. A protein that
is the source of a spiked peptide is never hidden: a real peptide found in one is dropped
instead, and the list is redrawn to keep 10,000.
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
N_REAL = 10_000
# Draws searched against UHGP-90: N_REAL plus room for the few dropped for sitting in a
# spiked peptide's source protein.
N_DRAWN = 12_000
SEED = 300
L_MIN, L_MAX = 9, 30


def collapse_il(seq):
    """I and L have the same mass; a search engine reports whichever the database had."""
    return seq.replace("I", "L")


def read_fasta(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    name, parts = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        yield name, "".join(parts)


def select(args):
    import openpyxl
    import polars as pl

    wb = openpyxl.load_workbook(args.xlsx, read_only=True)
    rows = wb.worksheets[0].iter_rows(values_only=True)
    header = list(next(rows))
    keep = [
        "Sequence",
        "Master Protein Accessions",
        "Database",
        "Master Protein Descriptions",
        "Confidence",
        "lca",
        "superkingdom",
        "phylum",
        "genus",
        "species",
    ]
    idx = [header.index(k) for k in keep]
    table = pl.DataFrame([[r[i] for i in idx] for r in rows], schema=keep, orient="row")
    n_total = len(table)
    table = table.filter(
        (pl.col("Database") == "MGDB") & (pl.col("Confidence") == "High")
    )
    n_mgdb_high = len(table)
    table = table.with_columns(length=pl.col("Sequence").str.len_chars()).filter(
        pl.col("length").is_between(L_MIN, L_MAX)
        & pl.col("Sequence").map_elements(
            lambda s: set(s) <= CANONICAL, return_dtype=pl.Boolean
        )
    )
    n_length = len(table)
    host = "\n".join(collapse_il(s) for path in args.human for _, s in read_fasta(path))
    host += "\n" + "\n".join(collapse_il(s) for _, s in read_fasta(args.crap))
    table = table.with_columns(
        in_human_or_crap=pl.col("Sequence").map_elements(
            lambda s: collapse_il(s) in host, return_dtype=pl.Boolean
        )
    )
    n_host = int(table["in_human_or_crap"].sum())
    table = table.filter(~pl.col("in_human_or_crap")).drop("in_human_or_crap")
    table = table.unique("Sequence", keep="first", maintain_order=True)
    n_pool = len(table)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_pool)
    table = (
        table[order.tolist()]
        .with_columns(draw_order=pl.int_range(pl.len()))
        .head(N_DRAWN)
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "real_peptide_pool.tsv", separator="\t")
    (out / "real_peptide_patterns_il.txt").write_text(
        "\n".join(sorted(set(collapse_il(s) for s in table["Sequence"]))) + "\n"
    )
    counts = {
        "rows_in_file": n_total,
        "mgdb_high_confidence": n_mgdb_high,
        "length_9_30_canonical": n_length,
        "exact_match_human_or_crap": n_host,
        "unique_pool": n_pool,
    }
    pl.DataFrame([counts]).write_csv(
        out / "real_peptide_filter_counts.tsv", separator="\t"
    )
    print(counts)


def hide(args):
    import polars as pl

    out = Path(args.out_dir)
    pool = pl.read_csv(out / "real_peptide_pool.tsv", separator="\t")
    spiked_sources = set(pl.read_csv(args.spiked, separator="\t")["source_protein"])
    patterns = {}
    for s in pool["Sequence"]:
        patterns.setdefault(collapse_il(s), []).append(s)
    by_len = sorted({len(p) for p in patterns})
    hits = {}
    with open(args.matched) as fh:
        for line in fh:
            acc, seq = line.rstrip("\n").split("\t")
            for L in by_len:
                for i in range(len(seq) - L + 1):
                    w = seq[i : i + L]
                    for original in patterns.get(w, ()):
                        hits.setdefault(original, set()).add(acc)
    pool = pool.with_columns(
        n_uhgp90_proteins_with_exact_match=pl.col("Sequence").map_elements(
            lambda s: len(hits.get(s, ())), return_dtype=pl.Int64
        ),
        in_spiked_source=pl.col("Sequence").map_elements(
            lambda s: bool(hits.get(s, set()) & spiked_sources), return_dtype=pl.Boolean
        ),
    )
    real = pool.filter(~pl.col("in_spiked_source")).sort("draw_order").head(N_REAL)
    if len(real) < N_REAL:
        print(f"only {len(real)} real peptides available", file=sys.stderr)
    real = real.with_columns(
        peptide_id=pl.format(
            "real{}", pl.int_range(1, pl.len() + 1).cast(pl.Utf8).str.zfill(5)
        )
    )
    hidden = sorted(set().union(*(hits.get(s, set()) for s in real["Sequence"])))
    real.write_csv(out / "real_peptides.tsv", separator="\t")
    (out / "uhgp90_hidden_proteins.txt").write_text("\n".join(hidden) + "\n")
    n_with = int((real["n_uhgp90_proteins_with_exact_match"] > 0).sum())
    print(
        f"{len(real)} real peptides; {n_with} have an exact UHGP-90 match; "
        f"{len(hidden)} UHGP-90 proteins hidden"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("xlsx")
    s.add_argument("human", nargs="+")
    s.add_argument("--crap", required=True)
    s.add_argument("--out-dir", required=True)
    h = sub.add_parser("hide")
    h.add_argument("matched")
    h.add_argument("spiked")
    h.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    select(args) if args.cmd == "select" else hide(args)


if __name__ == "__main__":
    main()
