#!/usr/bin/env python3
"""Changed domains of the standard path, per labelling.

Standard path steps 4 and 5 of notebooks/360_data/preregistered.md: each Trinity
transcript takes every Pfam family of its DIAMOND best hit (highest bit score, E <= 1e-5;
ties broken by target accession so the pick is total); the changed domains of a labelling
are the families of transcripts with padj < 0.05.

    standard_calls.py --diamond blastx.tsv --domains search_db.pfam.tsv --key answer_key.tsv \
        --organism ciona --out-prefix ciona.standard <transcripts>.<labelling>.deseq2.tsv.gz ...

Writes <out-prefix>.calls.tsv (labelling, transcript, best hit, family, in_key, padj,
log2FoldChange) and <out-prefix>.summary.tsv (same columns as landings.py's summary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

PADJ = 0.05
DIAMOND_COLS = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qstart", "qend", "sstart", "send"]


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--diamond", "--domains", "--key", "--organism", "--out-prefix"):
        ap.add_argument(a, required=True)
    ap.add_argument("deseq2", nargs="+")
    args = ap.parse_args()

    key = pl.read_csv(args.key, separator="\t")
    key_fams = set(key.filter(pl.col("status") != "unchanged")["pfam_acc"].to_list())
    assert len(key_fams) == 100, len(key_fams)
    hits = pl.read_csv(args.diamond, separator="\t", has_header=False, new_columns=DIAMOND_COLS,
                       schema_overrides={"qseqid": pl.Utf8, "sseqid": pl.Utf8})
    best = (hits.filter(pl.col("evalue") <= 1e-5).sort(["qseqid", "bitscore", "sseqid"], descending=[False, True, False])
            .unique("qseqid", keep="first", maintain_order=True))
    dom = pl.read_csv(args.domains, separator="\t").select(pl.col("target").alias("sseqid"), "pfam_acc").unique()
    tx_fams = best.select("qseqid", "sseqid", "pident", "evalue", "bitscore").join(dom, on="sseqid", how="inner")

    calls, rows = [], []
    for f in sorted(args.deseq2):
        lab = Path(f).name.split(".")[-4]
        d = pl.read_csv(f, separator="\t", null_values=["NA"], schema_overrides={"padj": pl.Float64, "log2FoldChange": pl.Float64})
        sig = d.filter(pl.col("padj") < PADJ).rename({"feature": "qseqid"})
        c = sig.join(tx_fams, on="qseqid", how="inner").with_columns(
            pl.lit(lab).alias("labelling"), pl.col("pfam_acc").is_in(list(key_fams)).alias("in_key"))
        calls.append(c)
        fams = set(c["pfam_acc"].to_list())
        rows.append(dict(
            organism=args.organism, path="standard", setting="trinity+diamond", status="ok",
            labelling=lab, evalue_cutoff=1e-5, primary=True,
            recovered=len(fams & key_fams), recovery_fraction=len(fams & key_fams) / 100,
            outside=len(fams - key_fams), n_changed_domains=len(fams),
            n_queries=d.height, n_landing_rows=sig.height,
        ))
    pl.concat(calls).sort("labelling", "qseqid", "pfam_acc").write_csv(f"{args.out_prefix}.calls.tsv", separator="\t")
    pl.DataFrame(rows).write_csv(f"{args.out_prefix}.summary.tsv", separator="\t")


if __name__ == "__main__":
    main()
