#!/usr/bin/env python3
"""Count the superkingdoms and phyla each Pfam family occurs in, over Pfam's sequences.

Pfam 38's sequence database (pfamseq) is the UniProtKB reference proteomes of release
2025_03. Pfam-A.regions.tsv.gz lists every family hit on those sequences but no taxon, and
pfamseq's FASTA headers carry only the UniProt entry name (ACCESSION_MNEMONIC). So:

    region (pfamseq_acc, pfamA_acc)
      -> pfamseq header: accession -> species mnemonic (the part after "_")
      -> UniProt speclist.txt: mnemonic -> kingdom letter (A, B, E, V, O) and NCBI taxid
      -> NCBI new_taxdump: taxid -> phylum (the node itself when its rank is phylum)

Superkingdom count: distinct kingdom letters among A (archaea), B (bacteria), E (eukaryota).
Viruses (V) and others (O) are counted in their own columns and not in the superkingdoms.
Phylum count: distinct phyla. A sequence whose taxon sits above phylum rank (a "virtual"
species code such as 9BACT = Bacteria) has no phylum; it still counts for its superkingdom,
and n_seq_no_phylum records how many such sequences each family has.

Inputs are plain TSVs the sbatch job cuts out of the two big gzip files, because polars
decompresses a gzip CSV whole in memory before it parses it.
"""

import argparse
import re
from pathlib import Path

import polars as pl


def read_speclist(path: Path) -> pl.DataFrame:
    """Mnemonic, kingdom letter and taxid for every code line of speclist.txt."""
    pat = re.compile(r"^([A-Z0-9]{1,5})\s+([ABEVOX])\s+(\d+):\s+N=")
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = pat.match(line)
        if m:
            rows.append((m.group(1), m.group(2), int(m.group(3))))
    return pl.DataFrame(rows, schema=["mnemonic", "kingdom", "taxid"], orient="row")


def read_phylum(nodes: Path, rankedlineage: Path) -> pl.DataFrame:
    """taxid -> phylum name. rankedlineage.dmp lists a node's ANCESTORS by rank, so a node
    that is itself a phylum has an empty phylum field; nodes.dmp gives its rank."""
    def dmp(path, cols):
        out = []
        with open(path) as fh:
            for line in fh:
                # Fields are separated by "\t|\t" and the line ends "\t|"; empty fields
                # are kept, so split on "\t|" and strip the leading tab of each.
                f = [x.strip() for x in line.rstrip("\n").split("\t|")]
                out.append(tuple(f[i] for i in cols))
        return out
    # rankedlineage.dmp: tax_id, tax_name, species, genus, family, order, class, phylum, ...
    rl = pl.DataFrame(dmp(rankedlineage, [0, 1, 7]),
                      schema=["taxid", "tax_name", "phylum"], orient="row")
    nd = pl.DataFrame(dmp(nodes, [0, 2]), schema=["taxid", "rank"], orient="row")
    df = rl.join(nd, on="taxid", how="left").with_columns(
        pl.col("taxid").cast(pl.Int64),
        pl.when(pl.col("rank") == "phylum").then(pl.col("tax_name"))
        .when(pl.col("phylum") != "").then(pl.col("phylum"))
        .otherwise(None).alias("phylum"),
    )
    return df.select("taxid", "phylum")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", type=Path, required=True,
                    help="pfamseq_acc<TAB>pfamA_acc, no header")
    ap.add_argument("--mnemonics", type=Path, required=True,
                    help="pfamseq_acc<TAB>mnemonic, no header")
    ap.add_argument("--speclist", type=Path, required=True)
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--rankedlineage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = read_speclist(args.speclist)
    phylum = read_phylum(args.nodes, args.rankedlineage)
    # The phylum column's position in rankedlineage.dmp is read, not trusted: human must
    # come out as Chordata and E. coli K-12 as Pseudomonadota.
    for taxid, want in [(9606, "Chordata"), (83333, "Pseudomonadota")]:
        got = phylum.filter(pl.col("taxid") == taxid)["phylum"].to_list()
        if got and got != [want]:
            raise SystemExit(f"rankedlineage.dmp layout changed: taxid {taxid} -> {got}")
    taxa = spec.join(phylum, on="taxid", how="left")
    print(f"speclist codes: {spec.height:,}; with a phylum: "
          f"{taxa['phylum'].is_not_null().sum():,}")

    opts = dict(separator="\t", has_header=False, quote_char=None)
    regions = pl.scan_csv(args.regions, new_columns=["acc", "pfam_id"], **opts)
    mnem = pl.scan_csv(args.mnemonics, new_columns=["acc", "mnemonic"], **opts)
    seq_family = regions.unique()
    joined = (
        seq_family.join(mnem, on="acc", how="left")
        .join(taxa.lazy(), on="mnemonic", how="left")
    )
    out = (
        joined.group_by("pfam_id")
        .agg(
            pl.len().alias("n_sequences"),
            pl.col("mnemonic").is_null().sum().alias("n_seq_not_in_pfamseq"),
            pl.col("kingdom").is_null().sum().alias("n_seq_no_kingdom"),
            pl.col("kingdom").filter(pl.col("kingdom").is_in(["A", "B", "E"]))
            .unique().sort().str.join("").alias("superkingdoms"),
            (pl.col("kingdom") == "V").sum().alias("n_seq_virus"),
            pl.col("phylum").drop_nulls().n_unique().alias("n_phyla"),
            (pl.col("kingdom").is_not_null() & pl.col("phylum").is_null())
            .sum().alias("n_seq_no_phylum"),
        )
        .with_columns(pl.col("superkingdoms").str.len_chars().alias("n_superkingdoms"))
        .sort("pfam_id")
        .collect(engine="streaming")
    )
    out.write_csv(args.out, separator="\t")
    print(out.describe())
    print(f"families: {out.height:,}")


if __name__ == "__main__":
    main()
