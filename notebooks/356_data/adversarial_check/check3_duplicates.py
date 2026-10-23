"""Check 3: no exon interval counted twice in the verdict set."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import *

D = "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota"
v = verdict_exons()
print("verdict rows", v.height)
print("unique exon_id:", v["exon_id"].n_unique())
print(
    "unique (species,target,start,end):",
    v.unique(["species", "target_acc", "prot_start", "prot_end"]).height,
)
print(
    "transcripts per target (max):",
    v.group_by("species", "target_acc")
    .agg(pl.col("transcript").n_unique())["transcript"]
    .max(),
)
pr = pairs()
print(
    "pairs rows",
    pr.height,
    "unique (species,target)",
    pr.select("species", "target_acc").unique().height,
    "unique (species,query)",
    pr.select("species", "query_acc").unique().height,
)
# overlap between exons of the same target beyond the shared split codon
s = v.sort("species", "target_acc", "prot_start")
s = s.with_columns(
    pl.col("prot_end").shift(1).over("species", "target_acc").alias("prev_end")
)
ov = s.filter(
    pl.col("prev_end").is_not_null() & (pl.col("prev_end") - pl.col("prot_start") > 1)
)
print(
    "verdict exons overlapping the previous verdict exon of the same target by >1 residue:",
    ov.height,
)


# identical target protein sequences (two UniProt entries, one gene) would count an exon twice
def read_fasta(path):
    out, name, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                out[name] = "".join(buf)
            name = line.split("|")[1]
            buf = []
        else:
            buf.append(line.strip())
    out[name] = "".join(buf)
    return out


for sp, stem in {
    "chicken": "UP000000539_9031",
    "zebrafish": "UP000000437_7955",
}.items():
    f = read_fasta(f"{D}/{stem}.fasta")
    t = v.filter(pl.col("species") == sp)
    seqs = [f[a] for a in t["target_acc"].unique()]
    print(
        sp,
        "verdict targets",
        len(seqs),
        "distinct sequences",
        len(set(seqs)),
        "distinct exon peptides",
        t.with_columns(
            pl.struct("target_acc", "prot_start", "prot_end")
            .map_elements(
                lambda r: f[r["target_acc"]][r["prot_start"] : r["prot_end"]],
                return_dtype=pl.String,
            )
            .alias("pep")
        )["pep"].n_unique(),
        "of",
        t.height,
    )
# genes
print(
    "unique genes in verdict:",
    v.select("species", "gene").unique().height,
    "targets:",
    v.select("species", "target_acc").unique().height,
)
