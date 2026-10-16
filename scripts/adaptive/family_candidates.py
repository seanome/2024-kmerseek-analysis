#!/usr/bin/env python3
"""Step 1: list candidate toxin Pfam families and what each held-out reference holds.

A candidate is a Pfam family carried by a Swiss-Prot entry with the keyword "Toxin" that is
the best DIAMOND hit of at least one sampled read. For each: Swiss-Prot entries carrying it,
entries outside the sample's genus and outside its taxonomic family, entries with an EMBL
protein id (for the coding sequence), and reads whose best hit carries it. Pfam names come
from the InterPro API and are cached in <out_dir>/pfam_names.json.

Usage: family_candidates.py <sprot_table.parquet> <reads_truth.parquet> <genus> <family> <ref_dir> <out_dir>
"""

import json
import os
import sys
import urllib.request

import polars as pl

table, truth, genus, family, ref_dir, out = sys.argv[1:7]
t = pl.read_parquet(table)
tr = pl.read_parquet(truth)
toxin_hits = tr.filter(pl.col("best_hit_is_toxin") == True)  # noqa: E712
fams = (
    toxin_hits.with_columns(pl.col("best_hit_pfam").str.split(";"))
    .explode("best_hit_pfam")
    .filter(pl.col("best_hit_pfam") != "")["best_hit_pfam"]
    .unique()
    .sort()
    .to_list()
)

names_path = os.path.join(out, "pfam_names.json")
names = json.load(open(names_path)) if os.path.exists(names_path) else {}
for f in fams:
    if f not in names:
        with urllib.request.urlopen(
            f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{f}", timeout=60
        ) as r:
            names[f] = json.load(r)["metadata"]["name"]["short"]
json.dump(names, open(names_path, "w"), indent=1, sort_keys=True)

rows = []
reads_by_fam = (
    tr.with_columns(pl.col("best_hit_pfam").str.split(";"))
    .explode("best_hit_pfam")
    .group_by("best_hit_pfam")
    .agg(n=pl.len())
)
reads_by_fam = dict(zip(reads_by_fam["best_hit_pfam"], reads_by_fam["n"]))
for f in fams:
    e = t.filter(pl.col("pfam").str.split(";").list.contains(f))
    lin = pl.col("lineage").str.split("; ")
    rows.append(
        {
            "pfam": f,
            "pfam_name": names[f],
            "n_sprot_entries": e.height,
            "n_sprot_toxin_keyword": e["is_toxin_kw"].sum(),
            "n_outside_genus": e.filter(
                ~lin.list.contains(genus)
                & ~pl.col("species").str.starts_with(genus + " ")
            ).height,
            "n_outside_family": e.filter(~lin.list.contains(family)).height,
            "n_with_embl_protein_id": e.filter(pl.col("embl_protein_ids") != "").height,
            "n_reads_best_hit": reads_by_fam.get(f, 0),
        }
    )
pl.DataFrame(rows).sort("n_sprot_entries", descending=True).write_csv(
    os.path.join(out, "toxin_family_candidates.csv")
)

comp = []
for lvl in ["ii", "iii"]:
    accs = [
        l[1:].strip()
        for l in open(os.path.join(ref_dir, f"ref_{lvl}.fasta"))
        if l.startswith(">")
    ]
    d = t.filter(pl.col("accession").is_in(accs)).with_columns(
        genus=pl.col("species").str.split(" ").list.get(0),
        taxonomic_family=pl.col("lineage")
        .str.split("; ")
        .list.eval(pl.element().filter(pl.element().str.ends_with("idae")))
        .list.last(),
    )
    comp.append(
        d.group_by("taxonomic_family", "genus")
        .agg(n_proteins=pl.len())
        .with_columns(holdout=pl.lit(lvl))
    )
pl.concat(comp).select("holdout", "taxonomic_family", "genus", "n_proteins").sort(
    "holdout", "n_proteins", descending=[False, True]
).write_csv(os.path.join(out, "reference_composition_PF02950_sprot_2026_03.csv"))
print(pl.read_csv(os.path.join(out, "toxin_family_candidates.csv")))
print(pl.read_csv(os.path.join(out, "reference_composition_PF02950_sprot_2026_03.csv")))
