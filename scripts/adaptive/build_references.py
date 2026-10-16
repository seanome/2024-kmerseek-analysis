#!/usr/bin/env python3
"""Step 1: the target family's Swiss-Prot proteins at three hold-out levels, and their CDS.

Writes, for Pfam family --pfam:
  ref_i.fasta    every Swiss-Prot entry carrying the family (all genera)
  ref_ii.fasta   all but the sample's genus
  ref_iii.fasta  all but the sample's taxonomic family
  cds_i/ii/iii.fasta  the coding sequence of each entry from ENA, by the first EMBL protein id
                      in the entry's cross-references; entries without one are counted
  reference_counts.tsv, cds_fetch_log.tsv
ENA responses are cached under <out>/ena_cache so re-runs do not refetch.
"""

import argparse
import gzip
import os
import time
import urllib.request

import polars as pl

p = argparse.ArgumentParser()
p.add_argument("--table", required=True, help="sprot_table.py parquet")
p.add_argument("--fasta", required=True, help="Swiss-Prot FASTA (same release)")
p.add_argument("--pfam", required=True)
p.add_argument("--genus", required=True, help="sample genus, held out at level ii")
p.add_argument(
    "--family", required=True, help="sample taxonomic family, held out at level iii"
)
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(os.path.join(a.out, "ena_cache"), exist_ok=True)

t = pl.read_parquet(a.table).filter(pl.col("pfam").str.contains(a.pfam))
lineage_words = pl.col("lineage").str.split("; ")
levels = {
    "i": t,
    "ii": t.filter(
        ~lineage_words.list.contains(a.genus)
        & ~pl.col("species").str.starts_with(a.genus + " ")
    ),
    "iii": t.filter(~lineage_words.list.contains(a.family)),
}

seqs = {}
with gzip.open(a.fasta, "rt") as f:
    acc, buf = None, []
    for line in f:
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            acc, buf = line.split("|")[1], []
        else:
            buf.append(line.strip())
    seqs[acc] = "".join(buf)


def fetch_cds(protein_id):
    path = os.path.join(a.out, "ena_cache", protein_id + ".fasta")
    if not os.path.exists(path):
        url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{protein_id}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    body = r.read().decode()
                break
            except Exception as e:  # noqa: BLE001
                body = ""
                time.sleep(2)
        with open(path, "w") as fo:
            fo.write(body)
    body = open(path).read()
    if not body.startswith(">"):
        return None
    return "".join(body.splitlines()[1:])


log = []
cds = {}
for row in t.iter_rows(named=True):
    pid = row["embl_protein_ids"].split(";")[0] if row["embl_protein_ids"] else ""
    s = fetch_cds(pid) if pid else None
    log.append(
        {
            "accession": row["accession"],
            "embl_protein_id": pid,
            "cds_found": s is not None,
            "cds_length": len(s) if s else 0,
            "protein_length": row["length"],
        }
    )
    if s:
        cds[row["accession"]] = s
pl.DataFrame(log).write_csv(os.path.join(a.out, "cds_fetch_log.tsv"), separator="\t")

counts = []
for lvl, d in levels.items():
    accs = d["accession"].to_list()
    with open(os.path.join(a.out, f"ref_{lvl}.fasta"), "w") as fo:
        for x in accs:
            fo.write(f">{x}\n{seqs[x]}\n")
    with open(os.path.join(a.out, f"cds_{lvl}.fasta"), "w") as fo:
        for x in accs:
            if x in cds:
                fo.write(f">{x}\n{cds[x]}\n")
    counts.append(
        {
            "holdout": lvl,
            "n_proteins": len(accs),
            "n_with_cds": sum(x in cds for x in accs),
            "n_genera": d.with_columns(g=pl.col("species").str.split(" ").list.get(0))[
                "g"
            ].n_unique(),
        }
    )
pl.DataFrame(counts).write_csv(
    os.path.join(a.out, "reference_counts.tsv"), separator="\t"
)
print(pl.DataFrame(counts))
