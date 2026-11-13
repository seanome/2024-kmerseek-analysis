#!/usr/bin/env python3
"""Write nextflow-runs/proteome-novelty/assets/proteomes.tsv for notebook 345.

One row per QfO 2020_04 reference proteome: UniProt proteome id, NCBI taxon id, the
species name from the release README, the NCBI lineage (from a local taxdump), the number
of sequences and residues in the canonical FASTA, and the clades the proteome belongs to.

The sequence and residue counts let the pipeline check that the FASTA it reads on
Sherlock is the same file this table was built from.

Usage:
    python scripts/nb345_build_proteome_table.py \
        --qfo-dir ~/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143 \
        --taxdump ~/data/taxonomy \
        --out nextflow-runs/proteome-novelty/assets/proteomes.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path

# Clades are NCBI taxa, fixed here before any word was counted. A clade needs at least six
# proteomes, so that hiding the final 20% (rounded up) leaves at least four to fit on.
CLADES = {
    "Archaea": 2157,
    "Bacteria": 2,
    "Eukaryota": 2759,
    "Vertebrata": 7742,
    "Mammalia": 40674,
    "Fungi": 4751,
}
MIN_CLADE_SIZE = 6


def read_taxdump(taxdump: Path, wanted: set[int]) -> tuple[dict[int, int], dict[int, str]]:
    parent: dict[int, int] = {}
    with open(taxdump / "nodes.dmp") as fh:
        for line in fh:
            f = line.split("\t|\t", 2)
            parent[int(f[0])] = int(f[1])
    lineage_ids: set[int] = set()
    for t in wanted:
        while t not in lineage_ids and t in parent:
            lineage_ids.add(t)
            if parent[t] == t:
                break
            t = parent[t]
    names: dict[int, str] = {}
    with open(taxdump / "names.dmp") as fh:
        for line in fh:
            f = line.rstrip("\t|\n").split("\t|\t")
            tid = int(f[0])
            if tid in lineage_ids and f[3] == "scientific name":
                names[tid] = f[1]
    return parent, names


def read_merged(taxdump: Path) -> dict[int, int]:
    """Old taxon id -> current id. USTMA's 237631 was merged into 5270 after 2020."""
    with zipfile.ZipFile(taxdump / "taxdmp.zip") as z:
        text = z.read("merged.dmp").decode()
    merged = {}
    for line in text.splitlines():
        old_id, new_id = line.split("\t|\t")[:2]
        merged[int(old_id)] = int(new_id.rstrip("\t|"))
    return merged


def lineage(tid: int, parent: dict[int, int]) -> list[int]:
    out = [tid]
    while parent.get(tid, tid) != tid:
        tid = parent[tid]
        out.append(tid)
    return out[::-1]


def fasta_counts(path: Path) -> tuple[int, int]:
    n_seq = n_res = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                n_seq += 1
            else:
                n_res += len(line.strip())
    return n_seq, n_res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qfo-dir", type=Path, required=True)
    ap.add_argument("--taxdump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # README rows: Proteome_ID Tax_ID OSCODE domain n_canonical n_isoforms n_gene2acc Species
    row_re = re.compile(r"^(UP\d+)\t(\d+)\t(\w+)\t(\w+)\t\d+\t\d+\t\d+\t(.+)$")
    rows = []
    for line in (args.qfo_dir / "README").read_text().splitlines():
        m = row_re.match(line)
        if m:
            upid, tax, oscode, domain, species = m.groups()
            rows.append(dict(proteome=upid, taxid=int(tax), oscode=oscode, domain=domain, species=species))
    if len(rows) != 78:
        raise SystemExit(f"expected 78 proteomes in the README, found {len(rows)}")

    merged = read_merged(args.taxdump)
    for r in rows:
        r["ncbi_taxid"] = merged.get(r["taxid"], r["taxid"])
    parent, names = read_taxdump(args.taxdump, {r["ncbi_taxid"] for r in rows})
    for r in rows:
        if r["ncbi_taxid"] not in parent:
            raise SystemExit(f"{r['oscode']}: taxon {r['ncbi_taxid']} not in the taxdump")
        lin = lineage(r["ncbi_taxid"], parent)
        r["lineage"] = "; ".join(names.get(t, str(t)) for t in lin)
        r["clades"] = ",".join(c for c, t in CLADES.items() if t in lin)
        sub = {"eukaryota": "Eukaryota", "bacteria": "Bacteria", "archaea": "Archaea"}[r["domain"]]
        fasta = args.qfo_dir / sub / f"{r['proteome']}_{r['taxid']}.fasta"
        r["fasta"] = f"{sub}/{fasta.name}"
        r["n_sequences"], r["n_residues"] = fasta_counts(fasta)

    sizes = {c: sum(c in r["clades"].split(",") for r in rows) for c in CLADES}
    small = [c for c, n in sizes.items() if n < MIN_CLADE_SIZE]
    if small:
        raise SystemExit(f"clades below {MIN_CLADE_SIZE} proteomes: {small} ({sizes})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["proteome", "taxid", "ncbi_taxid", "oscode", "domain", "species", "fasta", "n_sequences", "n_residues", "clades", "lineage"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["domain"], r["lineage"])):
            w.writerow({c: r[c] for c in cols})
    print("clade sizes:", sizes)


if __name__ == "__main__":
    main()
