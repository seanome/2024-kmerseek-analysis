#!/usr/bin/env python3
"""Pull every Swiss-Prot entry with a BCL-2 family Pfam cross-reference.

Reads the UniProt flat file (``uniprot_sprot.dat.gz``) and keeps entries whose
``DR   Pfam;`` lines name PF00452 (Bcl-2) or PF02180 (BH4). Writes one TSV row per
entry with accession, entry name, species, NCBI taxon id, lineage, the Pfam families
found, and the sequence. Notebook 247 picks its query panel from this table.

Usage:
    python3 scripts/bcl2_family_from_swissprot.py \
        /Users/olga/data/uniprot/uniprot_sprot.dat.gz \
        tables/247_swissprot_bcl2_family_PF00452_PF02180.tsv
"""

import csv
import gzip
import sys

FIELDS = [
    "accession",
    "entry_name",
    "species",
    "taxid",
    "lineage",
    "pfam",
    "length",
    "sequence",
]

BCL2_PFAM = {"PF00452": "Bcl-2", "PF02180": "BH4"}


def parse_entries(path):
    """Yield one dict per flat-file entry, holding only the fields used here."""
    entry = None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "ID":
                if entry is not None:
                    raise ValueError("Incomplete Swiss-Prot entry before next ID")
                entry = {
                    "entry_name": line.split()[1],
                    "ac": [],
                    "os": [],
                    "oc": [],
                    "ox": "",
                    "pfam": [],
                    "seq": [],
                    "in_seq": False,
                }
            elif tag == "AC":
                entry["ac"].extend(a.strip() for a in line[5:].split(";") if a.strip())
            elif tag == "OS":
                entry["os"].append(line[5:].strip())
            elif tag == "OC":
                entry["oc"].append(line[5:].strip())
            elif tag == "OX" and not entry["ox"]:
                entry["ox"] = line[5:].split("=")[1].split()[0].rstrip(";")
            elif tag == "DR" and line.startswith("DR   Pfam;"):
                entry["pfam"].append(line.split(";")[1].strip())
            elif tag == "SQ":
                entry["in_seq"] = True
            elif tag == "//":
                yield entry
                entry = None
            elif entry is not None and entry["in_seq"]:
                entry["seq"].append(line.replace(" ", "").strip())

    if entry is not None:
        raise ValueError("Incomplete Swiss-Prot entry at end of file")


def main(dat_path, out_path):
    n_seen = 0
    rows = []
    for e in parse_entries(dat_path):
        n_seen += 1
        fams = sorted(set(e["pfam"]) & set(BCL2_PFAM))
        if not fams:
            continue
        rows.append(
            {
                "accession": e["ac"][0],
                "entry_name": e["entry_name"],
                "species": " ".join(e["os"]).rstrip("."),
                "taxid": e["ox"],
                "lineage": " ".join(e["oc"]).rstrip("."),
                "pfam": ",".join(fams),
                "length": len("".join(e["seq"])),
                "sequence": "".join(e["seq"]),
            }
        )
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(
        f"{n_seen} Swiss-Prot entries read, {len(rows)} with PF00452 or PF02180 -> {out_path}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
