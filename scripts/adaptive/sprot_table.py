"""One row per Swiss-Prot entry: accession, species, taxon id, lineage, Pfam families,
toxin keyword, and the EMBL coding-sequence protein ids (for fetching CDS from ENA)."""

import gzip
import sys

import polars as pl

rows = []
e = None
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        tag = line[:2]
        if tag == "ID":
            e = {
                "acc": None,
                "os": "",
                "ox": None,
                "oc": "",
                "pfam": [],
                "kw": "",
                "embl_cds": [],
                "len": int(line.split()[-2]),
            }
        elif tag == "AC" and e["acc"] is None:
            e["acc"] = line[5:].split(";")[0].strip()
        elif tag == "OS":
            e["os"] += line[5:].strip() + " "
        elif tag == "OX" and e["ox"] is None:
            e["ox"] = int(line.split("NCBI_TaxID=")[1].split(";")[0].split()[0])
        elif tag == "OC":
            e["oc"] += line[5:].strip() + " "
        elif tag == "KW":
            e["kw"] += line[5:].strip() + " "
        elif tag == "DR":
            p = [x.strip() for x in line[5:].rstrip(".\n").split(";")]
            if p[0] == "Pfam":
                e["pfam"].append(p[1])
            elif p[0] == "EMBL" and len(p) > 2 and p[2] != "-":
                e["embl_cds"].append(p[2])
        elif tag == "//":
            oc = e["oc"].replace(".", "").strip()
            rows.append(
                {
                    "accession": e["acc"],
                    "species": e["os"].strip().rstrip("."),
                    "taxid": e["ox"],
                    "lineage": oc,
                    "pfam": ";".join(e["pfam"]),
                    "is_toxin_kw": "Toxin" in e["kw"],
                    "embl_protein_ids": ";".join(e["embl_cds"]),
                    "length": e["len"],
                }
            )
pl.DataFrame(rows).write_parquet(sys.argv[2])
print(len(rows))
