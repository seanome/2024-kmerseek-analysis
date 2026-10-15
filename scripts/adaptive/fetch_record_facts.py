#!/usr/bin/env python3
"""Fetch and cache the outside records notebook 292's text cites, so every fact has a
committed source.

Writes into <out_dir>:
  sra_sample_records.csv    library kit, flow cell, instrument, tissue and read count of the
                            runs section 1 discusses (NCBI EDirect efetch, SRA XML)
  interpro_pf02950_matches.csv  where Pfam PF02950 matches the tarantula reference proteins
                            (InterPro API)
  tsa_selenobrachys.csv     Selenobrachys transcriptome assemblies in GenBank (EDirect docsum)
  protein_selenobrachys.csv every Selenobrachys protein record in Entrez (EDirect docsum)
Each file records the date it was fetched. Re-running overwrites them.

Usage: fetch_record_facts.py <out_dir>
"""

import csv
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

OUT = sys.argv[1]
EDIRECT = "/Users/olga/anaconda3/envs/orthofinder/bin"
TODAY = datetime.date.today().isoformat()

# Section 1: the chosen run, the other venom-gland runs, the marine cnidarian runs, the
# king cobra run, and every run in the search larger than the chosen one.
SRA_RUNS = [
    "SRR32537183",
    "SRR33222190",
    "SRR39024076",
    "SRR32136522",
    "SRR12283963",
    "SRR32629518",
    "SRR24236433",
    "ERR15726137",
    "SRR29180535",
    "SRR32174933",
    "SRR32174867",
    "SRR35247654",
    "SRR40903635",
    "SRR20123506",
    "SRR27322302",
    "SRR21621151",
    "SRR36000876",
    "SRR15213155",
]
# Section 2: tarantula proteins in the reference.
TARANTULA_REFS = [
    "B1P1F1",
    "B1P1F3",
    "B1P1F4",
    "B1P1F7",
    "D2Y2H6",
    "D2Y2C3",
    "P0CH56",
    "P0DM66",
]


def efetch(db, acc, fmt):
    env = dict(os.environ, PATH=EDIRECT + ":" + os.environ["PATH"])
    for _ in range(3):
        r = subprocess.run(
            ["efetch", "-db", db, "-id", acc, "-format", fmt],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        if r.stdout.strip().startswith("<"):
            return r.stdout
    raise RuntimeError(f"efetch {db} {acc} returned no XML")


def flow_cell(text):
    m = re.search(r"FLO-[A-Z0-9]+|R9(?:\.4(?:\.1)?)?|R10(?:\.4(?:\.1)?)?", text)
    return m.group(0) if m else ""


def kit(text):
    m = re.search(r"SQK-[A-Z0-9]+", text)
    return m.group(0) if m else ""


rows = []
for run in SRA_RUNS:
    t = ET.fromstring(efetch("sra", run, "xml"))
    attrs = {s.findtext("TAG"): s.findtext("VALUE") for s in t.iter("SAMPLE_ATTRIBUTE")}
    design = " ".join(e.text or "" for e in t.iter("DESIGN_DESCRIPTION"))
    run_el = next(e for e in t.iter("RUN") if e.get("accession") == run)
    rows.append(
        {
            "run": run,
            "species": next((e.text for e in t.iter("SCIENTIFIC_NAME")), ""),
            "tissue": attrs.get("tissue", ""),
            "dev_stage": attrs.get("dev_stage", ""),
            "instrument": next((e.text for e in t.iter("INSTRUMENT_MODEL")), ""),
            "library_kit_in_record": kit(design),
            "flow_cell_in_record": flow_cell(design),
            "n_reads": run_el.get("total_spots", ""),
            "published": run_el.get("published", ""),
            "design_description": design[:300],
            "fetched": TODAY,
        }
    )
with open(os.path.join(OUT, "sra_sample_records.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

rows = []
for acc in TARANTULA_REFS:
    with urllib.request.urlopen(
        f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{acc}",
        timeout=60,
    ) as r:
        d = json.load(r)
    for res in d["results"]:
        if res["metadata"]["accession"] != "PF02950":
            continue
        for p in res["proteins"]:
            for loc in p["entry_protein_locations"]:
                for frag in loc["fragments"]:
                    rows.append(
                        {
                            "accession": acc,
                            "pfam": "PF02950",
                            "match_start": frag["start"],
                            "match_end": frag["end"],
                            "protein_length": p["protein_length"],
                            "source": "InterPro API entry/pfam/protein/uniprot",
                            "fetched": TODAY,
                        }
                    )
with open(os.path.join(OUT, "interpro_pf02950_matches.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

env = dict(os.environ, PATH=EDIRECT + ":" + os.environ["PATH"])
xml = subprocess.run(
    f'esearch -db nuccore -query "Selenobrachys[Organism] AND tsa-master[prop]" | efetch -format docsum',
    shell=True,
    capture_output=True,
    text=True,
    env=env,
    timeout=180,
).stdout
t = ET.fromstring(xml)
rows = []
for ds in t.iter("DocumentSummary"):
    rows.append(
        {
            "accession": ds.findtext("Caption"),
            "title": ds.findtext("Title"),
            "slen_docsum": ds.findtext("Slen"),
            "created": ds.findtext("CreateDate"),
            "query": "Selenobrachys[Organism] AND tsa-master[prop]",
            "fetched": TODAY,
        }
    )
with open(os.path.join(OUT, "tsa_selenobrachys.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
xml = subprocess.run(
    'esearch -db protein -query "Selenobrachys[Organism]" | efetch -format docsum',
    shell=True,
    capture_output=True,
    text=True,
    env=env,
    timeout=180,
).stdout
t = ET.fromstring(xml)
rows = [
    {
        "accession": ds.findtext("Caption"),
        "title": ds.findtext("Title"),
        "length": ds.findtext("Slen"),
        "query": "Selenobrachys[Organism]",
        "fetched": TODAY,
    }
    for ds in t.iter("DocumentSummary")
]
with open(os.path.join(OUT, "protein_selenobrachys.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(
    "wrote sra_sample_records.csv, interpro_pf02950_matches.csv, tsa_selenobrachys.csv, protein_selenobrachys.csv"
)
