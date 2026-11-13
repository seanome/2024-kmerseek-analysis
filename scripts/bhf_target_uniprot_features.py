#!/usr/bin/env python3
"""UniProt keywords and feature positions for every target of BHF's kmerseek regions.

    python3 bhf_target_uniprot_features.py /Users/olga/data/invertebrate-dark-set/bhf-view

Reads the target accessions of bhf_hp_lehninger2_k{17,19,21,23}_s1.csv in that folder and
writes target_uniprot_features.tsv next to them: accession, protein name, keywords, and the
zinc finger, transmembrane, intramembrane, DNA-binding and lipid-anchor features with their
positions (UniProt 1-based, inclusive). Asks UniProt's REST API 100 accessions at a time (300 made the URL too long: HTTP 400)
and refuses an answer that is not a table (an outage page) or that is missing accessions.
Added 2026-10-13 for notebook 279 (PR 120), to ask whether BHF's regions land on a zinc
finger, a membrane anchor or a DNA-binding region of their targets.
"""

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

FIELDS = "accession,protein_name,keyword,ft_zn_fing,ft_transmem,ft_intramem,ft_dna_bind,ft_lipid"
BATCH = 100


def main():
    folder = Path(sys.argv[1])
    accs = set()
    for k in (17, 19, 21, 23):
        accs |= set(pl.read_csv(folder / f"bhf_hp_lehninger2_k{k}_s1.csv", columns=["target_name"])["target_name"])
    accs = sorted(accs)
    out = folder / "target_uniprot_features.tsv"
    header, rows = None, []
    for i in range(0, len(accs), BATCH):
        chunk = accs[i:i + BATCH]
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": " OR ".join(f"accession:{a}" for a in chunk), "fields": FIELDS, "format": "tsv", "size": 500})
        for attempt in range(4):
            try:
                text = urllib.request.urlopen(url, timeout=180).read().decode()
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        if not text.startswith("Entry\t"):
            raise ValueError(f"UniProt did not return a table: {text[:200]}")
        lines = text.rstrip("\n").split("\n")
        header = lines[0]
        rows += lines[1:]
        print(f"{i + len(chunk):_} of {len(accs):_}", flush=True)
    found = {r.split("\t")[0] for r in rows}
    missing = sorted(set(accs) - found)
    if missing:
        raise ValueError(f"{len(missing)} accessions missing from UniProt's answer, e.g. {missing[:5]}")
    out.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"wrote {out}: {len(rows):_} rows")


if __name__ == "__main__":
    main()
