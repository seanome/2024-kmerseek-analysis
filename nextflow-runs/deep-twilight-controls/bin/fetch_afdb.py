#!/usr/bin/env python3
"""Download the current AlphaFold DB model of each family protein.

The model version is read from the AlphaFold DB API at run time, never written into a URL
here, and each model's sequence is checked against the FASTA before it is used: Foldseek
coordinates are only UniProt coordinates when the two sequences are identical.

    fetch_afdb.py family.fasta outdir versions.tsv
"""

import json
import sys
import urllib.request
from pathlib import Path

family_fasta, outdir, versions = sys.argv[1:4]
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

seqs, acc = {}, None
for line in open(family_fasta):
    line = line.rstrip("\n")
    if line.startswith(">"):
        acc = line[1:].split()[0].split("|")[1]
        seqs[acc] = ""
    elif acc:
        seqs[acc] += line.strip()

rows = ["accession\tmodel\tlatest_version\tmean_plddt\tsequence_matches_uniprot"]
for acc, seq in seqs.items():
    with urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}") as r:
        entry = next(e for e in json.load(r) if e.get("uniprotAccession") == acc)
    model_seq = entry.get("sequence") or entry.get("uniprotSequence")
    if model_seq != seq:
        sys.exit(f"{acc}: AlphaFold DB model sequence differs from the UniProt sequence")
    urllib.request.urlretrieve(entry["cifUrl"], outdir / f"AF-{acc}-F1.cif")
    rows.append(
        f"{acc}\t{entry['cifUrl'].rsplit('/', 1)[-1]}\t{entry.get('latestVersion')}\t"
        f"{entry.get('globalMetricValue')}\tTrue"
    )
Path(versions).write_text("\n".join(rows) + "\n")
print(f"{len(seqs)} models")
