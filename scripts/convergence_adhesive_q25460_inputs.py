"""Inputs for the adhesive case study: mussel foot protein 1 (Q25460) as the one query.

The convergence pipeline (nextflow-runs/convergence-afp) reads its queries, truth set and
query structures from one folder, --truth_dir. This script fills data/convergence_adhesive
so that folder can be passed as it is:

  queries.tsv            Q25460 only, with the columns of data/convergence_afp/queries.tsv
  extra_reference.fasta  truth entries that are not in reviewed Swiss-Prot (the sandcastle
                         worm cement proteins and one barnacle entry, all TrEMBL), taken
                         from truth_set.tsv, so every truth entry is in the searched
                         database. The Makefile appends them to Swiss-Prot 2026_03.
  structures/            the AlphaFold DB model of Q25460, for the Foldseek structure
                         query, and structures.tsv with its URL and MD5

Run from the repository root:
  python scripts/convergence_adhesive_q25460_inputs.py
"""

import argparse
import datetime
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import polars as pl

QUERY = "Q25460"
QUERY_COLUMNS = [
    "lineage",
    "Entry",
    "Entry Name",
    "Protein names",
    "Organism",
    "Organism (ID)",
    "Taxonomic lineage",
    "Protein families",
    "Sequence similarities",
    "Keywords",
    "Gene Ontology IDs",
    "Length",
    "matched_by",
    "n_experimental_evidence",
]
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=Path("data/convergence_adhesive"))
    args = ap.parse_args()

    truth = pl.read_csv(args.dir / "truth_set.tsv", separator="\t")
    q = truth.filter(pl.col("Entry") == QUERY)
    if q.height != 1:
        sys.exit(f"{QUERY} is in truth_set.tsv {q.height} times, not once")
    if q["Reviewed"][0] != "reviewed":
        sys.exit(f"{QUERY} is not a Swiss-Prot entry")
    if len(q["Sequence"][0]) != q["Length"][0]:
        sys.exit(f"{QUERY}: sequence length differs from Length")
    q.select(QUERY_COLUMNS).write_csv(args.dir / "queries.tsv", separator="\t")

    extra = truth.filter(pl.col("Reviewed") != "reviewed")
    with open(args.dir / "extra_reference.fasta", "w") as fh:
        for r in extra.iter_rows(named=True):
            if len(r["Sequence"]) != r["Length"]:
                sys.exit(f"{r['Entry']}: sequence length differs from Length")
            fh.write(f">tr|{r['Entry']}|{r['Entry Name']}\n{r['Sequence']}\n")

    sdir = args.dir / "structures"
    sdir.mkdir(exist_ok=True)
    meta = json.loads(fetch(AFDB_API.format(acc=QUERY)))
    model = next(m for m in meta if m["uniprotAccession"] == QUERY)
    cif_url = model["cifUrl"]
    cif = fetch(cif_url)
    name = cif_url.rsplit("/", 1)[1]
    (sdir / name).write_bytes(cif)
    row = {
        "accession": QUERY,
        "api_url": AFDB_API.format(acc=QUERY),
        "cif_url": cif_url,
        "model_version": model["latestVersion"],
        "uniprot_start": model["uniprotStart"],
        "uniprot_end": model["uniprotEnd"],
        "md5": hashlib.md5(cif).hexdigest(),
        "fetched_on": datetime.date.today().isoformat(),
    }
    pl.DataFrame([row]).write_csv(sdir / "structures.tsv", separator="\t")

    print(f"queries.tsv: {QUERY}, {q['Length'][0]} residues, lineage {q['lineage'][0]}")
    print(
        f"extra_reference.fasta: {extra.height} entries not in Swiss-Prot: "
        + ", ".join(f"{e} ({l})" for e, l in zip(extra["Entry"], extra["lineage"]))
    )
    print(
        f"structure: {name}, residues {row['uniprot_start']}-{row['uniprot_end']}, md5 {row['md5']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
