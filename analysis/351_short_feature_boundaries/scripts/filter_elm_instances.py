"""Filter the ELM instance table to short features with both coordinates.

Keeps true-positive instances shorter than 60 residues whose start and end
fall inside the current UniProt sequence. Writes the kept instances, the query
protein FASTA, and a count of what each filter step removed.

Usage: python -I filter_elm_instances.py <elm_instances.tsv> <out_dir>
"""

import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

MAX_LEN = 60  # keep features with length < 60 residues
UNIPROT_FASTA = (
    "https://rest.uniprot.org/uniprotkb/accessions?accessions={}&format=fasta"
)
UNIPROT_ISOFORM = "https://rest.uniprot.org/uniprotkb/{}.fasta"


def read_elm(path):
    lines = (line for line in open(path) if not line.startswith("#"))
    return list(csv.DictReader(lines, delimiter="\t"))


def parse_fasta(text):
    seqs = {}
    name = None
    for line in io.StringIO(text):
        line = line.strip()
        if line.startswith(">"):
            # >sp|P12345|NAME_HUMAN ... or >sp|P12345-2|...
            name = line[1:].split("|")[1]
            seqs[name] = []
        elif name:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def fetch(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode()
        except Exception as err:  # retry network errors, then give up loudly
            print(f"retry {attempt + 1} for {url[:80]}: {err}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def fetch_sequences(accessions):
    canonical = sorted(a for a in accessions if "-" not in a)
    isoforms = sorted(a for a in accessions if "-" in a)
    seqs = {}
    for i in range(0, len(canonical), 100):
        seqs.update(
            parse_fasta(fetch(UNIPROT_FASTA.format(",".join(canonical[i : i + 100]))))
        )
    for acc in isoforms:
        seqs.update(parse_fasta(fetch(UNIPROT_ISOFORM.format(acc))))
    return seqs


def main(elm_path, out_dir):
    out_dir = Path(out_dir)
    rows = read_elm(elm_path)
    steps = [("all ELM instances", len(rows))]

    rows = [r for r in rows if r["InstanceLogic"] == "true positive"]
    steps.append(("true positive", len(rows)))

    rows = [
        r for r in rows if r["Start"].strip().isdigit() and r["End"].strip().isdigit()
    ]
    steps.append(("start and end both present", len(rows)))

    for r in rows:
        r["start"] = int(r["Start"])
        r["end"] = int(r["End"])
        r["length"] = r["end"] - r["start"] + 1
    rows = [r for r in rows if 0 < r["length"] < MAX_LEN]
    steps.append((f"length < {MAX_LEN} residues", len(rows)))

    seqs = fetch_sequences({r["Primary_Acc"] for r in rows})
    rows = [r for r in rows if r["Primary_Acc"] in seqs]
    steps.append(("sequence found in UniProt today", len(rows)))

    rows = [
        r for r in rows if r["start"] >= 1 and r["end"] <= len(seqs[r["Primary_Acc"]])
    ]
    steps.append(("coordinates inside the sequence", len(rows)))

    proteins = sorted({r["Primary_Acc"] for r in rows})
    steps.append(("distinct proteins", len(proteins)))

    cols = [
        "Accession",
        "ELMIdentifier",
        "ELMType",
        "Primary_Acc",
        "start",
        "end",
        "length",
        "Organism",
    ]
    with open(out_dir / "instances_kept.tsv", "w") as fh:
        fh.write(
            "\t".join(
                [
                    "instance_id",
                    "elm_class",
                    "elm_type",
                    "protein",
                    "start",
                    "end",
                    "length",
                    "organism",
                    "feature_residues",
                ]
            )
            + "\n"
        )
        for r in rows:
            residues = seqs[r["Primary_Acc"]][r["start"] - 1 : r["end"]]
            fh.write("\t".join(str(r[c]) for c in cols) + "\t" + residues + "\n")
    with open(out_dir / "query_proteins.fasta", "w") as fh:
        for acc in proteins:
            fh.write(f">{acc}\n{seqs[acc]}\n")
    with open(out_dir / "filter_counts.tsv", "w") as fh:
        fh.write("step\tcount\n")
        for name, n in steps:
            fh.write(f"{name}\t{n}\n")
            print(f"{name}\t{n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
