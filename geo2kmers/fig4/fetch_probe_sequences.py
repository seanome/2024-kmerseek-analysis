"""Fetch the proteins of each probe set's transcripts, and the EST sequences of probe sets with no transcript, from NCBI.

Usage: fetch_probe_sequences.py <out folder> <sources.tsv> ...
sources.tsv is probe_set_sources.py's. NCBI_EMAIL must be set. refseq_mrna, refseq_model and genbank_mrna accessions
give the translated coding regions NCBI annotates on them (rettype fasta_cds_aa); est accessions give any protein annotated on them and the sequence itself
(rettype fasta), untranslated. ensembl_coding accessions are not in NCBI and are left for another step. Writes
proteins.fasta (one record per protein_id, the header naming its transcript), est.fasta, and fetched.tsv with one row per
accession asked for: what came back and how long it is.
"""

import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH = 200
PROTEIN_SOURCES = {"refseq_mrna", "refseq_model", "genbank_mrna"}
HEADER = re.compile(r"^>lcl\|([A-Z]+_?\d+)\.\d+_prot_.*?\[protein_id=([^\]]+)\]")

out, *tables = sys.argv[1:]
email = os.environ["NCBI_EMAIL"]
wanted = {"protein": set(), "est": set()}
for table in tables:
    with open(table) as f:
        next(f)
        for line in f:
            _, source, accessions, _ = line.rstrip("\n").split("\t")
            ids = [a for a in accessions.split(",") if a]
            # Many U133 consensus probe sets were designed on a full-length cDNA, so an est accession is asked for
            # its annotated protein as well as its own sequence
            if source in PROTEIN_SOURCES or source == "est":
                wanted["protein"].update(ids)
            if source == "est":
                wanted["est"].update(ids)


def fetch(ids, rettype):
    form = urllib.parse.urlencode(
        {
            "db": "nuccore",
            "id": ",".join(ids),
            "rettype": rettype,
            "retmode": "text",
            "email": email,
            "tool": "geo2kmers",
        }
    ).encode()
    for attempt in range(4):
        try:
            return urllib.request.urlopen(URL, form, timeout=300).read().decode()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 3:
                raise
            print(f"retrying a batch after {error}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))


def records(text):
    for block in text.split("\n>"):
        block = block.strip()
        if block:
            header, *lines = block.lstrip(">").split("\n")
            yield ">" + header, "".join(lines)


got = {"protein": {}, "est": {}}
with open(f"{out}/proteins.fasta", "w") as proteins, open(f"{out}/est.fasta", "w") as ests:
    for kind, rettype, handle in (("protein", "fasta_cds_aa", proteins), ("est", "fasta", ests)):
        ids = sorted(wanted[kind])
        for i in range(0, len(ids), BATCH):
            for header, sequence in records(fetch(ids[i : i + BATCH], rettype)):
                if kind == "protein":
                    match = HEADER.match(header)
                    if not match:
                        continue
                    accession, protein = match.groups()
                    handle.write(f">{protein} {accession}\n{sequence}\n")
                    got["protein"].setdefault(accession, []).append((protein, len(sequence)))
                else:
                    accession = header[1:].split()[0].split(".")[0]
                    handle.write(f">{accession}\n{sequence}\n")
                    got["est"].setdefault(accession, []).append(("est", len(sequence)))
            time.sleep(0.34)
            print(f"{kind}: {min(i + BATCH, len(ids))} of {len(ids)}", file=sys.stderr, flush=True)
with open(f"{out}/fetched.tsv", "w") as f:
    f.write("accession\tasked_for\tsequence\tlength\n")
    for kind in ("protein", "est"):
        for accession in sorted(wanted[kind]):
            for name, length in got[kind].get(accession, [("none", 0)]):
                f.write(f"{accession}\t{kind}\t{name}\t{length}\n")
