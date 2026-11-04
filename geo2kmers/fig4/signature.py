"""Get the signature's human proteins from UniProt, from the gene list in Ghosh et al. 2023.

Usage: signature.py <mmc2.xlsx> <out folder>
mmc2.xlsx is Supplementary Table 2 of Ghosh et al. 2023, sheet C13-14-3, from Europe PMC's supplementary
files for PMC10388732. NCBI_EMAIL names who is asking. Writes signature_proteins.tsv, with one row per
gene and whether its symbol resolved, and signature_proteins.fasta, named by gene.
"""

import csv
import os
import sys
import time
import urllib.parse
import urllib.request

import openpyxl

URL = "https://rest.uniprot.org/uniprotkb/search"
xlsx, out = sys.argv[1:]
rows = list(openpyxl.load_workbook(xlsx, read_only=True, data_only=True)["C13-14-3"].iter_rows(values_only=True))
weight_row = next(i for i, r in enumerate(rows) if any(str(c).startswith("Weight") for c in r if c))
columns = [i for i, c in enumerate(rows[weight_row]) if c]
clusters = dict(zip(columns, (13, 14, 3), strict=True))
weights = {clusters[i]: int(str(rows[weight_row][i]).split(":")[1]) for i in columns}
genes = [(clusters[i], str(r[i]).strip()) for r in rows[weight_row + 1 :] for i in columns if r[i]]


def protein_of(symbol):
    """The reviewed human protein whose gene name is this symbol, longest first: (accession, sequence, how many)."""
    query = f"(gene_exact:{symbol}) AND (organism_id:9606) AND (reviewed:true)"
    form = urllib.parse.urlencode({"query": query, "fields": "accession,sequence", "format": "tsv", "size": 25})
    request = urllib.request.Request(f"{URL}?{form}", headers={"User-Agent": f"geo2kmers {os.environ['NCBI_EMAIL']}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                lines = response.read().decode().splitlines()[1:]
            break
        except OSError:
            time.sleep(2**attempt)
    else:
        raise OSError(f"UniProt did not answer for {symbol}")
    found = sorted((line.split("\t") for line in lines), key=lambda x: (-len(x[1]), x[0]))
    return (*found[0], len(found)) if found else ("", "", 0)


with open(f"{out}/signature_proteins.tsv", "w", newline="") as t, open(f"{out}/signature_proteins.fasta", "w") as fasta:
    table = csv.writer(t, delimiter="\t", lineterminator="\n")
    table.writerow(
        ["gene", "cluster", "group", "weight_in_ghosh", "accession", "length_aa", "reviewed_entries", "resolved"]
    )
    written = set()  # a gene can sit in clusters 14 and 3: one table row each, one FASTA record
    for cluster, gene in genes:
        accession, sequence, hits = protein_of(gene)
        group = "M1" if cluster == 13 else "M2"
        table.writerow([gene, cluster, group, weights[cluster], accession, len(sequence), hits, bool(sequence)])
        if sequence and gene not in written:
            fasta.write(f">{gene}\n{sequence}\n")
            written.add(gene)
        time.sleep(0.2)
print(len(genes), "genes", {c: weights[c] for c in weights})
