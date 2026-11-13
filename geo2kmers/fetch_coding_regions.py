"""Write each record's annotated coding region, from NCBI's feature tables.

Usage: fetch_coding_regions.py <GPL table> <out.tsv>
NCBI_EMAIL must be set. Coordinates are 1-based. codon_start is 2 or 3 when the record starts inside a codon.
"""

import os
import re
import sys
import time
import urllib.parse
import urllib.request

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CDS = re.compile(r"^<?(\d+)\t>?(\d+)\tCDS\n((?:\t\t\t.*\n)*)", re.MULTILINE)
accessions = set()
with open(sys.argv[1]) as f:
    for line in f:
        cells = line.rstrip("\n").split("\t")
        if len(cells) > 1 and cells[0].isdigit() and cells[1]:
            accessions.add(cells[1].split(".")[0])
accessions = sorted(accessions)
with open(sys.argv[2], "w") as out:
    out.write("accession\tstart\tend\tcodon_start\n")
    for i in range(0, len(accessions), 200):
        form = {"db": "nuccore", "id": ",".join(accessions[i : i + 200]), "rettype": "ft", "retmode": "text"}
        form["email"] = os.environ["NCBI_EMAIL"]
        text = urllib.request.urlopen(URL, urllib.parse.urlencode(form).encode(), timeout=120).read().decode()
        for block in text.split(">Feature")[1:]:
            name = re.search(r"([A-Z]+_?\d+)(\.\d+)?\|?\s*$", block.split("\n")[0].strip())
            if name is None:
                continue
            for match in CDS.finditer(block):
                codon = re.search(r"codon_start\t(\d)", match.group(3))
                start = codon.group(1) if codon else 1
                out.write(f"{name.group(1)}\t{match.group(1)}\t{match.group(2)}\t{start}\n")
        time.sleep(0.34)
