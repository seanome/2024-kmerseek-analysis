#!/usr/bin/env python3
"""Search PRIDE for a public human gut metaproteomics dataset that ships a de novo peptide list.

Usage: pride_denovo_search.py <out_dir>

1. Pull every project the PRIDE archive search returns for the keywords metaproteomics,
   metaproteome, microbiome, fecal, stool and gut (PRIDE API v3, 100 per page).
2. Keep projects whose title, description, protocols, keywords, organisms or organism parts
   mention the gut (gut, fecal, faecal, stool, feces, faeces, intestin, colon) and a microbial
   community (metaprote, microbio, microbiota).
3. List every file of those projects whose name mentions novo, denovo, casanovo, novor, ALC or
   peaks, the names PEAKS, Novor and Casanovo give their de novo outputs.

Writes pride_gut_projects.tsv (accession, title, organisms) and pride_denovo_files.tsv
(accession, file, bytes). Run on 2026-10-21 for notebook 300; that run's tables are committed
in notebooks/300_data/step1/.
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

API = "https://www.ebi.ac.uk/pride/ws/archive/v3"
KEYWORDS = ["metaproteomics", "metaproteome", "microbiome", "fecal", "stool", "gut"]
GUT = re.compile(r"\b(gut|fecal|faecal|stool|feces|faeces|intestin|colon)")
COMMUNITY = re.compile(r"metaprote|microbio|microbiota")
DENOVO_FILE = re.compile(r"novo|denovo|casanovo|novor|ALC|peaks", re.I)
TEXT_FIELDS = [
    "title",
    "projectDescription",
    "sampleProcessingProtocol",
    "dataProcessingProtocol",
    "keywords",
    "organisms",
    "organismsPart",
]


def get(url):
    for attempt in range(3):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception:
            time.sleep(5)
    return []


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    projects = {}
    for kw in KEYWORDS:
        for page in range(50):
            batch = get(f"{API}/search/projects?keyword={kw}&pageSize=100&page={page}")
            for p in batch:
                projects[p["accession"]] = p
            if len(batch) < 100:
                break
    gut = []
    for acc, p in sorted(projects.items()):
        text = " ".join(str(p.get(k) or "") for k in TEXT_FIELDS).lower()
        if GUT.search(text) and COMMUNITY.search(text):
            gut.append(p)
    with open(out / "pride_gut_projects.tsv", "w") as fh:
        fh.write("accession\ttitle\torganisms\n")
        for p in gut:
            title = (p.get("title") or "").replace("\t", " ")
            fh.write(f"{p['accession']}\t{title}\t{p.get('organisms')}\n")
    with open(out / "pride_denovo_files.tsv", "w") as fh:
        fh.write("accession\tfile\tbytes\n")
        for p in gut:
            for f in get(f"{API}/projects/{p['accession']}/files/all"):
                if DENOVO_FILE.search(f["fileName"]):
                    fh.write(
                        f"{p['accession']}\t{f['fileName']}\t{f.get('fileSizeBytes')}\n"
                    )
    print(f"{len(projects)} projects returned, {len(gut)} gut microbiome projects")


if __name__ == "__main__":
    main()
