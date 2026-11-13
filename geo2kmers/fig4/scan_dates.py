"""List each sample's array scan date, to see whether pre and post were run on different days.

Usage: scan_dates.py <raw folder> <groups.tsv> <out.tsv>
Reads the date from the start of each CEL file. Writes dataset, comparison, scan date, and pre and post counts.
"""

import csv
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

raw, groups, out = sys.argv[1:]
FOLDER = {"GSE111555": "GSE111552", "GSE27536": "GSE27536", "GSE28998": "GSE28998", "GSE83578": "GSE83578"}
SUPPLEMENT_ONLY = {"full_model", "early_vs_rest"}


def scan_date(path):
    """The first date in the file's header: MM/DD/YY in a text header, or YYYY-MM-DD in the newer format."""
    with gzip.open(path, "rb") as f:
        head = f.read(200000).replace(b"\x00", b"").decode("latin-1")
    found = re.search(r"\d{4}-\d{2}-\d{2}T|\b\d{2}/\d{2}/\d{2}\b", head)
    return found.group().rstrip("T") if found else "unknown"


dates = {}
for dataset, folder in FOLDER.items():
    for path in Path(raw, folder).glob("GSM*.CEL.gz"):
        dates[dataset, re.match(r"GSM\d+", path.name).group()] = scan_date(path)
counts = Counter()
with open(groups) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if (r["dataset"], r["sample"]) in dates and r["comparison"] not in SUPPLEMENT_ONLY:
            counts[r["dataset"], r["comparison"], dates[r["dataset"], r["sample"]], r["group"]] += 1
with open(out, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["dataset", "comparison", "scan_date", "pre", "post"])
    for key in sorted({k[:3] for k in counts}):
        writer.writerow([*key, counts[(*key, "0")], counts[(*key, "1")]])
