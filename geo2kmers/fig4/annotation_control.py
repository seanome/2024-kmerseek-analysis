"""Compare each landing with the gene annotation its platform ships with. Run after scoring.

Usage: annotation_control.py <series folder> <landing folder> <out.tsv>
The method never reads this annotation, and this script changes no landing. A landing agrees when the
platform names the same gene for that feature, ignoring case, so mouse names can match human ones. An old
name on one side and a new name on the other counts as disagreement, so the shares are a lower bound.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from geo2kmers.geo import read_table

series, landing, out = sys.argv[1:]
MANY_KMERS = 5
COLUMNS = {"GPL13912": "GENE_SYMBOL", "GPL570": "Gene Symbol", "GPL13667": "Gene Symbol", "GPL17586": "gene_assignment"}


def names_in(cell, column):
    if column == "gene_assignment":
        names = {part.split(" // ")[1].strip().upper() for part in cell.split(" /// ") if " // " in part}
        return names - {"---"}
    return {name.strip().upper() for name in re.split(r"///", cell) if name.strip() and name.strip() != "---"}


rows_out = []
for folder in sorted(Path(series).iterdir()):
    gse = folder.name
    gpl = next(p.stem for p in folder.glob("GPL*.txt"))
    _, columns, rows = read_table(folder / f"{gpl}.txt", "platform")
    genes = defaultdict(set)
    for row in rows:
        genes[row[columns[0]]] |= names_in(row.get(COLUMNS[gpl], "") or "", COLUMNS[gpl])
    with open(Path(landing) / f"{gse}.landing.tsv") as f:
        landings = list(csv.DictReader(f, delimiter="\t"))
    annotated = [r for r in landings if genes[r["feature"]]]
    agreeing = [r for r in annotated if r["protein"].upper() in genes[r["feature"]]]
    agree = len(agreeing)
    share = f"{agree / len(annotated):.3f}" if annotated else ""
    many = [r for r in annotated if int(r["distinct_kmers"]) >= MANY_KMERS]
    many_agree = sum(r in agreeing for r in many)
    few_agree = agree - many_agree
    rows_out.append([gse, gpl, len(landings), len(annotated), agree, share, len(annotated) - len(many), few_agree])
    rows_out[-1] += [len(many), many_agree]
with open(out, "w") as f:
    f.write("gse\tgpl\tlandings\tlandings_with_annotation\tagree\tshare_agree\t")
    f.write("annotated_under_5_kmers\tagree_under_5_kmers\tannotated_5_or_more_kmers\tagree_5_or_more_kmers\n")
    f.writelines("\t".join(map(str, row)) + "\n" for row in rows_out)
