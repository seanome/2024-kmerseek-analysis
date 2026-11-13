"""List which probe set or gene cluster each probe sequence belongs to.

Usage: feature_table.py <GPL> <probe tables folder> <series folder> <out.tsv.gz>
Writes sequence and feature. The file is read only to say whether a SPLASH anchor sits inside one feature.
"""

import csv
import gzip
import sys

from geo2kmers.geo import read_table
from geo2kmers.sequences import find_column

gpl, tables, series, out = sys.argv[1:]
FILES = {"GPL570": "GPL570.probes.tsv.gz", "GPL13667": "GPL13667.probes.tsv.gz", "GPL17586": "GPL17586.raw.tsv.gz"}
with gzip.open(out, "wt") as f:
    f.write("sequence\tfeature\n")
    if gpl in FILES:
        with gzip.open(f"{tables}/{FILES[gpl]}", "rt") as table:
            reader = csv.DictReader(table, delimiter="\t")
            column = "man_fsetid" if "man_fsetid" in reader.fieldnames else "probe_set"
            f.writelines(f"{r['sequence']}\t{r[column]}\n" for r in reader if r["sequence"] not in ("", "NA"))
    else:
        _, columns, rows = read_table(f"{series}/{gpl}.txt", "platform")
        sequence = find_column(columns, ["SEQUENCE"])
        f.writelines(f"{r[sequence].strip()}\t{r[columns[0]]}\n" for r in rows if r.get(sequence, "").strip())
