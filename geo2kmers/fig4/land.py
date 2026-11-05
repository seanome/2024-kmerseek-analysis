"""Land one array series' probes on the signature proteins, using no gene annotation.

Usage: land.py <GSE> <GPL> <probe table> <proteins.fasta> <out folder>
The probe table is a GEO platform table with a SEQUENCE column, or a TSV of Bioconductor probes from
export_probe_tables.R. Writes <GSE>.landing.tsv (feature, protein) and <GSE>.landing_share.tsv. A
feature is what the series reports a value for: a probe, or on 25-letter arrays a probe set.

Some packages hold a probe as the complement of the strand, read in the same direction, and six frames
do not cover that. So each table is landed as given and complemented, and the reading with more probes
holding a k-mer is kept. This uses no annotation, and the landing share file records both counts.
"""

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo2kmers.geo import read_table
from geo2kmers.landing import land_probes, passing_landings
from geo2kmers.sequences import find_column
from geo2kmers.sketch import read_fasta

gse, gpl, table, proteins, out = sys.argv[1:]


def read_probes(path):
    """Return {probe name: sequence} and {probe name: its features}. One probe can serve several features."""
    probes, feature = {}, defaultdict(list)
    if str(path).endswith(".tsv.gz"):
        with gzip.open(path, "rt") as f:
            reader = csv.DictReader(f, delimiter="\t")
            set_column = "man_fsetid" if "man_fsetid" in reader.fieldnames else "probe_set"
            for i, row in enumerate(reader):
                if row["sequence"] and row["sequence"] != "NA":
                    name = row.get("fid") or str(i)
                    probes[name] = row["sequence"]
                    feature[name].append(row[set_column])
        return probes, feature
    _, columns, rows = read_table(path, "platform")
    sequence = find_column(columns, ["SEQUENCE"])
    for row in rows:
        if row.get(sequence, "").strip():
            probes[row[columns[0]]] = row[sequence].strip().upper()
            feature[row[columns[0]]].append(row[columns[0]])
    return probes, feature


probes, feature = read_probes(table)
COMPLEMENT = str.maketrans("ACGT", "TGCA")
flipped_probes = {name: s.translate(COMPLEMENT) for name, s in probes.items()}
given = land_probes(probes, [proteins])
flipped = land_probes(flipped_probes, [proteins])
orientation = "complement" if flipped[3] > given[3] else "as_given"
rule, k, rows, touched = flipped if orientation == "complement" else given
# The same landing on backward proteins, which code for nothing, shows how much of it chance and repeats allow.
backward = f"{out}/{gse}.backward.fasta"
with open(backward, "w") as f:
    f.writelines(f">{name}\n{sequence[::-1]}\n" for name, sequence in read_fasta(proteins).items())
chance_rows = land_probes(flipped_probes if orientation == "complement" else probes, [backward])[2]
Path(backward).unlink()
reversed_landings = sum(len(found) for found in passing_landings(rule, chance_rows, feature.__getitem__).values())
landed = passing_landings(rule, rows, feature.__getitem__)
landings = sum(len(found) for found in landed.values())
with open(f"{out}/{gse}.landing.tsv", "w") as f:
    f.write("feature\tprotein\tdistinct_kmers\n")
    f.writelines(f"{name}\t{p}\t{landed[name][p]}\n" for name in sorted(landed) for p in sorted(landed[name]))
all_proteins = {protein for names in landed.values() for protein in names}
median = float(np.median([len(s) for s in probes.values()]))
columns = "gse gpl rule k median_probe_length probes probes_with_a_kmer_in_proteome landing_share_of_sequenced_probes features_landed proteins_landed orientation probes_with_a_kmer_as_given probes_with_a_kmer_complemented landings reversed_landings"
values = [
    gse,
    gpl,
    rule,
    k,
    median,
    len(probes),
    touched,
    f"{touched / len(probes):.4f}",
    len(landed),
    len(all_proteins),
]
values += [orientation, given[3], flipped[3], landings, reversed_landings]
with open(f"{out}/{gse}.landing_share.tsv", "w") as f:
    f.write(columns.replace(" ", "\t") + "\n" + "\t".join(map(str, values)) + "\n")
print(*values)
