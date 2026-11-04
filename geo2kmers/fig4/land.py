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

import numpy as np

from geo2kmers.geo import read_table
from geo2kmers.landing import land_probes, passing_landings
from geo2kmers.sequences import find_column

gse, gpl, table, proteins, out = sys.argv[1:]


def read_probes(path):
    """Return {probe name: sequence} and {probe name: feature}."""
    probes, feature = {}, {}
    if str(path).endswith(".tsv.gz"):
        with gzip.open(path, "rt") as f:
            reader = csv.DictReader(f, delimiter="\t")
            set_column = "man_fsetid" if "man_fsetid" in reader.fieldnames else "probe_set"
            for i, row in enumerate(reader):
                if row["sequence"] and row["sequence"] != "NA":
                    probes[str(i)] = row["sequence"]
                    feature[str(i)] = row[set_column]
        return probes, feature
    _, columns, rows = read_table(path, "platform")
    sequence = find_column(columns, ["SEQUENCE"])
    for row in rows:
        if row.get(sequence, "").strip():
            probes[row[columns[0]]] = row[sequence].strip().upper()
            feature[row[columns[0]]] = row[columns[0]]
    return probes, feature


probes, feature = read_probes(table)
COMPLEMENT = str.maketrans("ACGT", "TGCA")
given = land_probes(probes, [proteins])
flipped = land_probes({name: s.translate(COMPLEMENT) for name, s in probes.items()}, [proteins])
orientation = "complement" if flipped[3] > given[3] else "as_given"
rule, k, rows, touched = flipped if orientation == "complement" else given
landed = passing_landings(rule, rows, feature.__getitem__)
with open(f"{out}/{gse}.landing.tsv", "w") as f:
    f.write("feature\tprotein\n")
    f.writelines(f"{name}\t{protein}\n" for name in sorted(landed) for protein in sorted(landed[name]))
all_proteins = {protein for names in landed.values() for protein in names}
median = float(np.median([len(s) for s in probes.values()]))
columns = "gse gpl rule k median_probe_length probes probes_with_a_kmer_in_proteome landing_share features_landed proteins_landed orientation probes_with_a_kmer_as_given probes_with_a_kmer_complemented"
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
values += [orientation, given[3], flipped[3]]
with open(f"{out}/{gse}.landing_share.tsv", "w") as f:
    f.write(columns.replace(" ", "\t") + "\n" + "\t".join(map(str, values)) + "\n")
print(*values)
