"""Count how many probe sets get a sequence long enough for a 30-amino-acid HP k-mer, by source.

Usage: protein_coverage.py <fetched.tsv>[,<fetched.tsv>...] <out.tsv> <array name>=<sources.tsv> ...
A protein counts when it has at least 30 amino acids; an EST counts when it has at least 90 nucleotides, the length of
30 codons, though it is not translated. Accessions over 5000 nucleotides in the est group are genomic clones, not ESTs,
and are counted apart as genomic_clone. An est
accession with an annotated protein of 30 amino acids or more is a full-length cDNA, counted as cdna_with_protein. Writes one row per array and source, and probe_set_proteins.tsv beside out.tsv,
one row per probe set and protein.
"""

import csv
import os
import sys
from collections import defaultdict

AA, NT = 30, 90
fetched, out, *arrays = sys.argv[1:]
got = defaultdict(list)
for path in fetched.split(","):
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["sequence"] != "none":
                got[r["accession"]].append((r["sequence"], int(r["length"]), r["asked_for"]))
rows, pairs = [], []
for item in arrays:
    name, table = item.split("=", 1)
    counts = defaultdict(lambda: [0, 0, 0])
    with open(table) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            # An EST has no protein name, so it is named by its own accession
            found = {
                (a if kind == "est" else s): (n, kind)
                for a in r["accessions"].split(",")
                if a
                for s, n, kind in got.get(a, [])
            }
            source = r["source"]
            if source == "est" and any(kind == "protein" and n >= AA for n, kind in found.values()):
                source, found = "cdna_with_protein", {s: v for s, v in found.items() if v[1] == "protein"}
            elif source == "est" and any(n > 5000 for n, _ in found.values()):
                source = "genomic_clone"
            long_enough = [s for s, (n, kind) in found.items() if n >= (AA if kind == "protein" else NT)]
            c = counts[source]
            c[0] += 1
            c[1] += bool(found)
            c[2] += bool(long_enough)
            pairs += [
                {
                    "array": name,
                    "probe_set": r["probe_set"],
                    "source": source,
                    "sequence": s,
                    "length": found[s][0],
                }
                for s in long_enough
            ]
    total = sum(c[0] for c in counts.values())
    for source, (n, any_found, enough) in sorted(counts.items(), key=lambda kv: -kv[1][0]):
        rows.append(
            {
                "array": name,
                "source": source,
                "probe_sets": n,
                "share_of_array": f"{n / total:.3f}",
                "with_a_sequence": any_found,
                "long_enough": enough,
            }
        )
for path, table_rows in ((out, rows), (os.path.join(os.path.dirname(out), "probe_set_proteins.tsv"), pairs)):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
