#!/usr/bin/env python3
"""Write taxid -> (genus, phylum) for every taxid in the Swiss-Prot FASTA.

Swiss-Prot headers carry the taxid as `OX=`, so the taxids come from the FASTA itself and
no extra download is needed beyond the NCBI dump. A taxid with no genus or no phylum rank
in its lineage gets an empty cell, and an empty cell matches no excluded clade, so such an
entry stays in every held-out reference.

Usage: resolve_taxonomy.py <download-dir>
"""

import collections
import gzip
import sys
from pathlib import Path


def main(out_dir: Path) -> None:
    parent, rank = {}, {}
    with open(out_dir / "taxdump_raw" / "nodes.dmp") as handle:
        for line in handle:
            fields = line.split("\t|\t")
            parent[int(fields[0])] = int(fields[1])
            rank[int(fields[0])] = fields[2]
    names = {}
    with open(out_dir / "taxdump_raw" / "names.dmp") as handle:
        for line in handle:
            fields = line.split("\t|\t")
            if fields[3].startswith("scientific name"):
                names[int(fields[0])] = fields[1]
    print(f"taxonomy: {len(parent)} nodes")

    def genus_and_phylum(taxid):
        found, seen = {}, set()
        while taxid in parent and taxid not in seen:
            seen.add(taxid)
            if rank.get(taxid) in ("genus", "phylum") and rank[taxid] not in found:
                found[rank[taxid]] = names.get(taxid, f"taxid{taxid}")
            if taxid == 1:
                break
            taxid = parent[taxid]
        return found.get("genus"), found.get("phylum")

    taxids = collections.Counter()
    n = 0
    with gzip.open(
        out_dir / "swissprot_raw" / "uniprot_sprot.fasta.gz", "rt"
    ) as handle:
        for line in handle:
            if line.startswith(">"):
                n += 1
                taxids[int(line[line.index("OX=") + 3 :].split()[0])] += 1
    print(f"{n} Swiss-Prot entries, {len(taxids)} distinct taxids")

    resolved = {t: genus_and_phylum(t) for t in taxids}
    no_genus = sum(c for t, c in taxids.items() if resolved[t][0] is None)
    no_phylum = sum(c for t, c in taxids.items() if resolved[t][1] is None)
    print(f"entries with no genus rank: {no_genus}; with no phylum rank: {no_phylum}")

    with open(out_dir / "taxid_genus_phylum.tsv", "w") as out:
        out.write("taxid\tn_entries\tgenus\tphylum\n")
        for taxid, count in sorted(taxids.items()):
            genus, phylum = resolved[taxid]
            out.write(f"{taxid}\t{count}\t{genus or ''}\t{phylum or ''}\n")
    print("wrote taxid_genus_phylum.tsv")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
