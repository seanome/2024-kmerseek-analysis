#!/usr/bin/env python3
"""Per-residue AlphaFold pLDDT for the human query proteins, from their mmCIF models.

Writes a tab-separated file with one row per residue: accession, fragment, residue
(1-based, the model's label_seq_id) and pLDDT (the B-factor column, the same value on every
atom of a residue; the CA atom's is taken). The leaderboard's region pile-up averages these
over each true region to get the region's pLDDT.

Standard library only, so it runs with any python3 on a Sherlock compute node:

    srun -p hns -c 1 --mem 4G -t 00:20:00 python3 leaderboard_residue_plddt.py \
        <structures/human> residue_plddt.tsv.gz

A model whose file name carries a fragment number above 1 (AF-<acc>-F2) covers a later
stretch of a long protein; its residues are written with that fragment number and their
label_seq_id unchanged, and the region step uses F1 only, where label_seq_id is the
protein's own residue number.
"""

import gzip
import re
import sys
from pathlib import Path


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_files = n_rows = 0
    with gzip.open(out, "wt") as fh:
        fh.write("accession\tfragment\tresidue\tplddt\n")
        for cif in sorted(src.glob("AF-*-F*.cif*")):
            m = re.match(r"AF-(.+)-F(\d+)", cif.name)
            if not m:
                continue
            acc, frag = m[1], int(m[2])
            opener = gzip.open if cif.suffix == ".gz" else open
            cols: list[str] = []
            seen = set()
            with opener(cif, "rt") as f:
                for line in f:
                    if line.startswith("_atom_site."):
                        cols.append(line.strip().split(".", 1)[1])
                        continue
                    if not line.startswith("ATOM"):
                        continue
                    parts = line.split()
                    rec = dict(zip(cols, parts))
                    if rec.get("label_atom_id") != "CA":
                        continue
                    res = int(rec["label_seq_id"])
                    if res in seen:
                        continue
                    seen.add(res)
                    fh.write(f"{acc}\t{frag}\t{res}\t{rec['B_iso_or_equiv']}\n")
                    n_rows += 1
            n_files += 1
    print(f"{n_files} models, {n_rows} residues -> {out}")


if __name__ == "__main__":
    main()
