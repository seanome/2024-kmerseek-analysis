#!/usr/bin/env python3
"""Check the notebook 281 tool re-run against the dark-set run's stored hits.

    python3 check_query_position_rerun.py <out_dir> <stored_hits_dir> <genes.txt> <lengths.tsv>

For phmmer and jackhmmer (HMMER per-domain tables in <out_dir>) and mmseqs2: every stored hit
of the proteins in <genes.txt> (query, target, target start, target end, bits, E-value, as
the run wrote them) should appear in the re-run with the same values, and nothing else
should. Also counts hits whose query length (HMMER model length, or mmseqs2 qlen) differs
from the protein's length in <lengths.tsv>: where it differs, the query positions are model
positions, not residues. Prints one line per tool.

Added 2026-10-13 for notebook 281 (PR 126). Python 3.6, standard library only.
"""
import glob
import gzip
import os
import sys


def main(out_dir, stored_dir, genes_path, lengths_path):
    genes = set(l.strip() for l in open(genes_path) if l.strip())
    length = {}
    for l in open(lengths_path):
        name, n = l.rstrip("\n").split("\t")
        length[name] = int(n)
    for tool in ("phmmer", "jackhmmer", "mmseqs2"):
        new, n, off, examples = set(), 0, 0, []
        for path in sorted(glob.glob(os.path.join(out_dir, tool + ".chunk_*.gz"))):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if line.startswith("#") or not line.strip():
                        continue
                    if tool == "mmseqs2":
                        q, t, qs, qe, ts, te, b, e, ql, tl = line.rstrip("\n").split("\t")
                    else:
                        f = line.split()
                        q, t, ts, te, b, e, ql = f[3], f[0], f[19], f[20], f[13], f[12], f[5]
                    n += 1
                    if int(ql) != length[q]:
                        off += 1
                        if len(examples) < 3:
                            examples.append((q, ql, length[q]))
                    new.add("\t".join((q, t, ts, te, b, e)))
        old = set()
        for path in glob.glob(os.path.join(stored_dir, "botryllus.chunk_*.%s.tsv.gz" % tool)):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if line.split("\t", 1)[0] in genes:
                        old.add(line.rstrip("\n"))
        print("%s: stored %d, re-run %d, same %d, only stored %d, only re-run %d; query length != protein length in %d of %d hits %s"
              % (tool, len(old), len(new), len(old & new), len(old - new), len(new - old), off, n, examples))
        for label, s in (("only stored", old - new), ("only re-run", new - old)):
            for x in sorted(s)[:3]:
                print("   %s: %s" % (label, x))
        del new, old


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
