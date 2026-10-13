#!/usr/bin/env python3
"""One table of phmmer, jackhmmer and mmseqs2 hits with positions on both proteins, from the
output of scripts/281_tool_search_query_positions.sbatch.

    python3 parse_query_position_hits.py <out_dir> <max_evalue> <hits.tsv>

Writes <hits.tsv>, with a header: tool, query, target, qstart, qend, tstart, tend, bits,
evalue, qlen, tlen. Positions are 1-based and inclusive.

HMMER per-domain table (phmmer, jackhmmer): the Botryllus protein is the query, so the
query positions are the model coordinates (columns 16-17, "hmm from/to"; phmmer and jackhmmer
build the model on the query, so model positions are query residues), the target positions
are the envelope (columns 20-21, what the dark-set run kept), bits is the domain score
(column 14) and evalue the domain's independent E-value (column 13), as in the run.
mmseqs2: its own qstart, qend, tstart, tend, bits, evalue columns.

Rows with evalue > <max_evalue> are dropped (HMMER reports domains above the -E cutoff when
the whole sequence passes it). Added 2026-10-13 for notebook 281 (PR 126). Python 3.6,
standard library only.
"""
import glob
import gzip
import os
import sys

COLS = ["tool", "query", "target", "qstart", "qend", "tstart", "tend", "bits", "evalue", "qlen", "tlen"]


def main(out_dir, max_evalue, out):
    max_evalue = float(max_evalue)
    counts = {}
    tmp = out + ".tmp"
    with open(tmp, "w") as w:
        w.write("\t".join(COLS) + "\n")
        for path in sorted(glob.glob(os.path.join(out_dir, "*.chunk_*.*.gz"))):
            tool = os.path.basename(path).split(".")[0]
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if line.startswith("#") or not line.strip():
                        continue
                    if tool in ("phmmer", "jackhmmer"):
                        f = line.split()
                        if len(f) < 22:
                            sys.exit("%s: short line %r" % (path, line[:80]))
                        row = [tool, f[3], f[0], f[15], f[16], f[19], f[20], f[13], f[12], f[5], f[2]]
                    elif tool == "mmseqs2":
                        f = line.rstrip("\n").split("\t")
                        if len(f) != 10:
                            sys.exit("%s: expected 10 columns, got %d" % (path, len(f)))
                        q, t, qs, qe, ts, te, bits, ev, ql, tl = f
                        row = [tool, q, t, qs, qe, ts, te, bits, ev, ql, tl]
                    else:
                        sys.exit("unexpected file %s" % path)
                    if float(row[8]) <= max_evalue:
                        w.write("\t".join(row) + "\n")
                        counts[tool] = counts.get(tool, 0) + 1
    os.replace(tmp, out)
    print("hits kept: %s" % counts)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
