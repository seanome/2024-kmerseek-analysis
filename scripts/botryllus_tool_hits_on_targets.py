#!/usr/bin/env python3
"""The phmmer, jackhmmer and mmseqs2 hits of every Botryllus protein on the notebook 281
target proteins, from the invertebrate dark-set run's stored hit files.

    python3 botryllus_tool_hits_on_targets.py <hits_dir> <all_targets.txt> <max_evalue> <out.tsv>

<hits_dir> holds botryllus.chunk_NNNN.<tool>.tsv.gz, one line per hit: query, target,
tstart, tend, bits, evalue (what computeDarkSet reads; on Sherlock,
/scratch/users/olgabot/invertebrate-dark-set/results/botryllus/hits/). The files hold
domain hits above E = 10 too, so lines with evalue > <max_evalue> are dropped.
<all_targets.txt> is written by scripts/mhc_target_sets.py.

Writes <out.tsv>, no header: tool, query, target, tstart, tend, bits, evalue.

Added 2026-10-13 for notebook 281, so the tool-hit table it reads comes from a committed
script. Python 3.6, standard library only, so it runs with Sherlock's system python3.
"""
import glob
import gzip
import os
import re
import sys

NAME = re.compile(r"^botryllus\.chunk_\d+\.(phmmer|jackhmmer|mmseqs2)\.tsv\.gz$")


def main(hits_dir, targets_path, max_evalue, out):
    targets = set(l.strip() for l in open(targets_path) if l.strip())
    max_evalue = float(max_evalue)
    paths = sorted(glob.glob(os.path.join(hits_dir, "botryllus.chunk_*.tsv.gz")))
    if not paths:
        sys.exit("no hit files in %s" % hits_dir)
    counts = {}
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        for path in paths:
            m = NAME.match(os.path.basename(path))
            if not m:
                sys.exit("unexpected file name %s" % path)
            tool = m.group(1)
            with gzip.open(path, "rt") as hits:
                for line in hits:
                    f = line.rstrip("\n").split("\t")
                    if len(f) != 6:
                        sys.exit("%s: expected 6 columns, got %d" % (path, len(f)))
                    if f[1] in targets and float(f[5]) <= max_evalue:
                        fh.write("\t".join([tool] + f) + "\n")
                        counts[tool] = counts.get(tool, 0) + 1
    os.replace(tmp, out)
    print("%d files; hits kept: %s" % (len(paths), counts))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
