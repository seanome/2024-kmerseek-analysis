#!/usr/bin/env python3
"""Write the first N bases of every read and of its 10 base-shuffled copies as FASTA, for
the minimap2 and DIAMOND comparators. Shuffling is the same function score_chunks.py uses,
so both rules see identical chunks. Sequence ids are <read_id>|<copy>."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide  # noqa: E402
from score_chunks import N_SHUFFLES, shuffled  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--reads", required=True)
p.add_argument("--chunk", type=int, required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
with open(a.out, "w") as fo:
    for rid, seq in decide.read_records(a.reads):
        for copy in range(N_SHUFFLES + 1):
            s = seq if copy == 0 else shuffled(seq, rid, copy)
            fo.write(f">{rid}|{copy}\n{s[:a.chunk]}\n")
