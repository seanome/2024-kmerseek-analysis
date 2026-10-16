#!/usr/bin/env python3
"""Score the first N bases of every read, and of 10 base-shuffled copies, against k-mer sets.

One output row per read x copy x chunk length x alphabet x k x hold-out. The rule
threshold m (keep when n_kmers_in_index >= m) is applied downstream, so each row carries
the count and not a keep/eject call. Copy 0 is the read as sequenced; copies 1..10 are the
whole read with its bases shuffled (seeded per read and copy), then cut to N.

Usage:
    score_chunks.py --reads sample.fastq --reference ii=ref_ii.fasta --reference iii=ref_iii.fasta \
        --chunk 400 --out scores.parquet
"""

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide  # noqa: E402

# k per alphabet from the experiment brief. sourmash's sketch defaults are protein k=10,
# dayhoff k=16 and hp k=42 (sourmash src/sourmash/command_sketch.py DEFAULTS), so protein
# k=10 is added beside the brief's 7, 9, 11.
KSIZES = {"protein": [7, 9, 10, 11], "dayhoff": [10, 13, 16], "hp": [20, 30, 42]}
N_SHUFFLES = 10
SEED = 20261014

_INDEXES = {}


def _init(ref_paths):
    for holdout, path in ref_paths.items():
        prots = [s for _, s in decide.read_records(path)]
        for alphabet, ks in KSIZES.items():
            for k in ks:
                _INDEXES[(holdout, alphabet, k)] = decide.build_index(
                    prots, alphabet, k
                )


def shuffled(seq, read_id, copy):
    """The read's bases in a random order, seeded by read id and copy number."""
    bases = list(seq)
    random.Random(f"{SEED}-{read_id}-{copy}").shuffle(bases)
    return "".join(bases)


def score_read(args):
    read_id, seq, chunks = args
    rows = []
    for copy in range(N_SHUFFLES + 1):
        s = seq if copy == 0 else shuffled(seq, read_id, copy)
        for n in chunks:
            chunk = s[:n]
            for (holdout, alphabet, k), index in _INDEXES.items():
                t0 = time.perf_counter()
                n_hits, frame, n_frame, _ = decide.decide(chunk, index, alphabet, k, 1)
                dt = time.perf_counter() - t0
                rows.append(
                    (
                        read_id,
                        copy,
                        len(chunk),
                        holdout,
                        alphabet,
                        k,
                        n_hits,
                        frame,
                        n_frame,
                        dt,
                    )
                )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reads", required=True)
    p.add_argument(
        "--reference", action="append", required=True, help="holdout=path.fasta"
    )
    p.add_argument("--chunk", type=int, action="append", required=True)
    p.add_argument(
        "--out",
        required=True,
        help="output directory; one parquet part per 10_000 reads",
    )
    p.add_argument("--processes", type=int, default=12)
    a = p.parse_args()
    refs = dict(r.split("=", 1) for r in a.reference)
    jobs = ((rid, s, a.chunk) for rid, s in decide.read_records(a.reads))
    cols = [
        "read_id",
        "copy",
        "chunk_bases",
        "holdout",
        "alphabet",
        "k",
        "n_kmers_in_index",
        "best_frame",
        "best_frame_kmers",
        "decision_seconds",
    ]
    os.makedirs(a.out, exist_ok=True)
    out, part, n_rows = [], 0, 0

    def flush():
        nonlocal out, part, n_rows
        if not out:
            return
        pl.DataFrame(out, schema=cols, orient="row").with_columns(
            pl.col(
                "copy", "chunk_bases", "k", "n_kmers_in_index", "best_frame_kmers"
            ).cast(pl.Int32),
            pl.col("decision_seconds").cast(pl.Float32),
        ).write_parquet(os.path.join(a.out, f"part{part:03d}.parquet"))
        n_rows += len(out)
        out, part = [], part + 1

    with Pool(a.processes, initializer=_init, initargs=(refs,)) as pool:
        for i, rows in enumerate(pool.imap(score_read, jobs, chunksize=200), start=1):
            out.extend(rows)
            if i % 10_000 == 0:
                flush()
                print(f"{i} reads", file=sys.stderr, flush=True)
    flush()
    print(f"wrote {n_rows} rows in {part} parts", file=sys.stderr)


if __name__ == "__main__":
    main()
