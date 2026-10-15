"""Score notebook 232's Pfam seed pairs with linked HP word pairs (notebook 294).

Takes the same 60,000 Pfam seed pairs notebook 232 scored (same filter, same polars sample
seed, same per-pair rng seed, so the two shuffled targets per pair are 232's), and writes
one row per comparison (real pair or shuffled target) with:

* B1_ext, B2_ext: 232's seed-10 and seed-8 scores with ungapped extension, penalty 2,
  recomputed here (Gate 1 compares B1_ext and B2_ext with 232's stored scores);
* seed<a>_count: shared exact a-mers on the alignment's diagonal (no extension);
* <setting>_ext and <setting>_count for every linked-pair setting in
  notebooks/linked_pairs_utils.py.

Usage:
    python scripts/linked_pairs_on_alignments.py \
        --out /Users/olga/data/pfam/294_linked_pairs_scores.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import linked_pairs_utils as lp  # noqa: E402

PFAM = "/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet"
N_PFAM = 60_000  # notebook 232's sample size
MIN_ALIGNED = 50  # notebook 232's lali filter
N_SHUFFLE = 2  # notebook 232's shuffles per pair


def one(job):
    keys, qaln, taln, seed = job
    rng = np.random.default_rng(seed)
    return [{**keys, **r} for r in lp.score_pair(qaln, taln, rng, N_SHUFFLE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N pairs only (testing)"
    )
    args = ap.parse_args()

    pf = (
        pl.read_parquet(PFAM)
        .filter(pl.col("lali") >= MIN_ALIGNED)
        .sample(n=N_PFAM, seed=0, shuffle=True)
    )
    jobs = [
        (
            {
                "pair_index": i,
                "family": r["family"],
                "query": r["query"],
                "target": r["target"],
                "seqid_ali": r["seqid_ali"],
                "q_len": r["q_len"],
                "t_len": r["t_len"],
            },
            r["qaln"],
            r["taln"],
            i,  # 232 seeded each pair's rng with its row index; Pfam rows came first
        )
        for i, r in enumerate(pf.iter_rows(named=True))
    ]
    if args.limit:
        jobs = jobs[: args.limit]
    t0 = time.time()
    rows = []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap(one, jobs, chunksize=20)):
            rows += r
            if i % 5_000 == 0:
                print(f"  {i:_} pairs  {time.time() - t0:.0f} s", flush=True)
    out = pl.DataFrame(rows)
    out.write_parquet(args.out)
    print(out.shape, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
