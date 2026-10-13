"""Score every aligned pair with seed-and-extend in several alphabets, real and shuffled.

Reads the pair alignments from notebook 230's inputs, writes one row per
(pair, alphabet, penalty, seed length, real/null) with the best seeded ungapped segment
score. See notebooks/seed_extend_utils.py for the scoring.

Usage:
    python scripts/seed_extend_on_alignments.py --out /Users/olga/data/pfam/232_seed_extend_scores.parquet
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import seed_extend_utils as se  # noqa: E402

PFAM = "/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet"
SCOPE = "/Users/olga/data/scope/230_scope40_pair_alignments.parquet"

SEEDS = {
    "hp_thomas_dill2": [0, 6, 8, 10, 12, 15, 19, 23],
    "gbmr4": [0, 4, 5, 6, 8, 10, 13],
    "sdm12": [0, 3, 4, 5, 6, 7],
    "protein20": [0, 2, 3, 4, 5, 6],
}
PENALTIES = [1.0, 2.0, 3.0]
N_SHUFFLE = 2


def one(job):
    keys, qaln, taln, seed = job
    rng = np.random.default_rng(seed)
    rows = []
    for alphabet, seeds in SEEDS.items():
        for r in se.pair_seed_extend(qaln, taln, alphabet, PENALTIES, seeds, N_SHUFFLE, rng):
            rows.append({**keys, **r})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pfam", type=int, default=60_000)
    ap.add_argument("--procs", type=int, default=14)
    args = ap.parse_args()

    pf = pl.read_parquet(PFAM).filter(pl.col("lali") >= 50).sample(n=args.n_pfam, seed=0, shuffle=True)
    pf = pf.select(pl.lit("pfam").alias("dataset"), "family", "query", "target", "seqid_ali", "qaln", "taln")
    sc = (pl.read_parquet(SCOPE)
          .filter((pl.col("category") == "same_superfamily_diff_family") & (pl.col("lali") >= 50)
                  & (pl.max_horizontal("tm_q", "tm_t") >= 0.5))
          .select(pl.lit("scope").alias("dataset"), pl.lit(None, dtype=pl.String).alias("family"), "query", "target", "seqid_ali", "qaln", "taln"))
    df = pl.concat([pf, sc])
    print(df.group_by("dataset").len())
    jobs = [({"dataset": r["dataset"], "family": r["family"], "query": r["query"], "target": r["target"], "seqid_ali": r["seqid_ali"]},
             r["qaln"], r["taln"], i) for i, r in enumerate(df.iter_rows(named=True))]
    rows = []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=50)):
            rows += r
            if i % 10_000 == 0:
                print(f"  {i} pairs", flush=True)
    out = pl.DataFrame(rows)
    out.write_parquet(args.out)
    print(out.shape)


if __name__ == "__main__":
    main()
