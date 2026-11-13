#!/usr/bin/env python3
"""Propose the time-split sample (notebook 280, Stop 1): up to 2_000 queries per Pfam
type at t1, all of them where fewer, drawn at random with a fixed seed.

A query is in a type's draw when it has at least one kept instance of that type. The
sample is the union of the draws; every kept instance on a sampled query is scored, and
each instance is reported under its own type. One decoy per query is added at Stop 2.

    propose_sample.py --qs <query_set dir> --out <dir>
writes proposed_sample_by_type.csv and proposed_sample_queries.csv (md5, acc, half).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from time_split_common import SHORT_AA, query_half  # noqa: E402

SEED = 280
PER_TYPE = 2_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    kept = pl.read_parquet(a.qs / "instances_kept.parquet")
    rng = random.Random(SEED)
    picked: dict[str, list[str]] = {}
    for t in sorted(kept["pfam_type"].unique().to_list()):
        q = sorted(kept.filter(pl.col("pfam_type") == t)["md5"].unique().to_list())
        picked[t] = q if len(q) <= PER_TYPE else sorted(rng.sample(q, PER_TYPE))
    union = sorted(set().union(*picked.values()))
    sample = kept.filter(pl.col("md5").is_in(union))
    rows = []
    for t, q in picked.items():
        n_avail = kept.filter(pl.col("pfam_type") == t)["md5"].n_unique()
        inst = sample.filter(pl.col("pfam_type") == t)
        rows.append({"pfam_type": t, "queries_available": n_avail, "queries_drawn": len(q),
                     "instances_in_sample": inst.height,
                     f"instances_under_{SHORT_AA}_aa": inst.filter(pl.col("length") < SHORT_AA).height})
    by_type = pl.DataFrame(rows)
    by_type.write_csv(a.out / "proposed_sample_by_type.csv")
    qtab = (sample.select("md5", "acc").unique("md5").sort("acc")
            .with_columns(half=pl.col("acc").map_elements(query_half, return_dtype=pl.String)))
    qtab.write_csv(a.out / "proposed_sample_queries.csv")
    print(by_type)
    print(f"sample: {len(union):_} queries (+ {len(union):_} decoys), {sample.height:_} instances; "
          f"tune {qtab.filter(pl.col('half') == 'tune').height:_}, "
          f"test {qtab.filter(pl.col('half') == 'test').height:_}; "
          f"residues {sample.unique('md5')['query_length'].sum():_}")


if __name__ == "__main__":
    main()
