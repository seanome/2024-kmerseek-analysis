#!/usr/bin/env python3
"""How decoy family transfers spread over decoy queries, per setting and half, at each
setting's matched cutoff (notebook 280, check on the full run).

    decoy_spread.py --scores <results-full/scores> --inputs <inputs/full> \
        --decoy-pairs <scoreTransfers work dir>/decoy_pairs --out <dir>

--decoy-pairs reads the per-batch best score of every decoy-family pair that the scoring
task wrote (a few GB). Without it the decoy calls are read from scores/calls, which for the
full run is 245 GB and ran out of memory at 128 GB (job 48803080).

A decoy-family pair is counted once, at its best score, as score_transfers.py counts it.
Writes decoy_spread.csv: per setting and half, the decoys, the decoys with at least one
transfer, all transfers, the most transfers on one decoy, the share carried by the 10
decoys with the most, and the share of decoy rows (calls, or with --decoy-pairs per-batch pairs) whose score is not finite (kmerseek
extend gives a region E-value of inf when it has no fit for it). Also writes
decoy_top.csv: the 10 decoys with the most transfers per setting and half, with length.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decoy-pairs", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    q = pl.read_csv(a.inputs / "queries.tsv", separator="\t")
    dec = q.filter(pl.col("is_decoy") & (pl.col("role") == "sample")).select("query", "half", "length")
    n_dec = dict(dec.group_by("half").len().iter_rows())
    cut = dict(pl.read_csv(a.scores / "setting_summary.csv").select("setting", "cutoff_tune").iter_rows())

    if a.decoy_pairs is not None:
        calls = pl.scan_parquet(a.decoy_pairs / "*.parquet").select(
            "setting", "query", "pfam_acc", "half", "score")
    else:
        calls = (pl.scan_parquet(a.scores / "calls" / "*.parquet")
                 .filter(pl.col("is_decoy") & (pl.col("role") == "sample"))
                 .select("setting", "query", "pfam_acc", "half", "score"))
    # One setting at a time, and only pairs at or above its cutoff are grouped: all of the
    # full run's pairs at once ran out of memory at 64 GB (job 48805045).
    finite_parts, decoy_parts = [], []
    for s, c in cut.items():
        cs = calls.filter(pl.col("setting") == s)
        finite_parts.append(cs.group_by("setting", "half").agg(
            decoy_rows=pl.len(), nonfinite_share=(~pl.col("score").is_finite()).mean())
            .collect(engine="streaming"))
        decoy_parts.append(
            cs.filter(pl.col("score") >= c)
            .group_by("setting", "half", "query", "pfam_acc").agg(pl.col("score").max())
            .group_by("setting", "half", "query").agg(n_transfers=pl.len())
            .collect(engine="streaming"))
    finite = pl.concat(finite_parts)
    per_decoy = pl.concat(decoy_parts)

    rows = []
    for (s, h), g in per_decoy.group_by(["setting", "half"]):
        t = sorted(g["n_transfers"].to_list(), reverse=True)
        total = sum(t)
        rows.append({"setting": s, "half": h, "cutoff_tune": cut[s], "decoys": n_dec[h],
                     "decoys_with_transfer": len(t), "transfers": total,
                     "transfers_per_decoy": total / n_dec[h], "most_on_one_decoy": t[0],
                     "top10_share": sum(t[:10]) / total if total else math.nan})
    spread = (pl.DataFrame(rows).join(finite, on=["setting", "half"], how="left")
              .sort("setting", "half"))
    spread.write_csv(a.out / "decoy_spread.csv")
    top = (per_decoy.join(dec, on=["query", "half"]).sort("n_transfers", descending=True)
           .group_by("setting", "half", maintain_order=True).head(10))
    top.write_csv(a.out / "decoy_top.csv")
    with pl.Config(tbl_rows=200, tbl_width_chars=250, tbl_cols=20):
        print(spread)


if __name__ == "__main__":
    main()
