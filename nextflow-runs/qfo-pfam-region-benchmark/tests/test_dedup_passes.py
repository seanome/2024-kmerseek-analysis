#!/usr/bin/env python3
"""Checks on the two pairwise suppression passes in evaluate_domain_calls.

Run it directly; it needs nothing but polars and prints a FAILURES count.

    python3 tests/test_dedup_passes.py

Three things are checked, and each one is a bug that has actually happened here:

  equivalence   the batched pass keeps exactly the rows the plain self-join kept, at every
                batch size, so bounding memory did not change any published number.
  determinism   the same input gives the same survivors every run. The lazy version of
                this pass did not: it removed rows by a row index computed inside the
                query plan, the plan was re-executed for each collect, and a re-executed
                scan does not return rows in the same order. Identical reruns of one arm
                kept 16, then 6, then 17 of the same 30 calls.
  brute force   on a small case, the survivors match a direct O(n^2) reading of the rule.
"""
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
from evaluate_domain_calls import (  # noqa: E402
    dedup_fragment_regions, dedup_transferred_calls,
)

FAILURES = 0


def check(ok: bool, what: str) -> None:
    global FAILURES
    if not ok:
        FAILURES += 1
        print(f"FAIL {what}")


def reference(df: pl.DataFrame, keys: list[str], intervals: list[tuple[str, str]],
              iou_min: float) -> pl.DataFrame:
    """The rule read literally, one pair at a time, in python."""
    rows = list(df.iter_rows(named=True))

    def iou(a, b, lo, hi):
        inter = max(0, min(a[hi], b[hi]) - max(a[lo], b[lo]))
        union = max(a[hi], b[hi]) - min(a[lo], b[lo])
        return inter / union if union > 0 else 0.0

    keep = []
    for i, a in enumerate(rows):
        beaten = False
        for j, b in enumerate(rows):
            if i == j or any(a[k] != b[k] for k in keys):
                continue
            if all(iou(a, b, lo, hi) >= iou_min for lo, hi in intervals) and (
                b["score"] > a["score"] or (b["score"] == a["score"] and j < i)
            ):
                beaten = True
                break
        if not beaten:
            keep.append(a)
    return pl.DataFrame(keep, schema=df.schema) if keep else df.head(0)


CALL_SCHEMA = {"query_acc": pl.String, "pfam_id": pl.String,
               "qstart": pl.Int64, "qend": pl.Int64, "score": pl.Float64}
REGION_SCHEMA = {"query_acc": pl.String, "target_acc": pl.String,
                 "qstart": pl.Int64, "qend": pl.Int64,
                 "tstart": pl.Int64, "tend": pl.Int64, "score": pl.Float64}


def make_calls(n, seed):
    rng = random.Random(seed)
    return pl.DataFrame([
        {"query_acc": f"Q{rng.randrange(4)}", "pfam_id": f"PF{rng.randrange(3):05d}",
         "qstart": (s := rng.randrange(0, 60)),
         "qend": s + rng.choice([10, 20, 20, 21, 40]),
         # coarse scores so exact ties are common and the tie-break gets exercised
         "score": float(rng.randrange(5))}
        for _ in range(n)
    ], schema=CALL_SCHEMA)


def make_regions(n, seed):
    rng = random.Random(seed)
    return pl.DataFrame([
        {"query_acc": f"Q{rng.randrange(3)}", "target_acc": f"T{rng.randrange(3)}",
         "qstart": (qs := rng.randrange(0, 60)),
         "qend": qs + rng.choice([20, 21, 40]),
         "tstart": (ts := rng.randrange(0, 60)),
         "tend": ts + rng.choice([20, 21, 40]),
         "score": float(rng.randrange(5))}
        for _ in range(n)
    ], schema=REGION_SCHEMA)


CASES = [
    ("transferred", dedup_transferred_calls, make_calls,
     ["query_acc", "pfam_id"], [("qstart", "qend")]),
    ("fragment", dedup_fragment_regions, make_regions,
     ["query_acc", "target_acc"], [("qstart", "qend"), ("tstart", "tend")]),
]

for name, fn, make, keys, intervals in CASES:
    for seed in range(15):
        for n in (0, 1, 5, 60):
            df = make(n, seed * 100 + n)
            want = reference(df, keys, intervals, 0.5)
            order = df.columns
            want_sorted = want.sort(order)
            for budget in (1, 7, 50, 20_000_000):
                got = fn(df, 0.5, pair_budget=budget)
                check(got.sort(order).equals(want_sorted),
                      f"{name} seed={seed} n={n} budget={budget}: "
                      f"kept {got.height}, reference kept {want.height}")
            repeats = {fn(df, 0.5).write_json() for _ in range(3)}
            check(len(repeats) == 1, f"{name} seed={seed} n={n}: not deterministic")

print(f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
