"""Checks for notebook 356's scorer: the placement p-value against a brute-force count, and
the reading-frame step against UniProt's positions on the committed exon table."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sc = pytest.importorskip("score_356_exon_landing")


def brute_p_place(start: int, end: int, a: int, b: int, n: int) -> float:
    """Share of start positions s where [s, s + L) lands on [a, b): inside >= 0.8, cover >= 0.3."""
    L, E = end - start, b - a
    hits = 0
    for s in range(n - L + 1):
        ov = max(0, min(s + L, b) - max(s, a))
        hits += ov >= sc.INSIDE_MIN * L and ov >= sc.COVER_MIN * E
    return hits / (n - L + 1)


@pytest.mark.parametrize(
    "start,end,a,b,n",
    [(10, 30, 8, 40, 300), (0, 5, 0, 30, 50), (100, 160, 90, 140, 400),
     (5, 9, 3, 60, 61), (200, 233, 198, 231, 233), (0, 300, 10, 40, 300)],
)
def test_p_place_matches_brute_force(start, end, a, b, n):
    df = pl.DataFrame({"tstart": [start], "tend": [end], "prot_start": [a],
                       "prot_end": [b], "target_len": [n]})
    got = df.select(sc.p_place_expr()).item()
    assert math.isclose(got, brute_p_place(start, end, a, b, n))


def test_reading_frame_matches_uniprot():
    ex = pl.read_parquet(REPO / "notebooks" / "356_data" / "exons.parquet")
    assert ex["agrees_uniprot"].all()
    # Consecutive exons share a residue exactly when the codon is split (phase 1 or 2).
    nxt = ex.sort("species", "target_acc", "exon_rank").with_columns(
        next_start=pl.col("prot_start").shift(-1).over("species", "target_acc"),
        next_phase=pl.col("start_phase").shift(-1).over("species", "target_acc"),
    ).drop_nulls("next_start")
    shared = nxt["next_start"] == nxt["prot_end"] - 1
    assert (shared == (nxt["next_phase"] != 0)).all()
