"""tables/244_landed_by_type_species_tool.csv, from scripts/export_244_landed_by_type_species.py."""

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import export_244_landed_by_type_species as ex  # noqa: E402


@pytest.fixture(scope="module")
def committed():
    return pl.read_csv(ex.OUT, infer_schema_length=None)


def _one(start, end, fstart, fend):
    d = pl.DataFrame(
        {"s": [start], "e": [end], "fs": [fstart], "fe": [fend]},
        schema={k: pl.Int64 for k in ("s", "e", "fs", "fe")},
    )
    exprs = ex.closed_overlap(pl.col("s"), pl.col("e"), pl.col("fs"), pl.col("fe"))
    return d.select(**exprs).row(0, named=True)


def test_closed_overlap_counts_both_ends():
    # SCUBE3 in mouse: kmerseek 323-347 on the feature 318-356 shares 25 of 39 residues.
    r = _one(323, 347, 318, 356)
    assert r["iou_closed"] == pytest.approx(25 / 39)
    assert r["inside"] == 1.0
    # Its MMseqs2 iterative call, 303-386: 39 of 84 residues inside.
    assert _one(303, 386, 318, 356)["inside"] == pytest.approx(39 / 84)
    # A one-residue call on the feature's first residue overlaps it by one residue.
    assert _one(318, 318, 318, 356)["iou_closed"] == pytest.approx(1 / 39)
    # Adjacent but not overlapping.
    assert _one(300, 317, 318, 356)["iou_closed"] == 0.0


def test_no_call_is_null_not_one():
    d = pl.DataFrame(
        {"s": [None], "e": [None], "fs": [10], "fe": [20]},
        schema={k: pl.Int64 for k in ("s", "e", "fs", "fe")},
    )
    r = d.select(
        **ex.closed_overlap(pl.col("s"), pl.col("e"), pl.col("fs"), pl.col("fe"))
    )
    assert r.row(0) == (None, None)


def test_committed_csv_is_current(committed):
    cases = pl.read_csv(ex.CANDIDATES, infer_schema_length=None)
    calls = pl.read_csv(ex.CALLS, infer_schema_length=None)
    fresh = ex.summarise(ex.per_call(cases, calls))
    assert fresh.equals(committed.select(fresh.columns), null_equal=True)


def test_every_type_species_has_nine_tools_and_counts_fit(committed):
    per = committed.group_by("feature_type", "species").agg(
        n_tools=pl.len(), n_cases=pl.col("n_cases").n_unique()
    )
    assert (per["n_tools"] == 9).all() and (per["n_cases"] == 1).all()
    n = pl.read_csv(ex.CANDIDATES, infer_schema_length=None).height
    assert (
        committed.group_by("tool").agg(pl.col("n_cases").sum())["n_cases"].to_list()
        == [n] * 9
    )
    s = committed["n_landed"] + committed["n_spills"] + committed["n_no_call"]
    assert (s <= committed["n_cases"]).all()
    # A landed call (IoU >= 0.5) has at least half of its length inside, so it never spills.
    with_call = committed["n_cases"] - committed["n_no_call"]
    assert ((committed["mean_iou"].is_null()) == (with_call == 0)).all()
