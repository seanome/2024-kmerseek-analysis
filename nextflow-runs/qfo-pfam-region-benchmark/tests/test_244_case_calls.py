"""tables/244_case_calls.csv must carry the coordinates notebook 244 printed for its top 10.

scripts/export_244_case_calls.py writes one row per (case, tool) for every row of
tables/244_hero_candidates.csv. The four pairs below are the kmerseek chosen-arm calls the
notebook printed (1-based, inclusive, human then target). The committed CSV is checked
everywhere; the exporter itself is re-run on those four cases only where the midi-plus
landing tables are on disk (the laptop), and must give the same rows as the committed CSV.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]
CALLS = ROOT / "tables" / "244_case_calls.csv"
CANDIDATES = ROOT / "tables" / "244_hero_candidates.csv"

# (gene, species, target) -> (query_start, query_end, target_start, target_end)
PRINTED = {
    ("COL9A1", "chicken", "P12106"): (669, 700, 667, 698),
    ("ROS1", "yeast", "P38070"): (1950, 1959, 133, 142),
    ("COL12A1", "zebrafish", "A5PN28"): (2820, 2848, 216, 244),
    ("PHACTR2", "chicken", "Q801X6"): (475, 499, 341, 365),
}
TOOLS = [
    "kmerseek chosen arm",
    "kmerseek best other alphabet",
    "phmmer",
    "MMseqs2",
    "MMseqs2 iterative",
    "Foldseek",
    "ProstT5",
    "Reseek",
    "Kyte-Doolittle scan",
]
COORDS = ["query_start", "query_end", "target_start", "target_end"]


@pytest.fixture(scope="module")
def calls():
    return pl.read_csv(CALLS, infer_schema_length=None)


def test_every_case_has_the_nine_tools_once(calls):
    n_cases = pl.read_csv(CANDIDATES, infer_schema_length=None).height
    assert calls["case_id"].n_unique() == n_cases
    per_case = calls.group_by("case_id", maintain_order=True).agg(pl.col("tool"))
    for case_id, tools in per_case.iter_rows():
        assert tools == TOOLS, (case_id, tools)


@pytest.mark.parametrize("key", list(PRINTED))
def test_top10_coordinates_match_the_notebook(calls, key):
    gene, species, target = key
    row = calls.filter(
        (pl.col("gene") == gene)
        & (pl.col("species") == species)
        & (pl.col("target") == target)
        & (pl.col("tool") == "kmerseek chosen arm")
    )
    assert row.height == 1, row
    assert tuple(row.select(COORDS).row(0)) == PRINTED[key]
    assert row["outcome"][0] == "lands"


def test_no_call_rows_have_no_coordinates(calls):
    no = calls.filter(pl.col("outcome") == "no call")
    assert no.select(COORDS + ["iou"]).null_count().row(0) == (no.height,) * 5


def test_call_elsewhere_keeps_target_columns_empty(calls):
    el = calls.filter(pl.col("outcome") == "best call on another target")
    assert el["target_start"].is_null().all() and el["target_end"].is_null().all()
    assert (el["other_target"] != el["target"]).all()
    assert el["other_target_start"].is_not_null().all()


def test_exporter_reproduces_the_committed_rows(calls):
    sys.path.insert(0, str(ROOT / "notebooks"))
    sys.path.insert(0, str(ROOT / "scripts"))
    he = pytest.importorskip("hero_example_utils")
    if not he.LANDING_DIR.exists():
        pytest.skip(f"landing tables not on this machine: {he.LANDING_DIR}")
    import export_244_case_calls as ex

    cases = he.load_cases()
    ids = (
        calls.filter(
            pl.struct("gene", "species", "target").is_in(
                [dict(gene=g, species=s, target=t) for g, s, t in PRINTED]
            )
        )["case_id"]
        .unique()
        .to_list()
    )
    fresh = ex.export_calls(cases.filter(pl.col("case_id").is_in(ids)))
    committed = calls.filter(pl.col("case_id").is_in(ids))
    assert fresh.sort("case_id", "tool").equals(
        committed.select(fresh.columns).sort("case_id", "tool"), null_equal=True
    )
