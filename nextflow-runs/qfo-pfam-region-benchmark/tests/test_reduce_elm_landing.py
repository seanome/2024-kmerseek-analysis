"""Tests for scripts/reduce_elm_landing.py (notebook 250): ranking, landing and the placement null."""

import importlib.util
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "reduce_elm_landing.py"
spec = importlib.util.spec_from_file_location("reduce_elm_landing", SCRIPT)
rel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel)


def test_rank_and_cut_ranks_within_query_and_breaks_ties_by_target():
    df = pl.DataFrame(
        {
            "query_acc": ["q1", "q1", "q1", "q2"],
            "target_acc": ["tB", "tA", "tC", "tA"],
            "qstart": [0, 0, 5, 0],
            "qend": [10, 10, 15, 10],
            "tstart": [0, 0, 0, 0],
            "tend": [10, 10, 10, 10],
            "score": [2.0, 2.0, 5.0, 1.0],
        }
    )
    out = rel.rank_and_cut(df, 2)
    q1 = out.filter(pl.col("query_acc") == "q1")
    # tC has the best score; tA beats tB on the tie; the third row is cut at L = 2.
    assert q1["target_acc"].to_list() == ["tC", "tA"]
    assert q1["rank"].to_list() == [1, 2]
    assert out.filter(pl.col("query_acc") == "q2")["rank"].to_list() == [1]


def test_place_share_counts_windows_covering_80pct_of_the_motif():
    # Protein 20 aa, motif [10, 14) (4 aa), window 6: a start p lands if the window covers
    # >= 80% of the motif (3.2, so all 4 residues), i.e. p in 8..10 -> 3 of 15 starts.
    assert rel.place_share(20, 6, [(10, 14)]) == pytest.approx(3 / 15)
    assert rel.place_share(20, 21, [(10, 14)]) is None
    assert rel.place_share(20, 4, []) is None


def test_score_calls_lands_on_half_cover_and_keeps_iou():
    calls = pl.DataFrame(
        {
            "query_acc": ["q"],
            "elm_class": ["LIG_X"],
            "qstart": [100],
            "qend": [119],
            "target_acc": ["t"],
            "tstart": [0],
            "tend": [19],
            "score": [1.0],
            "rank": [1],
        }
    )
    human = pl.DataFrame(
        {
            "accession": ["q", "q"],
            "elm_instance": ["i1", "i2"],
            "elm_class": ["LIG_X", "LIG_X"],
            "start": [110, 117],
            "end": [116, 123],
        }
    )
    m = rel.score_calls(calls, human, by_label=True).sort("elm_instance")
    # i1 lies inside the 19-residue call: cover 1, IoU 6/19. i2 has 2 of 6 covered: not landed.
    assert m["landed"].to_list() == [True, False]
    assert m["iou"][0] == pytest.approx(6 / 19)
