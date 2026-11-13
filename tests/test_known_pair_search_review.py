"""Reject missing controls and duplicate database records before reporting rates."""

import sys
from pathlib import Path
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import known_pair_search_utils as ku


def rows():
    return [
        dict(
            query="q",
            target=t,
            kind=k,
            shuffle_failed=0,
            **{s: value for s in ku.SCORES},
            **{s + "_count": 1 for s in ku.SCORES}
        )
        for k in ["real", *ku.DECOYS]
        for t, value in [("partner", 4), ("other", 4)]
    ]


def test_complete_tied_ranking_and_missing_decoys():
    data = rows()
    got = ku.search_table(pl.DataFrame(data), "q", "partner")
    assert got["rank"].to_list() == [2] * 4
    assert got["decoy_rate_dipeptide"].to_list() == [1.0] * 4
    with pytest.raises(ValueError, match="no usable dipeptide"):
        ku.search_table(
            pl.DataFrame([r for r in data if r["kind"] != "dipeptide"]), "q", "partner"
        )


def test_duplicate_database_records_are_not_extra_evidence():
    data = rows()
    with pytest.raises(ValueError, match="duplicate real"):
        ku.search_table(pl.DataFrame(data + [data[0]]), "q", "partner")
    with pytest.raises(ValueError, match="duplicate uniform"):
        ku.search_table(pl.DataFrame(data + [data[2]]), "q", "partner")
