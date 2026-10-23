"""Undefined precision cannot qualify a frozen threshold."""

import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import region_combiner_262 as rb


def test_empty_curve_and_nan_precision_do_not_qualify():
    r = pl.DataFrame(
        schema={
            "rid": pl.Int64,
            "score": pl.Float64,
            "query_has_pfam": pl.Boolean,
            "pfam_is_true": pl.Boolean,
        }
    )
    curve = rb.pr_curve(
        rb.Calls(r, pl.DataFrame(), {"pfam": 10}), "score", True, "pfam"
    )
    assert curve.height == 0
    assert rb.average_precision(curve) == 0.0
    assert rb.loosest_threshold(curve) is None
    assert (
        rb.loosest_threshold(
            pl.DataFrame({"precision": [float("nan"), 0.3], "threshold": [1.0, 2.0]})
        )
        is None
    )
