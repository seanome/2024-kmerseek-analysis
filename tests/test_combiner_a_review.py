"""Resolve identical objectives by the documented smaller E-value cutoff."""

import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import region_combiner_261 as rc


def test_freeze_tie_uses_emax_before_panel_name():
    paths = pl.DataFrame(
        {
            "is_stop": [True, True],
            "precision_swissprot": [0.6, 0.6],
            "n_called": [10, 10],
            "panel": ["z", "a"],
            "emax": [1.0, 10.0],
        }
    )
    assert rc.freeze(paths)["emax"] == 1.0
