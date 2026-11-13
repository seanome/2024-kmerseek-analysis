import importlib.util
from pathlib import Path

import polars as pl
import pytest

spec = importlib.util.spec_from_file_location(
    "score_calls", Path(__file__).parents[1] / "bin/score_calls.py"
)
score_calls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_calls)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -1.0, None]
)
@pytest.mark.parametrize("side", ["target", "decoy"])
def test_invalid_evalues_are_rejected(value, side):
    good = pl.Series("evalue", [0.01], dtype=pl.Float64)
    bad = pl.Series("evalue", [value], dtype=pl.Float64)
    with pytest.raises(ValueError, match="finite, nonnegative"):
        score_calls.decoy_threshold(
            bad if side == "target" else good, bad if side == "decoy" else good, 0.05
        )


def test_threshold_keeps_entire_score_tie():
    result = score_calls.decoy_threshold(pl.Series([0.1, 0.1]), pl.Series([0.1]), 0.5)
    assert result == {"threshold": 0.1, "n_target": 2, "n_decoy": 1}


def test_empty_calls_have_no_threshold():
    empty = pl.Series([], dtype=pl.Float64)
    assert score_calls.decoy_threshold(empty, empty, 0.05)["threshold"] is None


def test_threshold_matches_independent_tied_count_enumeration():
    import random

    rng = random.Random(114)
    for _ in range(100):
        target = rng.choices([0.0, 0.01, 0.1, 1.0, 10.0], k=rng.randrange(21))
        decoy = rng.choices([0.0, 0.01, 0.1, 1.0, 10.0], k=rng.randrange(21))
        candidates = [
            (e, sum(x <= e for x in target), sum(x <= e for x in decoy))
            for e in sorted(set(target + decoy))
        ]
        passing = [(e, t, d) for e, t, d in candidates if t and d <= 0.5 * t]
        expected = (
            dict(zip(["threshold", "n_target", "n_decoy"], passing[-1]))
            if passing
            else {"threshold": None, "n_target": 0, "n_decoy": 0}
        )
        assert (
            score_calls.decoy_threshold(
                pl.Series(target, dtype=pl.Float64),
                pl.Series(decoy, dtype=pl.Float64),
                0.5,
            )
            == expected
        )
