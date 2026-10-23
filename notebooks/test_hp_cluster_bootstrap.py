"""Resample related pairs together."""

import numpy as np
import polars as pl
import hp_conservation_utils as hc


def test_duplicate_pairs_do_not_narrow_cluster_interval():
    d = pl.DataFrame({"family": ["a", "b"], "value": [0.0, 1.0]})
    original = hc.cluster_mean_ci(d, "value", "family")
    assert original == hc.cluster_mean_ci(pl.concat([d] * 50), "value", "family")
    assert original == hc.cluster_mean_ci(d.reverse(), "value", "family")
    assert original == (0.5, 0.0, 1.0)


def test_cluster_mean_retains_pair_weighting():
    d = pl.DataFrame({"family": ["a", "a", "b"], "value": [0.0, 0.0, 1.0]})
    assert np.isclose(hc.cluster_mean_ci(d, "value", "family")[0], 1 / 3)


def test_offset_seed_is_absent_from_aligned_runs():
    assert hc.longest_true_run(np.array([False] * 4)) == 0
    a, b = "HHPP", "PPHH"
    assert set(a[i : i + 2] for i in range(3)) & set(
        b[i : i + 2] for i in range(3)
    ) == {"HH", "PP"}
