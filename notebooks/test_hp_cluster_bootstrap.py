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
    # L is hydrophobic and K is polar in hp_thomas_dill2, so the class strings are
    # HHPP and PPHH. They share the 2-mers HH and PP, but only at different offsets:
    # no aligned column has the same class, so the longest aligned run must be 0.
    query, target = "LLKK", "KKLL"
    table = hc.TABLES["hp_thomas_dill2"]
    classes = ["".join("HP"[table[ord(r)]] for r in seq) for seq in (query, target)]
    assert classes == ["HHPP", "PPHH"]
    kmers = [{c[i : i + 2] for i in range(3)} for c in classes]
    assert kmers[0] & kmers[1] == {"HH", "PP"}
    (row,) = hc.pair_stats(query, target, ["hp_thomas_dill2"])
    assert row["n_cols"] == 4
    assert row["longest_run"] == 0
    # Shift the alignment by two columns and the two L's line up: a run of 2.
    (row,) = hc.pair_stats("--LLKK", "KKLL--", ["hp_thomas_dill2"])
    assert row["longest_run"] == 2
    (row,) = hc.pair_stats("LLKK", "LLKK", ["hp_thomas_dill2"])
    assert row["longest_run"] == 4


def test_missing_values_do_not_poison_cluster_statistics():
    d = pl.DataFrame({"family": ["a", "a", "b"], "value": [0.0, float("nan"), 1.0]})
    assert hc.cluster_mean_ci(d, "value", "family") == (0.5, 0.0, 1.0)
