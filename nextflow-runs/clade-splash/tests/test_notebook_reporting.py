"""Missing annotations and exploratory power plots must not become biological verdicts."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import clade_splash_357_utils as u


def data(with_annotations=True):
    sig = pl.DataFrame(
        {
            "setting": ["dayhoff:13"] * 100,
            "gap": [0] * 100,
            "anchor": range(100),
            "labelling": ["real"] * 100,
            "q_comp": [0.5] * 100,
            "anchor_entropy_bits": [1.0] * 100,
        }
    ).with_columns(pl.col("anchor").cast(pl.UInt64))
    pfam = pl.DataFrame(
        {
            "setting": ["dayhoff:13"] * 100,
            "gap": [0] * 100,
            "anchor": range(100),
            "n_occurrences": [20] * 100,
            "n_occ_in_pfam": [0] * 100,
            "main_pfam": [None] * 100,
            "outside_pfam": [True] * 100,
        },
        schema=u.PFAM_SCHEMA,
    )
    counts = pl.DataFrame(
        {
            "setting": ["dayhoff:13"] * 3,
            "gap": [0] * 3,
            "labelling": ["real", "shuffle_tree", "shuffle_random"],
            "n_anchors_tested": [1000] * 3,
        }
    )
    return {
        "sig": sig,
        "pfam": pfam if with_annotations else pfam.head(0),
        "counts": counts,
    }


def test_missing_annotations_are_insufficient_not_a_negative_result():
    assert u.verdict(data(False))["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert u.verdict(data(True))["verdict"] == "PASS"


def test_missing_primary_tests_are_insufficient():
    d = data()
    d["counts"] = d["counts"].filter(pl.col("labelling") != "shuffle_tree")
    assert u.verdict(d)["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_bh_can_reject_above_the_bonferroni_threshold():
    """The plot's alpha / m line is sufficient for rejection, not necessary."""
    p = np.full(100, 0.01)
    assert np.all(p > 0.05 / len(p))
    assert np.all(u.cs.bh(p) < 0.05)
