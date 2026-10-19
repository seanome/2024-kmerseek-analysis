"""Tests for scripts/stage_diagnostic: encodings, scorers, seed counts, fixtures, metrics."""

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "stage_diagnostic"))

import alphabets  # noqa: E402
import evaluate  # noqa: E402
import fixtures  # noqa: E402
import seeds  # noqa: E402
from scoring import Scorer, blosum62, collapsed_matrix  # noqa: E402

SEQ = "MKTWLRDEHPAAAAACYFNQSGVCCKLLIVWGSTPDENQHKR"


@pytest.mark.parametrize("a", list(alphabets.CLUSTERS))
def test_every_alphabet_covers_the_20_residues_once(a):
    assert sorted(alphabets.residue_map(a)) == sorted(alphabets.CANONICAL)


@pytest.mark.parametrize("a", ["dayhoff6", "hp_lehninger2"])
def test_our_table_matches_sourmash(a):
    # two windows get the same sourmash dayhoff / hp hash exactly when our table gives
    # them the same encoding: encoding -> hash is one-to-one over many random windows
    import random

    rng = random.Random(0)
    enc_to_hash, hash_to_enc = {}, {}
    for _ in range(20000):
        w = "".join(rng.choice(alphabets.CANONICAL) for _ in range(3))
        (h,) = seeds.hashes(w, a, 3)
        e = alphabets.encode(w, a)
        assert enc_to_hash.setdefault(e, h) == h
        assert hash_to_enc.setdefault(h, e) == e
    assert len(enc_to_hash) == len(alphabets.CLUSTERS[a]) ** 3


def test_pr152_example_has_19_distinct_5mers():
    # kmerseek PR 152: "short" holds 19 distinct protein20 5-mers
    assert len(seeds.hashes("MKTWLRDEHPAAAAACYFNQSGV", "protein20", 5)) == 19


def test_scaled_keeps_a_subset():
    s = SEQ * 5
    full = seeds.hashes(s, "dayhoff6", 5, scaled=1)
    sub = seeds.hashes(s, "dayhoff6", 5, scaled=10)
    assert sub < full


def test_mask_drops_homopolymer_windows():
    # every window of a poly-A run is dropped, the window reaching the K is kept
    assert seeds.kept_windows("AAAAAAK", "protein20", 5, mask=True) == ["AAAAK"]
    # cys2: a window with no C is one class letter repeated, so it goes too
    assert seeds.kept_windows("KLMNPC", "cys2", 5, mask=True) == ["LMNPC"]


def test_collapsed_protein20_is_blosum62():
    m, expected = collapsed_matrix("protein20")
    b = blosum62()
    worst = max(abs(m[x][y] - b[x][y]) for x in alphabets.CANONICAL for y in alphabets.CANONICAL)
    assert worst < 0.5 and expected < 0


@pytest.mark.parametrize("a", ["dayhoff6", "sdm12", "hp_lehninger2", "cys2"])
def test_collapsed_matrices_have_negative_expected_score(a):
    assert collapsed_matrix(a)[1] < 0


def test_reduced_twin_scores_as_the_duplicate_under_b_and_lower_under_a():
    q, r = fixtures.build("dayhoff6", 9, 10, False)
    A, B = Scorer("full"), Scorer("dayhoff6")
    base, dup, twin = q["fx_base"], r["fx_exact_duplicate"], r["fx_reduced_twin"]
    assert alphabets.encode(twin, "dayhoff6") == alphabets.encode(base, "dayhoff6")
    assert twin != base
    assert B.score(B.prep(base), B.prep(twin)) == B.score(B.prep(base), B.prep(dup))
    assert A.score(base, twin) < A.score(base, dup)


def test_fixtures_are_deterministic_and_the_empty_sketch_is_empty():
    q1, r1 = fixtures.build("sdm12", 7, 10, False)
    q2, r2 = fixtures.build("sdm12", 7, 10, False)
    assert q1 == q2 and r1 == r2
    e = r1["fx_empty_sketch"]
    assert seeds.hashes(e, "sdm12", 7, scaled=1) and not seeds.hashes(e, "sdm12", 7, scaled=10)
    assert len(q1["fx_short_query"]) < 7


def test_coords_are_zero_based_end_exclusive():
    S = Scorer("full")
    q = "WWWWCCCCHHHH"
    t = "AAAAA" + q + "AAAAA"
    assert S.coords(q, t) == (0, 12, 5, 17)


def _frame(scores_pos, scores_neg):
    rows = [
        dict(query="q", reference=f"p{i}", relation="positive", **{m: s for m in evaluate.METHODS})
        for i, s in enumerate(scores_pos)
    ]
    rows += [
        dict(
            query="q", reference=f"n{i}", relation="background", **{m: s for m in evaluate.METHODS}
        )
        for i, s in enumerate(scores_neg)
    ]
    rows.append(
        dict(query="q", reference="q", relation="self", **{m: 1e9 for m in evaluate.METHODS})
    )
    return pl.DataFrame(rows)


def test_budget_threshold_and_ties():
    f = _frame([10, 5, 3, -math.inf], [7, 5, 1])
    pq = evaluate.per_query(f, [0, 1, 5]).filter(pl.col("method") == "A")
    got = dict(zip(pq["budget"], pq["n_found"]))
    # b=0: above 7 -> 10 only; b=1: above 5 (tie not counted) -> 10; b=5: fewer negatives
    # than 6, so every finite positive counts
    assert got == {0: 1, 1: 1, 5: 3}
    assert pq["n_pos"][0] == 4  # self is left out, the -inf positive still counts


def test_pr_points_end_at_the_finite_recall():
    f = _frame([10, 5, -math.inf], [7, 1])
    pr = evaluate.pr_points(f, "A")
    assert pr["recall"].max() == pytest.approx(2 / 3)
    assert pr["precision"][0] == 1.0


def test_bootstrap_resamples_groups():
    vals = {"a1": 1.0, "a2": 1.0, "b1": 0.0}
    groups = {"a1": "A", "a2": "A", "b1": "B"}
    lo, hi = evaluate.bootstrap(vals, groups, 500, 1)
    assert lo == 0.0 and hi == 1.0
    assert all(math.isnan(x) for x in evaluate.bootstrap({"a1": 1.0}, {"a1": "A"}, 10, 1))
