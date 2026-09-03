"""A pooled PR curve must not change what it is averaging halfway across, or stop early.

The first version of pool_curve_over_species binned RECALL into hundredths, averaged
precision inside a bin, and stopped at the first bin some species had not reached. The stop
was defending something real: past the point where the shortest species curve ends, a mean
taken inside an x bin silently becomes the mean of whichever species got that far, and
those are the easiest species, so precision turns back upward showing a subset rather than
a pooled result.

The cure was worse than the disease. Binning on the OUTPUT axis lets the least reachable
proteome decide where every line stops. On the midi-plus run that reduced four kmerseek
arms to a single point at recall 0 and hhblits to three points topping out at recall 0.02
against a recall_reachable of 0.780, and walking recall at precision >= 0.5 off those
curves returned 0 for nearly every arm while the scalar metrics reported real numbers. The
figure and the table disagreed.

Averaging BOTH coordinates at matched RANK removes the discontinuity at its source. Each
species' curve is a list of operating points from its most stringent score to its most
permissive, and position u in [0, 1] down that list is defined for every species at every
u, so the set being averaged never changes -- which is what the old stop was defending --
and nothing is cut off. At u = 1 the x value is the mean of the per-species maximum
recalls, the same denominator the recall-at-precision table is built from, so the curve and
that table can be read against each other.

Rank rather than matched score threshold, which is the other way to keep every species in
the mean. A threshold is not the same stringency in two proteomes: an E-value depends on
the size of the database it was computed against and the target proteomes differ
several-fold in size, so pairing on the number would pair different operating points.

These tests fix both halves: no mid-curve recovery, and no truncation.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def curve(species, points):
    """points are (score_threshold, recall, precision), most lenient first."""
    return pl.DataFrame({
        "species": [species] * len(points),
        "score_threshold": [p[0] for p in points],
        "recall_reachable": [p[1] for p in points],
        "precision": [p[2] for p in points],
    })


# One easy species that reaches recall 0.30, and two hard ones that stop at 0.12. Same
# shape as the real data: mouse against everything else. Recall falls as the threshold
# rises, which is what orders each species' own curve.
EASY = curve("mouse", [(1, 0.30, 0.60), (2, 0.20, 0.62), (3, 0.12, 0.65),
                       (4, 0.10, 0.70), (5, 0.05, 0.80), (6, 0.00, 0.90)])
HARD_A = curve("ciona", [(1, 0.12, 0.05), (2, 0.10, 0.12), (3, 0.05, 0.20),
                         (4, 0.00, 0.30)])
HARD_B = curve("ecoli", [(1, 0.12, 0.04), (2, 0.10, 0.11), (3, 0.05, 0.18),
                         (4, 0.00, 0.28)])
SUB = pl.concat([EASY, HARD_A, HARD_B])


def pooled():
    return bmi.pool_curve_over_species(SUB, "recall_reachable", "precision")


# --- the truncation is gone ----------------------------------------------------------

def test_the_line_is_not_cut_off_by_the_least_reachable_species():
    pairs = pooled()
    assert pairs
    # mouse alone reaches 0.30, so the mean over three species at the permissive end of
    # the rank axis is (0.30 + 0.12 + 0.12) / 3. The old binning stopped the whole line at
    # 0.12 because that is where the shortest species ended.
    assert max(x for x, _ in pairs) == pytest_approx((0.30 + 0.12 + 0.12) / 3)


def test_every_species_is_in_the_mean_at_every_point():
    # A species whose calls have run out contributes recall 0 rather than dropping out, so
    # the number of curve points does not depend on which species is shortest.
    fewer = pl.concat([EASY, HARD_A])
    assert len(bmi.pool_curve_over_species(
        fewer, "recall_reachable", "precision")) == len(pooled())


def test_a_single_species_curve_survives_whole():
    pairs = bmi.pool_curve_over_species(EASY, "recall_reachable", "precision")
    assert max(x for x, _ in pairs) == pytest_approx(0.30)
    assert min(x for x, _ in pairs) == pytest_approx(0.0)
    # Resampled onto the shared grid rather than kept at its own six operating points, so
    # a one-species arm and a nine-species arm are drawn at the same resolution.
    assert len(pairs) == bmi.POOLED_CURVE_GRID


# --- and precision still does not recover mid-curve ----------------------------------

def test_precision_does_not_jump_when_a_species_runs_out():
    pairs = pooled()
    # Walked from the permissive end of the rank axis to the stringent one, precision
    # rises smoothly.
    # The failure this protects against is a JUMP: the old pooling produced a 3-4x step at
    # the recall where the hard species ended.
    ys = [y for _, y in pairs]
    for before, after in zip(ys, ys[1:]):
        assert after <= before * 2.0, f"precision jumped mid-curve: {ys}"


def test_the_old_recall_binning_would_have_shown_the_jump():
    # The behaviour this replaced, kept so the test says what it is protecting against:
    # binning on recall, the bin at 0.20 holds mouse alone and reads 0.62, up from 0.05 in
    # the last bin every species reached.
    binned = (
        SUB.with_columns(
            (pl.col("recall_reachable") * 100).round(0).cast(pl.Int64).alias("bin"))
        .group_by("bin").agg(pl.col("precision").mean().alias("y"))
        .sort("bin")
    )
    unguarded = {r["bin"]: r["y"] for r in binned.to_dicts()}
    assert unguarded[12] < 0.30
    assert unguarded[20] > 0.60


# --- what the curve and the table have to agree on -----------------------------------

def test_recall_at_precision_is_read_off_the_same_pairs_the_curve_draws():
    pairs = pooled()
    # 0.3 rather than the report's 0.5: the fixture's pooled precision tops out at 0.49,
    # because two of its three species are hard. The threshold is not what is under test.
    got = bmi.recall_at_precision(pairs, 0.3)
    reachable = [x for x, y in pairs if y >= 0.3]
    assert got == max(reachable)
    # and it is a point the drawn series actually contains
    assert f"{got:.4f}" in bmi.curve_series(pairs)


def test_an_arm_that_never_reaches_the_precision_returns_zero_not_none():
    low = curve("mouse", [(1, 0.4, 0.10), (2, 0.2, 0.20), (3, 0.0, 0.30)])
    pairs = bmi.pool_curve_over_species(low, "recall_reachable", "precision")
    assert bmi.recall_at_precision(pairs, 0.5) == 0.0


def test_series_keys_are_numeric_strings_at_fixed_precision():
    series = bmi.curve_series(pooled())
    for key in series:
        float(key)
        assert len(key.split(".")[1]) == 4


# --- degenerate inputs ---------------------------------------------------------------

def test_a_frame_without_species_pools_to_nothing_rather_than_lying():
    assert bmi.pool_curve_over_species(
        SUB.drop("species"), "recall_reachable", "precision") == []


def test_a_frame_without_a_threshold_falls_back_to_ordering_on_x():
    # Rank is what makes the species comparable, and a curve without a score column can
    # still be ranked -- by its own x. The endpoint, which is what the recall table reads,
    # is the same either way, because it is the mean of the per-species maxima.
    fallback = bmi.pool_curve_over_species(
        SUB.drop("score_threshold"), "recall_reachable", "precision")
    assert max(x for x, _ in fallback) == pytest_approx((0.30 + 0.12 + 0.12) / 3)


def pytest_approx(value, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

        def __repr__(self):
            return f"~{value}"
    return _Approx()
