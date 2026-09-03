"""A pooled PR curve must average over every species at every drawn point, and must not
be truncated by a gap in one species' sampling.

The old drawing binned recall to two decimals, averaged precision inside a bin, and
stopped at the first bin that did not hold every species. The intent was to stop where the
shortest species curve ends. The effect was to stop at the first recall value no species
happened to land on -- a species curve is a few hundred scattered operating points, not a
dense grid, so a gap appears within the first bin or two.

In the midi-plus report that drew four kmerseek arms as a SINGLE point at recall 0.0 and
hhblits as three points ending at recall 0.02, while the same arms' recall at precision
0.5 was 0.087 and their recall_reachable 0.78. The curve panel and the recall table
contradicted each other and the curve was the one that was wrong.

The replacement averages BOTH coordinates at matched rank down each species' own curve, so
every drawn point is a mean over all species and the line's right-hand end is the mean over
species of each species' own maximum recall -- the same denominator "Recall at precision
>= 0.5" is built from.

It was a drawing bug, not a ranking bug: AUPRC, ROC AUC, Fmax and sensitivity-to-first-FP
are each computed per species on the unpooled curve inside evaluate_domain_calls.py and
only averaged afterwards, so none of them inherited either behaviour.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def curve(species, points):
    """A species curve, most stringent operating point first."""
    return pl.DataFrame({
        "species": [species] * len(points),
        "score_threshold": [float(len(points) - i) for i in range(len(points))],
        "recall_reachable": [p[0] for p in points],
        "precision": [p[1] for p in points],
    })


# One easy species that reaches recall 0.30 at high precision, and two hard ones that stop
# at 0.12 after decaying. Same shape as the real midi data: mouse against everything else.
EASY = curve("mouse", [(0.00, 0.90), (0.05, 0.80), (0.10, 0.70), (0.12, 0.65),
                       (0.20, 0.62), (0.30, 0.60)])
HARD_A = curve("ciona", [(0.00, 0.30), (0.05, 0.20), (0.10, 0.12), (0.12, 0.05)])
HARD_B = curve("ecoli", [(0.00, 0.28), (0.05, 0.18), (0.10, 0.11), (0.12, 0.04)])
SUB = pl.concat([EASY, HARD_A, HARD_B])


def pooled(sub=SUB):
    return bmi.pool_curve_over_species(sub, "recall_reachable", "precision")


def test_the_line_ends_at_the_mean_of_the_species_maxima():
    series, info = pooled()
    assert series
    # (0.30 + 0.12 + 0.12) / 3, and NOT 0.12: truncating to the shortest species throws
    # away two thirds of the easy species' curve and reports a recall no arm was scored at.
    assert max(float(x) for x in series) == 0.18
    assert abs(info["max_x"] - 0.18) < 1e-9
    assert info["shortest"][1] == 0.12
    assert info["longest"] == ("mouse", 0.30)


def test_every_drawn_point_averages_every_species():
    series, info = pooled()
    assert info["n_species"] == 3
    # The first point is the mean of the three species' most stringent precisions, so no
    # species is missing from it. Under the old binning this was the ONLY point drawn.
    first = series[min(series, key=float)]
    assert abs(first - (0.90 + 0.30 + 0.28) / 3) < 1e-9
    assert len(series) > 1


def test_a_sampling_gap_no_longer_truncates_the_line():
    """The failure that produced the one-point kmerseek curves.

    Two species whose recalls interleave and never share a value. The old code binned on
    recall, found bin 1 held one species, and stopped there.
    """
    ragged = pl.concat([
        curve("mouse", [(0.00, 0.9), (0.01, 0.8), (0.03, 0.6), (0.05, 0.4)]),
        curve("ecoli", [(0.00, 0.8), (0.02, 0.7), (0.04, 0.5), (0.06, 0.3)]),
    ])
    series, info = pooled(ragged)
    assert len(series) > 2, f"a sampling gap truncated the line: {series}"
    assert abs(max(float(x) for x in series) - 0.055) < 1e-9
    assert info["n_species"] == 2


def test_precision_never_recovers_by_dropping_a_species():
    """The other half of the old bug: past the shortest curve the mean silently became
    the mean of whichever species got that far, which are the easiest ones, so precision
    turned back upward. Averaging at matched rank cannot do that -- every point has all
    three species in it -- so a rise here would mean a real rise in all of them."""
    series, _ = pooled()
    ys = [series[k] for k in sorted(series, key=float)]
    assert ys == sorted(ys, reverse=True), f"precision recovered mid-curve: {ys}"
    assert ys[-1] < ys[0]


def test_a_one_point_species_is_carried_flat_rather_than_dropped():
    """Dropping it would put that species back in the "averaged over a subset" failure
    this function exists to remove."""
    mixed = pl.concat([
        curve("mouse", [(0.0, 0.9), (0.1, 0.5), (0.2, 0.3)]),
        curve("ecoli", [(0.1, 0.4)]),
    ])
    series, info = pooled(mixed)
    assert info["n_species"] == 2
    assert abs(min(series.values()) - (0.3 + 0.4) / 2) < 1e-9
    assert abs(max(float(x) for x in series) - (0.2 + 0.1) / 2) < 1e-9


def test_species_that_all_reach_the_same_recall_are_unchanged():
    same = pl.concat([
        curve("mouse", [(0.0, 0.9), (0.1, 0.5), (0.2, 0.3)]),
        curve("ecoli", [(0.0, 0.8), (0.1, 0.4), (0.2, 0.2)]),
    ])
    series, info = pooled(same)
    assert max(float(x) for x in series) == 0.2
    assert info["shortest"][1] == info["longest"][1] == 0.2


def test_a_frame_without_species_pools_to_nothing_rather_than_lying():
    no_species = SUB.drop("species")
    assert bmi.pool_curve_over_species(
        no_species, "recall_reachable", "precision") == ({}, {})
