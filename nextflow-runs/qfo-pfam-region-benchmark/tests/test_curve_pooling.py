"""A pooled PR curve must not change what it is averaging halfway across.

section_curves draws one line per tool by binning recall and averaging precision within a
bin. Each species contributes its own curve, and the species curves end at different
recalls, so past the point where the shortest one stops the mean silently becomes the mean
of whichever species got that far. Those are the EASIEST species, so precision jumps back
up.

That is what the midi report showed: four kmerseek arms fell to ~0.15 precision at recall
0.12 and jumped to ~0.60 at recall 0.13, because every target species except mouse tops out
around recall 0.10-0.12. hhblits does it at 0.88 and phmmer at 0.23, each at its own
crossover. Precision recovering 3-4x mid-curve is not a PR shape any ranking produces.

It is a drawing bug, not a ranking bug: AUPRC, ROC AUC, Fmax and sensitivity-to-first-FP
are each computed per species on the unpooled curve inside evaluate_domain_calls.py and
only averaged afterwards, so none of them inherit it.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def curve(species, points):
    return pl.DataFrame({
        "species": [species] * len(points),
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


def pooled():
    return bmi.pool_curve_over_species(SUB, "recall_reachable", "precision")


def test_the_line_stops_where_the_shortest_species_curve_stops():
    series = pooled()
    assert series
    assert max(float(x) for x in series) <= 0.12


def test_precision_never_recovers_mid_curve():
    series = pooled()
    ys = [series[k] for k in sorted(series, key=float)]
    running_min = ys[0]
    for y in ys[1:]:
        assert y <= running_min * 2.0, f"precision recovered mid-curve: {ys}"
        running_min = min(running_min, y)


def test_the_dropped_tail_would_have_shown_the_jump():
    # The old behaviour, kept here so the test says what it is protecting against: without
    # the guard the bin at recall 0.20 holds mouse alone and reads 0.62, up from 0.25 in
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


def test_species_that_all_reach_the_same_recall_are_not_truncated():
    same = pl.concat([
        curve("mouse", [(0.0, 0.9), (0.1, 0.5), (0.2, 0.3)]),
        curve("ecoli", [(0.0, 0.8), (0.1, 0.4), (0.2, 0.2)]),
    ])
    series = bmi.pool_curve_over_species(same, "recall_reachable", "precision")
    assert max(float(x) for x in series) == 0.2
    assert len(series) == 3


def test_a_frame_without_species_pools_to_nothing_rather_than_lying():
    no_species = SUB.drop("species")
    assert bmi.pool_curve_over_species(
        no_species, "recall_reachable", "precision") == {}
