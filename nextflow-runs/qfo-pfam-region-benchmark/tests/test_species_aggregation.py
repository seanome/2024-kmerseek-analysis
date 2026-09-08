"""Across-species summaries, and the aggregation-vs-total distinction they encode.

Every metrics row is one (tool, variant, TARGET SPECIES, ...), so a leaderboard number has
always been collapsed over species. Which collapse it was used to be invisible: `fmax` was
a mean and `n_proteins_ranked` was also a mean, while nothing offered a sum at all. The
suffix makes the choice explicit, and these tests pin it -- particularly that a `__total`
over a per-species count is NOT the size of the answer key.
"""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402

SPECIES = [("mouse", 0.30, 100), ("yeast", 0.20, 200), ("ecoli", 0.10, 300)]


def metrics() -> pl.DataFrame:
    rows = []
    for sp, fmax, calls in SPECIES:
        rows.append({
            "truth_set": "pfam", "split": "all", "tool": "kmerseek",
            "variant": "hp_pbotc_1st_ed2_k19_lcFalse", "species": sp,
            "target_species": sp, "stratum_axis": "all", "stratum": "all",
            "fmax": fmax, "recall_reachable": fmax / 2, "smin": 1 - fmax,
            "n_calls": calls, "n_truth_instances": 1_000, "n_proteins_ranked": calls,
        })
    return pl.DataFrame(rows)


@pytest.fixture
def board():
    return bmi.best_variants(bmi.ungrouped(metrics()))


def test_every_headline_metric_gets_the_five_number_summary(board):
    for metric in ("fmax", "recall_reachable", "smin"):
        for agg in bmi.SPECIES_AGGREGATIONS:
            assert f"{metric}__{agg}" in board.columns, f"{metric}__{agg} missing"


def test_the_aggregations_are_over_species_and_weight_them_equally(board):
    r = board.to_dicts()[0]
    assert r["fmax__mean"] == pytest.approx(0.20)      # not weighted by call count
    assert r["fmax__median"] == pytest.approx(0.20)
    assert r["fmax__min"] == pytest.approx(0.10)
    assert r["fmax__max"] == pytest.approx(0.30)
    assert r["fmax__sd"] == pytest.approx(0.10)


def test_a_total_is_a_sum_and_a_mean_is_not(board):
    r = board.to_dicts()[0]
    assert r["n_calls__total"] == 600            # 100 + 200 + 300
    assert r["n_calls__mean"] == pytest.approx(200.0)
    assert r["n_calls__total"] != r["n_calls__mean"]


def test_an_instance_total_is_not_the_size_of_the_answer_key(board):
    """The regression this whole convention exists for. Each of the three species scores
    the same 1_000 human instances, so the total is 3_000 -- three times the answer key,
    not the answer key. A reader must not be able to reach that number without a column
    name that says TOTAL."""
    r = board.to_dicts()[0]
    assert r["n_truth_instances__total"] == 3_000
    assert r["n_truth_instances__mean"] == pytest.approx(1_000.0)
    h = bmi.fmt_metric_headers(["n_truth_instances__total"])["n_truth_instances__total"]
    assert "TOTAL" in h["description"]
    assert "not the size of the answer key" in h["description"]


def test_rates_offer_no_total(board):
    """A summed Fmax has no interpretation, so the column must not exist to be misread."""
    for metric in ("fmax", "recall_reachable", "smin", "precision"):
        assert f"{metric}__total" not in board.columns


def test_bare_metric_stays_the_mean_so_existing_sections_are_unchanged(board):
    r = board.to_dicts()[0]
    assert r["fmax"] == pytest.approx(r["fmax__mean"])
    # n_proteins_ranked is the one count whose BARE name is a mean, because every existing
    # caption means the mean by it. Its total is available under the suffix.
    assert r["n_proteins_ranked"] == pytest.approx(200.0)
    assert r["n_proteins_ranked__total"] == 600


def test_legacy_fmax_spread_columns_still_agree_with_the_suffixed_ones(board):
    r = board.to_dicts()[0]
    assert r["fmax_sd"] == pytest.approx(r["fmax__sd"])
    assert r["fmax_min"] == pytest.approx(r["fmax__min"])
    assert r["fmax_max"] == pytest.approx(r["fmax__max"])


def test_sd_over_one_species_is_null_not_zero():
    """A single-proteome run has no spread to report, and 0 would read as perfect
    consistency across species that were never compared."""
    one = metrics().filter(pl.col("species") == "mouse")
    r = bmi.best_variants(bmi.ungrouped(one)).to_dicts()[0]
    assert r["fmax__sd"] is None
    assert r["n_species"] == 1


def test_headers_name_the_collapse_and_hide_all_but_the_primary():
    h = bmi.fmt_metric_headers(
        ["fmax", "fmax__median", "fmax__sd", "smin__min", "n_calls__total"])
    assert h["fmax__median"]["title"] == "Fmax (median)"       # not "Fmax (mean) (median)"
    assert h["fmax__sd"]["title"] == "Fmax (SD)"
    assert h["smin__min"]["title"] == "Smin (min)"
    assert h["n_calls__total"]["title"] == "Calls (total)"     # not "n_calls (total)"
    assert "AGGREGATION" in h["fmax__median"]["description"]
    assert all(h[c].get("hidden") for c in
               ("fmax__median", "fmax__sd", "smin__min", "n_calls__total"))
    assert not h["fmax"].get("hidden")


def test_sd_drops_the_metrics_own_bounds():
    """An SD is a spread, not a score, so a 0-1 clamp and a red-green ramp would both lie."""
    h = bmi.fmt_metric_headers(["fmax__sd"])["fmax__sd"]
    assert "min" not in h and "max" not in h
