"""Recognition and placement Fmax against alphabet cardinality, on a log x axis.

The claim this section exists to make is that family recognition falls as the alphabet gets
finer while interval placement does not follow it down anything like as far. These tests
fix the four things that would let such a plot lie:

  * the averaging, which has to be identical to section_ceiling_recognition's or the two
    sections would report different heights for the same quantity;
  * the spread, because a monotone trend asserted over nineteen means with no dispersion
    reported is the shape a reviewer attacks first, and this report estimates no sampling
    error anywhere;
  * the class count, which is read off the alphabet name and must drop rather than default
    an unparsable one;
  * the plot payload's SHAPE, since MultiQC's custom content reads a top-level list as a
    list of datasets and crashes the whole report on a bare list of points.
"""
import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def row(variant, species, fmax, family_fmax, split="all", truth="swissprot",
        tool="kmerseek"):
    return {"tool": tool, "variant": variant, "species": species, "truth_set": truth,
            "split": split, "stratum_axis": "all", "stratum": "all",
            "fmax": fmax, "family_fmax": family_fmax,
            # section_ceiling_recognition reports these as medians, and the agreement test
            # below runs that function on the same frame.
            "n_family_truth": 100, "n_family_calls": 80, "coverage": 0.9}


# Two 2-class alphabets that stack at x=2, one 4-class, and one 20-class built from a
# single cell so the "no spread to report" branch is exercised. The 2-class alphabet is
# swept over two k and both low-complexity arms, which is the axis the mean collapses.
SWEEP = pl.DataFrame([
    row("hp_thomas_dill2_k18_lcFalse", "yeast", 0.10, 0.40),
    row("hp_thomas_dill2_k18_lcTrue", "yeast", 0.12, 0.44),
    row("hp_thomas_dill2_k20_lcFalse", "yeast", 0.08, 0.36),
    row("hp_thomas_dill2_k20_lcTrue", "yeast", 0.10, 0.40),
    row("hp_lehninger2_k18_lcFalse", "yeast", 0.09, 0.34),
    row("hp_lehninger2_k18_lcFalse", "ecoli", 0.07, 0.30),
    row("gbmr4_k12_lcFalse", "yeast", 0.08, 0.26),
    row("gbmr4_k14_lcFalse", "yeast", 0.06, 0.22),
    row("protein20_k10_lcFalse", "yeast", 0.05, 0.14),
])


def build(tmp_path, metrics=SWEEP, truth="swissprot"):
    bmi.section_ceiling_cardinality(tmp_path, metrics)
    plot = tmp_path / f"qfo_ceiling_cardinality_{truth}_mqc.json"
    table = tmp_path / f"qfo_ceiling_cardinality_table_{truth}_mqc.json"
    return (json.loads(plot.read_text()) if plot.exists() else None,
            json.loads(table.read_text()) if table.exists() else None)


def means(plot, group_prefix):
    """{alphabet: y} for the mean markers of one series, ignoring the min/max caps."""
    out = {}
    for series in plot["data"].values():
        for point in series:
            if point["group"].startswith(group_prefix) and "annotation" in point:
                out[point["annotation"]] = point["y"]
    return out


RECOGNITION = "Recognition (family Fmax"
PLACEMENT = "Placement (Fmax"


# --- the statistics helpers ---------------------------------------------------------

def test_tied_ranks_are_averaged_not_taken_in_list_order():
    # Six of the sweep's nineteen alphabets have two classes, so ties on x are the common
    # case. Ranking them by position would make the correlation depend on the order the
    # alphabets happened to be listed in.
    assert bmi.rank_avg([2, 2, 4]) == [0.5, 0.5, 2.0]
    assert bmi.rank_avg([5, 1, 3]) == [2.0, 0.0, 1.0]


def test_spearman_is_minus_one_on_a_perfectly_decreasing_series():
    assert bmi.spearman_rho([2, 4, 8, 20], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(-1.0)
    assert bmi.spearman_rho([2, 4, 8, 20], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_spearman_is_none_when_it_cannot_be_computed():
    assert bmi.spearman_rho([2, 4], [0.4, 0.3]) is None, "too few points"
    assert bmi.spearman_rho([2, 4, 8], [0.3, 0.3, 0.3]) is None, "y is constant"


def test_pairs_at_the_same_cardinality_count_neither_way():
    # The two 2-class alphabets disagree with each other; that says nothing about the
    # direction of the trend and must not be counted as an inversion.
    down, up = bmi.descending_pairs([2, 2, 4], [0.40, 0.30, 0.20])
    assert (down, up) == (2, 0)


def test_a_pair_running_the_wrong_way_is_counted():
    down, up = bmi.descending_pairs([2, 4, 8], [0.40, 0.20, 0.30])
    assert (down, up) == (2, 1)


# --- the averaging, which must match the section it sits beside ----------------------

def test_one_point_per_alphabet_averaged_over_ksize_arm_and_species(tmp_path):
    plot, _ = build(tmp_path)
    rec = means(plot, RECOGNITION)
    assert set(rec) == {"hp_thomas_dill2", "hp_lehninger2", "gbmr4", "protein20"}
    # 0.40, 0.44, 0.36, 0.40 over two k and both arms.
    assert rec["hp_thomas_dill2"] == pytest.approx(0.40)
    # 0.34 and 0.30 over two target species.
    assert rec["hp_lehninger2"] == pytest.approx(0.32)
    assert means(plot, PLACEMENT)["hp_thomas_dill2"] == pytest.approx(0.10)


def test_it_reports_the_same_numbers_as_the_bar_chart_beside_it(tmp_path):
    # The two sections plot the same two columns. If their averaging ever diverges, a
    # reader comparing them sees two different heights for one measurement and neither
    # section is wrong on its own face.
    plot, _ = build(tmp_path)
    bmi.section_ceiling_recognition(tmp_path, SWEEP, "swissprot")
    bars = json.loads(
        (tmp_path / "qfo_ceiling_recognition_mqc.json").read_text())["data"][0]
    for alphabet, y in means(plot, RECOGNITION).items():
        assert bars[alphabet]["family_fmax"] == pytest.approx(y), alphabet
    for alphabet, y in means(plot, PLACEMENT).items():
        assert bars[alphabet]["fmax"] == pytest.approx(y), alphabet


def test_a_cell_missing_either_metric_is_dropped_from_both(tmp_path):
    # Otherwise the mean of one series is over more cells than the other and the vertical
    # distance between a point pair stops being a like-for-like difference.
    partial = pl.concat([SWEEP, pl.DataFrame([
        row("gbmr4_k16_lcFalse", "yeast", 0.99, None)])])
    plot, table = build(tmp_path, partial)
    assert means(plot, PLACEMENT)["gbmr4"] == pytest.approx(0.07)
    assert table["data"]["gbmr4"]["n_cells"] == 2


def test_a_run_with_no_family_fmax_at_all_writes_nothing_rather_than_crashing(tmp_path):
    # The shape of the mini result set, whose kmerseek arms predate the family metrics.
    blank = SWEEP.with_columns(pl.lit(None, dtype=pl.Float64).alias("family_fmax"))
    bmi.section_ceiling_cardinality(tmp_path, blank)
    assert not list(tmp_path.glob("qfo_ceiling_cardinality*"))
    bmi.section_ceiling_cardinality(tmp_path, SWEEP.drop("family_fmax"))
    assert not list(tmp_path.glob("qfo_ceiling_cardinality*"))


# --- the spread, which is the whole reason this is not just a line of means ----------

def test_every_averaged_point_carries_its_min_and_max(tmp_path):
    plot, _ = build(tmp_path)
    caps = sorted(p["y"] for series in plot["data"].values() for p in series
                  if p["name"] == "hp_thomas_dill2 min/max"
                  and p["group"].startswith(RECOGNITION))
    assert caps == pytest.approx([0.36, 0.44])


def test_an_alphabet_with_one_cell_draws_no_caps_and_reports_no_sd(tmp_path):
    plot, table = build(tmp_path)
    assert not [p for series in plot["data"].values() for p in series
                if p["name"] == "protein20 min/max"]
    assert table["data"]["protein20"]["recognition_sd"] is None
    assert table["data"]["protein20"]["n_cells"] == 1


def test_the_table_carries_the_spread_beside_every_mean(tmp_path):
    _, table = build(tmp_path)
    r = table["data"]["hp_thomas_dill2"]
    assert r["recognition_mean"] == pytest.approx(0.40)
    assert r["recognition_min"] == pytest.approx(0.36)
    assert r["recognition_max"] == pytest.approx(0.44)
    assert r["placement_min"] == pytest.approx(0.08)
    assert r["placement_max"] == pytest.approx(0.12)
    assert r["gap"] == pytest.approx(0.30)
    assert r["n_cells"] == 4


def test_the_description_says_the_spread_is_not_a_confidence_interval(tmp_path):
    # This report estimates no sampling error anywhere, and the section must not let a
    # reader mistake a min/max cap for one.
    plot, table = build(tmp_path)
    assert "not a confidence interval" in plot["description"].lower()
    assert "sampling error" in plot["description"]
    assert "sampling error" in table["description"]


def test_the_description_states_the_trend_with_its_inversions(tmp_path):
    plot, _ = build(tmp_path)
    assert "Spearman rho" in plot["description"]
    assert "run the other way" in plot["description"]


# --- the axis and the labels ---------------------------------------------------------

def test_x_is_alphabet_size_on_a_log_axis(tmp_path):
    plot, _ = build(tmp_path)
    assert plot["pconfig"]["xlog"] is True
    assert "classes" in plot["pconfig"]["xlab"]
    xs = {p["x"] for series in plot["data"].values() for p in series}
    assert xs == {2, 4, 20}


def test_the_x_bounds_are_outside_the_data_so_no_alphabet_is_dropped(tmp_path):
    # MultiQC's scatter DROPS points outside xmin/xmax instead of clipping the axis, so a
    # bound set inside the data would silently lose an alphabet rather than look wrong.
    plot, _ = build(tmp_path)
    xs = [p["x"] for series in plot["data"].values() for p in series]
    assert plot["pconfig"]["xmin"] < min(xs) and plot["pconfig"]["xmax"] > max(xs)


def test_alphabets_stacked_at_one_cardinality_are_each_named(tmp_path):
    # Six alphabets share x=2 in the real sweep; an unlabelled point there cannot be
    # identified at all, which is the one thing this axis makes worse than a bar chart.
    plot, _ = build(tmp_path)
    at_two = {p["annotation"] for series in plot["data"].values() for p in series
              if p["x"] == 2 and p["group"].startswith(RECOGNITION)
              and "annotation" in p}
    assert at_two == {"hp_thomas_dill2", "hp_lehninger2"}


def test_each_cardinality_present_gets_a_labelled_rule(tmp_path):
    # Plotly's log axis labels 20 as a bare "2", which on an axis whose left end really is
    # 2 is worse than no label.
    plot, _ = build(tmp_path)
    assert [line["label"] for line in plot["pconfig"]["x_lines"]] == ["2", "4", "20"]


def test_the_two_series_are_drawn_in_different_colours(tmp_path):
    plot, _ = build(tmp_path)
    colors = {p["group"].split(" —")[0]: p["color"]
              for series in plot["data"].values() for p in series}
    assert len(set(colors.values())) == 2


def test_the_plot_payload_is_a_dict_of_series_not_a_list_of_points(tmp_path):
    # A top-level list is read by custom content as a list of DATASETS, and its numeric-x
    # coercion then raises a TypeError that aborts the entire report, not just this
    # section.
    plot, _ = build(tmp_path)
    assert isinstance(plot["data"], dict)
    assert all(isinstance(v, list) for v in plot["data"].values())


# --- class counts, and the conventions that run through the whole report -------------

def test_an_alphabet_with_no_class_count_is_left_out_and_named(tmp_path, capsys):
    mystery = pl.concat([SWEEP, pl.DataFrame([
        row("mystery_alphabet_k9_lcFalse", "yeast", 0.99, 0.99)])])
    plot, _ = build(tmp_path, mystery)
    assert "mystery_alphabet" not in means(plot, RECOGNITION)
    assert "mystery_alphabet" in capsys.readouterr().out


def test_non_kmerseek_tools_are_not_drawn(tmp_path):
    with_baseline = pl.concat([SWEEP, pl.DataFrame([
        row("default", "yeast", 0.9, 0.9, tool="hmmscan")])])
    plot, _ = build(tmp_path, with_baseline)
    assert set(means(plot, RECOGNITION)) == {
        "hp_thomas_dill2", "hp_lehninger2", "gbmr4", "protein20"}


def test_truth_sets_are_never_pooled(tmp_path):
    mixed = pl.concat([SWEEP, pl.DataFrame([
        row("gbmr4_k12_lcFalse", "yeast", 0.99, 0.99, split="heldout", truth="pfam")])])
    bmi.section_ceiling_cardinality(tmp_path, mixed)
    assert (tmp_path / "qfo_ceiling_cardinality_swissprot_mqc.json").exists()
    assert (tmp_path / "qfo_ceiling_cardinality_pfam_mqc.json").exists()
    sprot, _ = build(tmp_path, mixed)
    assert sprot["description"].count("swissprot truth") == 1
    assert means(sprot, RECOGNITION)["gbmr4"] == pytest.approx(0.24), "pfam rows leaked in"


def test_the_heldout_split_is_used_where_the_truth_set_has_one(tmp_path):
    # Every reporting section reads the heldout half; only the encoding-choice sections
    # read the selection half. This one reports.
    split = pl.DataFrame([
        row("gbmr4_k12_lcFalse", "yeast", 0.08, 0.26, split="selection", truth="pfam"),
        row("gbmr4_k12_lcFalse", "ecoli", 0.02, 0.10, split="heldout", truth="pfam"),
    ])
    plot, _ = build(tmp_path, split, truth="pfam")
    assert means(plot, RECOGNITION)["gbmr4"] == pytest.approx(0.10)
    assert "<code>heldout</code>" in plot["description"]


def test_only_the_ungrouped_rows_are_averaged(tmp_path):
    # Stratified rows are the same measurement cut by protein length, pLDDT and so on.
    # Averaging them in would weight an alphabet by how many strata it happened to
    # populate.
    strata = pl.concat([SWEEP, pl.DataFrame([
        {**row("gbmr4_k12_lcFalse", "yeast", 0.99, 0.99),
         "stratum_axis": "feature_length_bin", "stratum": "short"}])])
    plot, _ = build(tmp_path, strata)
    assert means(plot, RECOGNITION)["gbmr4"] == pytest.approx(0.24)


def test_table_rows_run_coarsest_first_not_alphabetically(tmp_path):
    _, table = build(tmp_path)
    assert list(table["data"]) == [
        "hp_lehninger2", "hp_thomas_dill2", "gbmr4", "protein20"]
    assert table["pconfig"]["sort_rows"] is False


def test_the_sections_carry_the_ceiling_parent(tmp_path):
    plot, table = build(tmp_path)
    assert plot["parent_id"] == bmi.CEILING_PARENT["parent_id"]
    assert table["parent_id"] == bmi.CEILING_PARENT["parent_id"]
