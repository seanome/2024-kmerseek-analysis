"""Winning encoding vs divergence: alphabet size, ksize and bits per k-mer against Mya.

The claim the section exists to test is that the encoding kmerseek needs gets COARSER and
LONGER as the target proteome moves away from human, while the product -- k * log2(classes),
the information in one k-mer -- holds still. These tests fix the three things that would
make such a plot lie: the bits arithmetic, ranking distinct encodings rather than the same
one with the low-complexity filter flipped, and dropping an alphabet whose class count is
not knowable instead of defaulting it onto the bits axis.
"""
import json
import math
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def row(variant, species, mya, fmax, recall, split="selection", truth="pfam"):
    return {"tool": "kmerseek", "variant": variant, "species": species,
            # Float, because the real column is: evaluate_domain_calls writes --species-mya
            # as a float and the x-axis keys are str() of it, so an int fixture would key
            # the series "100" where the pipeline keys it "100.0".
            "species_mya": None if mya is None else float(mya),
            "truth_set": truth, "split": split,
            "stratum_axis": "all", "stratum": "all",
            "fmax": fmax, "recall_reachable": recall, "smin": 1.0 - fmax}


# A clean version of the hypothesis: close in, a 20-class alphabet at k=5 wins; far out, a
# 2-class alphabet at k=25 does. Both are ~21.6 and 25.0 bits, so the alphabet and k axes
# move a long way and the bits axis barely does.
NEAR_FAR = pl.DataFrame([
    row("protein20_k5_lcFalse",     "mouse", 100,  0.90, 0.80),
    row("hp_thomas_dill2_k25_lcFalse", "mouse", 100,  0.40, 0.30),
    row("gbmr4_k11_lcFalse",        "mouse", 100,  0.60, 0.50),
    row("protein20_k5_lcFalse",     "ecoli", 2000, 0.10, 0.05),
    row("hp_thomas_dill2_k25_lcFalse", "ecoli", 2000, 0.50, 0.60),
    row("gbmr4_k11_lcFalse",        "ecoli", 2000, 0.30, 0.40),
])


def build(tmp_path, metrics=NEAR_FAR, truth="pfam"):
    bmi.section_encoding_vs_divergence(tmp_path, metrics)
    plot = tmp_path / f"qfo_encoding_divergence_{truth}_mqc.json"
    table = tmp_path / f"qfo_encoding_divergence_table_{truth}_mqc.json"
    return (json.loads(plot.read_text()) if plot.exists() else None,
            json.loads(table.read_text()) if table.exists() else None)


# --- the arithmetic the whole section rests on --------------------------------------

def test_bits_is_k_times_log2_of_the_class_count():
    df = bmi.encoding_axes(bmi.parse_kmerseek_variants(pl.DataFrame([
        row("protein20_k5_lcFalse", "mouse", 100, 0.9, 0.8)])))
    assert df["n_classes"].to_list() == [20]
    assert df["bits_per_kmer"].to_list() == pytest.approx([5 * math.log2(20)])


def test_every_alphabet_this_run_swept_has_a_class_count():
    # The names seen in the midi run's report. A rename that dropped the trailing count
    # would put the alphabet on the bits axis at log2(10_000) per residue, which plots.
    for name, expected in {
            "protein20": 20, "uniprot18": 18, "hsdm17": 17, "sdm12": 12, "mmseqs12": 12,
            "wass14": 14, "funcgroups8": 8, "dayhoff6": 6, "wwmj5": 5, "gbmr4": 4,
            "polarity4": 4, "hp_lehninger_hpc3": 3, "hp_thomas_dill2": 2,
            "hp_thomas_dill_no_c2": 2, "hp_kyte_doolittle2": 2,
            "hp_pbotc_1st_ed2": 2}.items():
        assert bmi.alphabet_classes(name) == expected, name


def test_an_alphabet_with_no_class_count_is_dropped_and_named(capsys):
    df = bmi.encoding_axes(bmi.parse_kmerseek_variants(pl.DataFrame([
        row("protein20_k5_lcFalse", "mouse", 100, 0.9, 0.8),
        row("mystery_alphabet_k5_lcFalse", "mouse", 100, 0.99, 0.99)])))
    assert df["alphabet"].to_list() == ["protein20"], "unknown alphabet must not plot"
    assert "mystery_alphabet" in capsys.readouterr().out, "and must be named, not silent"


# --- the plot -----------------------------------------------------------------------

def test_three_panels_alphabet_size_ksize_and_bits(tmp_path):
    plot, _ = build(tmp_path)
    assert [d["name"] for d in plot["pconfig"]["data_labels"]] == [
        "Alphabet size", "K-mer length", "Bits per k-mer"]
    assert len(plot["data"]) == 3


def test_x_axis_is_divergence_time_not_species_name(tmp_path):
    plot, _ = build(tmp_path)
    for panel in plot["data"]:
        for series in panel.values():
            assert sorted(series) == ["100.0", "2000.0"]
    assert "Mya" in plot["pconfig"]["xlab"]


def test_the_winning_encoding_gets_coarser_and_longer_with_divergence(tmp_path):
    plot, _ = build(tmp_path)
    classes, ksize, bits = plot["data"]
    line = "Fmax (mean) · best"
    near, far = sorted(classes[line])[0], sorted(classes[line])[1]
    assert classes[line][near] == 20 and classes[line][far] == 2
    assert ksize[line][near] == 5 and ksize[line][far] == 25
    # And the bits axis is the one that barely moves, which is the point of the third panel.
    assert bits[line][near] == pytest.approx(5 * math.log2(20))
    assert bits[line][far] == pytest.approx(25.0)


def test_each_metric_gets_its_own_series_because_they_disagree(tmp_path):
    # On mouse, Fmax picks protein20 k5 and reachable recall picks it too; on ecoli they
    # both pick hp. What matters is that both are drawn rather than one chosen silently.
    plot, _ = build(tmp_path)
    names = set(plot["data"][0])
    assert "Fmax (mean) · best" in names
    assert any(n.startswith("Recall (reachable)") for n in names)


def test_the_three_ranks_of_one_metric_share_a_colour(tmp_path):
    plot, _ = build(tmp_path)
    colors = plot["pconfig"]["colors"]
    fmax = {c for n, c in colors.items() if n.startswith("Fmax")}
    recall = {c for n, c in colors.items() if n.startswith("Recall")}
    assert len(fmax) == 1 and len(recall) == 1 and fmax != recall


def test_runner_up_lines_are_drawn_so_a_lone_winner_cannot_pass_as_a_trend(tmp_path):
    plot, _ = build(tmp_path)
    names = set(plot["data"][0])
    assert {"Fmax (mean) · best", "Fmax (mean) · 2nd", "Fmax (mean) · 3rd"} <= names


# --- ranking distinct encodings, not the same one twice ------------------------------

def test_low_complexity_arms_of_one_encoding_are_one_point_not_three(tmp_path):
    # This is the failure that would make the band look tight for the wrong reason: with
    # raw combos ranked, ranks 1-3 are protein20 k5 lcF/lcT and one real runner-up.
    both_arms = pl.DataFrame([
        row("protein20_k5_lcFalse",     "mouse", 100, 0.90, 0.80),
        row("protein20_k5_lcTrue",      "mouse", 100, 0.89, 0.79),
        row("gbmr4_k11_lcFalse",        "mouse", 100, 0.60, 0.50),
        row("hp_thomas_dill2_k25_lcTrue", "mouse", 100, 0.40, 0.30),
    ])
    top = bmi.best_encodings_per_species(
        bmi.encoding_axes(bmi.parse_kmerseek_variants(both_arms)), "fmax")
    assert top.sort("rank")["alphabet"].to_list() == [
        "protein20", "gbmr4", "hp_thomas_dill2"]


def test_an_encoding_is_taken_at_its_better_low_complexity_arm(tmp_path):
    arms = pl.DataFrame([
        row("gbmr4_k11_lcFalse", "mouse", 100, 0.20, 0.10),
        row("gbmr4_k11_lcTrue",  "mouse", 100, 0.70, 0.60),
    ])
    top = bmi.best_encodings_per_species(
        bmi.encoding_axes(bmi.parse_kmerseek_variants(arms)), "fmax")
    assert top.height == 1 and top["fmax"].to_list() == [0.70]


def test_smin_ranks_the_other_way_round():
    # Semantic distance: the best encoding is the SMALLEST. Ranking it like the rest would
    # crown the worst one and nothing on the plot would look wrong.
    assert "smin" in bmi.LOWER_IS_BETTER
    top = bmi.best_encodings_per_species(
        bmi.encoding_axes(bmi.parse_kmerseek_variants(NEAR_FAR)), "smin")
    best = top.filter((pl.col("rank") == 0) & (pl.col("species") == "mouse"))
    assert best["alphabet"].to_list() == ["protein20"], "smin 0.10 is the best on mouse"


# --- the table beside the plot -------------------------------------------------------

def test_the_table_reports_how_far_clear_the_winner_was(tmp_path):
    _, table = build(tmp_path)
    assert table["data"]["mouse"]["fmax__margin"] == pytest.approx(0.30)
    assert table["data"]["mouse"]["fmax__enc"] == "protein20 k5"
    assert table["data"]["mouse"]["fmax__bits"] == pytest.approx(5 * math.log2(20))


def test_a_species_whose_winner_has_no_runner_up_gets_a_blank_margin(tmp_path):
    lone = pl.DataFrame([row("gbmr4_k11_lcFalse", "mouse", 100, 0.6, 0.5)])
    _, table = build(tmp_path, lone)
    assert table["data"]["mouse"]["fmax__margin"] is None


def test_table_rows_run_in_divergence_order_not_alphabetical(tmp_path):
    _, table = build(tmp_path)
    assert list(table["data"]) == ["mouse", "ecoli"]


# --- the conventions that run through the whole report -------------------------------

def test_truth_sets_are_never_pooled(tmp_path):
    mixed = pl.concat([
        NEAR_FAR,
        pl.DataFrame([row("protein20_k5_lcFalse", "mouse", 100, 0.11, 0.12,
                          split="all", truth="swissprot")]),
    ])
    bmi.section_encoding_vs_divergence(tmp_path, mixed)
    assert (tmp_path / "qfo_encoding_divergence_pfam_mqc.json").exists()
    assert (tmp_path / "qfo_encoding_divergence_swissprot_mqc.json").exists()
    sprot, _ = build(tmp_path, mixed, truth="swissprot")
    assert list(sprot["data"][0]["Fmax (mean) · best"]) == ["100.0"], "pfam rows leaked in"


def test_the_selection_split_is_preferred_over_heldout(tmp_path):
    # This section is for CHOOSING an encoding, so it must not read the reporting half.
    mixed = pl.concat([
        NEAR_FAR,
        pl.DataFrame([row("hsdm17_k9_lcFalse", "mouse", 100, 0.99, 0.99,
                          split="heldout")]),
    ])
    plot, table = build(tmp_path, mixed)
    assert table["data"]["mouse"]["fmax__enc"] == "protein20 k5"
    assert plot["data"][0]["Fmax (mean) · best"]["100.0"] == 20


def test_the_hmmscan_ceiling_species_is_not_placed_on_the_divergence_axis(tmp_path):
    # `all` is the ceiling arm, which reads no target proteome and so has no Mya.
    with_ceiling = pl.concat([
        NEAR_FAR,
        pl.DataFrame([row("protein20_k5_lcFalse", "all", None, 0.99, 0.99)]),
    ], how="diagonal_relaxed")
    plot, table = build(tmp_path, with_ceiling)
    assert "all" not in table["data"]
    assert sorted(plot["data"][0]["Fmax (mean) · best"]) == ["100.0", "2000.0"]


def test_non_kmerseek_tools_are_not_ranked_here(tmp_path):
    with_baseline = pl.concat([
        NEAR_FAR,
        pl.DataFrame([{**row("single_seq", "mouse", 100, 0.99, 0.99), "tool": "hhblits"}]),
    ])
    plot, _ = build(tmp_path, with_baseline)
    assert plot["data"][0]["Fmax (mean) · best"]["100.0"] == 20


def test_a_run_with_no_divergence_column_writes_nothing(tmp_path):
    bmi.section_encoding_vs_divergence(tmp_path, NEAR_FAR.drop("species_mya"))
    assert not list(tmp_path.glob("qfo_encoding_divergence*"))
