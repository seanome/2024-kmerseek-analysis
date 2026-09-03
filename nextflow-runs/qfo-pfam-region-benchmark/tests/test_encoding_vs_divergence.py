"""The encoding score surface, and the bits budget read off it.

The claim these panels exist to test is that the encoding kmerseek needs gets COARSER and
LONGER as the target proteome moves away from human, while the product -- k * log2(classes),
the information in one k-mer -- holds still.

What this block used to draw was an ARGMAX: the best encoding per proteome, its class
count, its k and its bits, three metrics times three ranks, nine crossing lines behind a
truth-set switcher. The winner's margin over the runner-up is routinely in the third
decimal across a sweep this size, so that figure plotted the least stable summary of the
surface; alphabet size and k do not interpolate between proteomes, so the connecting lines
asserted encodings that do not exist; and the score distribution -- the thing that says
whether the winner means anything -- was discarded before drawing.

Two panels replace it. Panel A is the surface itself, every encoding against every
proteome. Panel B is the budget as a number: per proteome, the encodings within one
standard error of that proteome's best, strip-plotted on the bits axis with their median,
so the caption can quote the height with an interval or say there is no single height.

These tests fix the arithmetic both panels rest on, the shape of each panel, and the
conventions that run through the report.
"""
import json
import math
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def row(variant, species, mya, fmax, recall, split="selection", truth="pfam",
        reachable=10_000):
    return {"tool": "kmerseek", "variant": variant, "species": species,
            # The denominator one_se divides by. Without it the bits-budget panel has no
            # noise scale and writes nothing, which is the honest behaviour and not what
            # these fixtures are testing.
            "n_reachable_instances": reachable,
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
    """(Panel A, Panel B) for one truth set, or (None, None) where a panel is not written."""
    bmi.section_encoding_vs_divergence(tmp_path, metrics)
    surface = tmp_path / f"qfo_encoding_divergence_{truth}_mqc.json"
    budget = tmp_path / f"qfo_bits_budget_{truth}_mqc.json"
    return (json.loads(surface.read_text()) if surface.exists() else None,
            json.loads(budget.read_text()) if budget.exists() else None)


def cell(surface, encoding, species):
    """Panel A's value for one encoding against one proteome."""
    col = next(i for i, c in enumerate(surface["xcats"]) if c.startswith(species))
    return surface["data"][surface["ycats"].index(encoding)][col]


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


# --- Panel A, the score surface -----------------------------------------------------

def test_the_surface_is_a_heatmap_of_every_encoding_against_every_proteome(tmp_path):
    surface, _ = build(tmp_path)
    assert surface["plot_type"] == "heatmap"
    assert sorted(surface["ycats"]) == sorted(
        ["protein20 k5", "gbmr4 k11", "hp_thomas_dill2 k25"])
    assert len(surface["xcats"]) == 2
    assert len(surface["data"]) == 3 and len(surface["data"][0]) == 2


def test_encodings_are_sorted_by_bits_so_a_band_is_readable(tmp_path):
    surface, _ = build(tmp_path)
    # protein20 k5 is 5 * log2(20) = 21.6 bits, gbmr4 k11 is 22, hp_thomas_dill2 k25
    # is 25. Low at the bottom.
    assert surface["ycats"] == ["protein20 k5", "gbmr4 k11", "hp_thomas_dill2 k25"]


def test_proteomes_run_in_divergence_order(tmp_path):
    surface, _ = build(tmp_path)
    assert surface["xcats"][0].startswith("mouse")
    assert surface["xcats"][1].startswith("ecoli")


def test_the_tick_label_carries_the_denominator(tmp_path):
    # The reachable count varies between proteomes, so the same score is a different
    # amount of evidence in each column.
    surface, _ = build(tmp_path)
    assert "10,000" in surface["xcats"][0]


def test_the_colour_range_comes_from_the_data_not_from_the_metric_bound(tmp_path):
    surface, _ = build(tmp_path)
    assert surface["pconfig"]["max"] == pytest.approx(0.90)
    assert surface["pconfig"]["colstops"] == bmi.SEQUENTIAL_COLSTOPS


def test_the_cells_are_the_measured_scores(tmp_path):
    surface, _ = build(tmp_path)
    assert cell(surface, "protein20 k5", "mouse") == pytest.approx(0.90)
    assert cell(surface, "hp_thomas_dill2 k25", "ecoli") == pytest.approx(0.50)


def test_the_per_proteome_best_is_named_rather_than_drawn_as_its_own_series(tmp_path):
    # An argmax is one cell of the surface, not a line across it. MultiQC's heatmap has no
    # annotation layer, so the winner is named in the caption and marked in Panel B.
    surface, _ = build(tmp_path)
    assert "protein20 k5" in surface["description"]
    assert "hp_thomas_dill2 k25" in surface["description"]


# --- Panel B, the bits budget --------------------------------------------------------

def test_the_budget_panel_is_points_with_a_median_rule(tmp_path):
    _, budget = build(tmp_path)
    assert budget["plot_type"] == "scatter"
    assert "bits per k-mer" in budget["pconfig"]["ylab"]
    assert budget["pconfig"]["y_lines"][0]["label"].startswith("pooled median")


def test_the_budget_panel_holds_only_encodings_within_one_se_of_best(tmp_path):
    # On mouse, protein20 k5 wins at 0.90 and the SE over 10_000 instances is 0.003, so
    # nothing else is inside the band. The strip is a set, not a top-N.
    _, budget = build(tmp_path)
    mouse = [k for k in budget["data"] if k.startswith("mouse")]
    assert mouse == ["mouse protein20 k5"]
    assert "median mouse" in budget["data"], "the median of the strip is drawn too"


def test_a_noisier_denominator_widens_the_strip(tmp_path):
    # On ecoli the best is 0.50; at n=5 the SE is 0.224, so gbmr4 at 0.30 comes inside the
    # band and the winner stops standing alone. That widening is what the panel is for.
    noisy = pl.DataFrame([{**r, "n_reachable_instances": 5} for r in NEAR_FAR.to_dicts()])
    _, budget = build(tmp_path, noisy)
    ecoli = [k for k in budget["data"] if k.startswith("ecoli")]
    assert len(ecoli) == 2


def test_the_argmax_is_the_marked_point_not_the_whole_figure(tmp_path):
    _, budget = build(tmp_path)
    best = budget["data"]["mouse protein20 k5"]
    assert best["group"] == "best encoding"
    assert best["annotation"] == "protein20 k5"
    assert best["y"] == pytest.approx(5 * math.log2(20))


def test_the_caption_quotes_the_interval_rather_than_asserting_a_budget(tmp_path):
    _, budget = build(tmp_path)
    text = budget["description"]
    assert "interquartile range" in text
    assert ("The strips sit at the same height" in text
            or "The strips do not sit at one height" in text)


def test_x_axis_is_divergence_time_not_species_name(tmp_path):
    _, budget = build(tmp_path)
    assert "Mya" in budget["pconfig"]["xlab"]
    xs = {round(v["x"]) for k, v in budget["data"].items() if k.startswith("median")}
    assert xs == {100, 2000}


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


def test_the_surface_also_collapses_the_low_complexity_arm(tmp_path):
    arms = pl.DataFrame([
        row("gbmr4_k11_lcFalse", "mouse", 100, 0.20, 0.10),
        row("gbmr4_k11_lcTrue",  "mouse", 100, 0.70, 0.60),
    ])
    surface, _ = build(tmp_path, arms)
    assert surface["ycats"] == ["gbmr4 k11"]
    assert cell(surface, "gbmr4 k11", "mouse") == pytest.approx(0.70)


def test_smin_ranks_the_other_way_round():
    # Semantic distance: the best encoding is the SMALLEST. Ranking it like the rest would
    # crown the worst one and nothing on the plot would look wrong.
    assert "smin" in bmi.LOWER_IS_BETTER
    top = bmi.best_encodings_per_species(
        bmi.encoding_axes(bmi.parse_kmerseek_variants(NEAR_FAR)), "smin")
    best = top.filter((pl.col("rank") == 0) & (pl.col("species") == "mouse"))
    assert best["alphabet"].to_list() == ["protein20"], "smin 0.10 is the best on mouse"


# --- the noise scale both panels use -------------------------------------------------

def test_one_se_is_the_binomial_scale_and_shrinks_with_the_denominator():
    assert bmi.one_se(0.5, 100) == pytest.approx(0.05)
    assert bmi.one_se(0.5, 10_000) == pytest.approx(0.005)


def test_one_se_is_none_without_a_denominator_rather_than_a_fabricated_zero():
    assert bmi.one_se(0.5, None) is None
    assert bmi.one_se(0.5, 0) is None
    assert bmi.one_se(None, 100) is None


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
    assert len(sprot["xcats"]) == 1, "pfam rows leaked in"
    assert cell(sprot, "protein20 k5", "mouse") == pytest.approx(0.11)


def test_the_truth_set_the_report_does_not_lead_on_is_marked_supplementary(tmp_path):
    mixed = pl.concat([
        NEAR_FAR,
        pl.DataFrame([row("protein20_k5_lcFalse", "mouse", 100, 0.11, 0.12,
                          split="all", truth="swissprot")]),
    ])
    surface, _ = build(tmp_path, mixed)
    assert surface["parent_id"] == bmi.SUPP_PARENT_ID
    assert surface["section_name"].startswith("Supp: ")
    primary, _ = build(tmp_path, mixed, truth=bmi.PRIMARY_TRUTH)
    assert primary["parent_id"] == "qfo_region"


def test_the_selection_split_is_preferred_over_heldout(tmp_path):
    # This section is for CHOOSING an encoding, so it must not read the reporting half.
    mixed = pl.concat([
        NEAR_FAR,
        pl.DataFrame([row("hsdm17_k9_lcFalse", "mouse", 100, 0.99, 0.99,
                          split="heldout")]),
    ])
    surface, _ = build(tmp_path, mixed)
    assert "hsdm17 k9" not in surface["ycats"]


def test_the_hmmscan_ceiling_species_is_not_placed_on_the_divergence_axis(tmp_path):
    # `all` is the ceiling arm, which reads no target proteome and so has no Mya.
    with_ceiling = pl.concat([
        NEAR_FAR,
        pl.DataFrame([row("protein20_k5_lcFalse", "all", None, 0.99, 0.99)]),
    ], how="diagonal_relaxed")
    surface, _ = build(tmp_path, with_ceiling)
    assert not any(c.startswith("all") for c in surface["xcats"])
    assert len(surface["xcats"]) == 2


def test_non_kmerseek_tools_are_not_ranked_here(tmp_path):
    with_baseline = pl.concat([
        NEAR_FAR,
        pl.DataFrame([{**row("single_seq", "mouse", 100, 0.99, 0.99), "tool": "hhblits"}]),
    ])
    surface, _ = build(tmp_path, with_baseline)
    assert cell(surface, "protein20 k5", "mouse") == pytest.approx(0.90)


def test_a_run_with_no_divergence_column_writes_nothing(tmp_path):
    bmi.section_encoding_vs_divergence(tmp_path, NEAR_FAR.drop("species_mya"))
    assert not list(tmp_path.glob("qfo_encoding_divergence*"))
    assert not list(tmp_path.glob("qfo_bits_budget*"))
