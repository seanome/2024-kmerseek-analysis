"""The Divergence-retention section drew nineteen zeroes, and the zeroes were honest.

On the midi-plus run the Pfam-N section came out as a flat row of nineteen kmerseek points
at exactly y = 0.0 across every alphabet size, with the baselines beside them on real
values (reseek 26%, prostt5 25%, hhblits 16%). That asymmetry was not a join dropping
kmerseek rows and not a null read as zero. Every kmerseek arm HAS rows at the far end --
812 of them at E. coli for Pfam-N -- and every one carries n_tp_calls 0, so `fmax` is
exactly 0.0 and `fmax_far / fmax_near` is exactly 0.0 nineteen times over.

The section already refused to divide BY zero and said so in the report instead. It had no
matching refusal for a numerator that is zero at every alphabet at once, which is what a
proteome past the whole method's detection floor looks like. The result is a plot whose y
axis carries no information about its own x axis, sitting under a caption that reads the
gap to the baselines as a fact about alphabets.

These tests pin the endpoint rule: step inward to the furthest proteome where the plotted
encodings can still be measured, move the baselines to that same pair so the comparison
stays like-for-like, and name the proteome that was stepped over. An arm that is zero
where others are not keeps its zero -- that difference is the section's whole subject.

Plus the label overlap that shipped with it: MultiQC 1.35 renders a y_lines label as a
centred plotly shape label and drops every positioning field, so two baselines a
percentage point apart printed their names on top of each other.
"""
import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


SPECIES = {100.0: "mouse", 900.0: "yeast", 1500.0: "arabidopsis", 2000.0: "ecoli"}


def row(tool, variant, mya, fmax, truth="pfamn"):
    return {"truth_set": truth, "tool": tool, "variant": variant,
            "species": SPECIES[mya], "species_mya": float(mya), "split": "all",
            "stratum_axis": "all", "stratum": "all", "fmax": fmax}


def arm(variant, near, mid, far):
    """One kmerseek encoding at the three proteomes the fixtures use."""
    return [row("kmerseek", variant, 100, near),
            row("kmerseek", variant, 1500, mid),
            row("kmerseek", variant, 2000, far)]


# The midi-plus shape: both alphabets score at mouse, both still score at arabidopsis and
# by different amounts, and both are flat zero at E. coli. gbmr4 keeps 3.5% of its mouse
# score out to arabidopsis and protein20 keeps 0.6% -- a factor of six on the alphabet
# axis that the E. coli endpoint erased. The baselines score at every proteome.
FLOORED = pl.DataFrame(
    arm("gbmr4_k17_lcFalse", 0.0340, 0.00120, 0.0)
    + arm("protein20_k6_lcFalse", 0.0330, 0.00020, 0.0)
    + [row("foldseek", "-", 100, 0.0400), row("foldseek", "-", 1500, 0.0290),
       row("foldseek", "-", 2000, 0.0130),
       row("hhblits", "-", 100, 0.0620), row("hhblits", "-", 1500, 0.0359),
       row("hhblits", "-", 2000, 0.0100)])


def build(tmp_path, metrics=FLOORED, truth="pfamn"):
    bmi.section_alphabet_retention(tmp_path, metrics)
    plot = tmp_path / f"qfo_retention_{truth}_mqc.json"
    table = tmp_path / f"qfo_retention_table_{truth}_mqc.json"
    return (json.loads(plot.read_text()) if plot.exists() else None,
            json.loads(table.read_text()) if table.exists() else None)


# --- the bug ---------------------------------------------------------------------------

def test_a_proteome_every_alphabet_scores_zero_at_is_not_the_far_endpoint(tmp_path):
    # THE regression. Before the fix every point came back at exactly 0.0, because the far
    # endpoint was mya[-1] taken over every tool's rows and the kmerseek numerator there
    # was a real 0.0 rather than a null the join could drop.
    plot, _ = build(tmp_path)
    ys = {name: pt["y"] for name, pt in plot["data"].items()}
    assert set(ys) == {"gbmr4", "protein20"}
    assert any(y > 0 for y in ys.values()), "a flat row of zeroes is the bug"
    assert ys["gbmr4"] == pytest.approx(0.00120 / 0.0340)
    assert ys["protein20"] == pytest.approx(0.00020 / 0.0330)
    assert ys["gbmr4"] > ys["protein20"] * 5, (
        "the alphabet axis has to survive the endpoint choice")


def test_the_axis_says_which_proteome_it_actually_reached(tmp_path):
    plot, _ = build(tmp_path)
    assert plot["pconfig"]["ylab"] == "Fmax at 1,500 Mya / Fmax at 100 Mya"
    assert "arabidopsis (1,500 Mya)" in plot["description"]


def test_the_proteome_stepped_over_is_named_not_dropped_quietly(tmp_path):
    plot, table = build(tmp_path)
    for text in (plot["description"], table["description"]):
        assert "ecoli (2,000 Mya)" in text
        assert "not the far end of this plot" in text


def test_the_baselines_move_to_the_same_pair_of_proteomes(tmp_path):
    # A baseline line left at Fmax(E. coli)/Fmax(mouse) while the points are drawn at
    # arabidopsis is a comparison between two different questions.
    plot, table = build(tmp_path)
    values = sorted(round(ln["value"], 6) for ln in plot["pconfig"]["y_lines"])
    assert values == sorted([round(0.0359 / 0.0620, 6), round(0.0290 / 0.0400, 6)]), (
        "one line per method class, both on the arabidopsis pair")
    assert table["data"]["foldseek"]["far"] == pytest.approx(0.0290)
    assert table["data"]["hhblits"]["far"] == pytest.approx(0.0359)


def test_the_furthest_proteome_is_kept_when_the_alphabets_still_score_there(tmp_path):
    # The Swiss-Prot shape. Nothing may creep inward when the far end is measurable.
    ok = pl.DataFrame(
        arm("gbmr4_k17_lcFalse", 0.34, 0.20, 0.12)
        + arm("protein20_k6_lcFalse", 0.33, 0.10, 0.01)
        + [row("foldseek", "-", 100, 0.40), row("foldseek", "-", 1500, 0.30),
           row("foldseek", "-", 2000, 0.28)])
    plot, _ = build(tmp_path, ok)
    assert plot["pconfig"]["ylab"] == "Fmax at 2,000 Mya / Fmax at 100 Mya"
    assert plot["data"]["gbmr4"]["y"] == pytest.approx(0.12 / 0.34)
    assert "not the far end of this plot" not in plot["description"]


def test_one_alphabet_at_zero_keeps_its_zero(tmp_path):
    # Stepping inward is about a numerator that is zero for EVERY alphabet at once. A
    # single arm dying where its neighbours do not is the result the section exists for.
    one_dead = pl.DataFrame(
        arm("gbmr4_k17_lcFalse", 0.34, 0.20, 0.12)
        + arm("protein20_k6_lcFalse", 0.33, 0.10, 0.0))
    plot, _ = build(tmp_path, one_dead)
    assert plot["pconfig"]["ylab"] == "Fmax at 2,000 Mya / Fmax at 100 Mya"
    assert plot["data"]["protein20"]["y"] == 0.0
    assert plot["data"]["gbmr4"]["y"] == pytest.approx(0.12 / 0.34)


def test_zero_at_every_proteome_past_the_near_one_refuses_to_draw(tmp_path):
    # No inner proteome to fall back to, so there is nothing to plot. The reason given has
    # to be the numerator, not the divide-by-zero message -- the denominator is fine here.
    dead = pl.DataFrame(arm("gbmr4_k17_lcFalse", 0.34, 0.0, 0.0))
    plot, table = build(tmp_path, dead)
    assert plot["plot_type"] == "html"
    assert "detection floor" in plot["data"]
    assert "any target proteome past" in plot["data"]
    assert table is None


def test_a_zero_denominator_still_says_it_is_a_zero_denominator(tmp_path):
    # The pre-existing refusal, kept distinct from the new one so the report names the
    # right end of the ratio.
    dead = pl.DataFrame(arm("gbmr4_k17_lcFalse", 0.0, 0.0, 0.0))
    plot, _ = build(tmp_path, dead)
    assert plot["plot_type"] == "html"
    assert "not a retention of zero" in plot["data"]


# --- the labels ------------------------------------------------------------------------

def test_reference_lines_too_close_to_read_share_one_label(tmp_path):
    # The three values the broken Pfam-N section shipped with. prostt5 and reseek are a
    # percentage point apart and printed their labels over each other.
    lines = [{"value": 0.246, "label": "prostt5 (25%)"},
             {"value": 0.260, "label": "reseek (26%)"},
             {"value": 0.155, "label": "hhblits (16%)"}]
    spaced = bmi._space_line_labels(lines, 0.260 * bmi.RETENTION_LABEL_GAP)
    assert [ln["value"] for ln in spaced] == [0.155, 0.246, 0.260], (
        "the lines stay exactly where their values put them")
    labelled = [ln["label"] for ln in spaced if ln["label"]]
    assert labelled == ["hhblits (16%)", "prostt5 (25%) · reseek (26%)"]
    assert spaced[1]["label"] is None, "the crowded line is drawn unlabelled"


def test_well_separated_lines_keep_their_own_labels(tmp_path):
    lines = [{"value": 0.10, "label": "a"}, {"value": 0.50, "label": "b"},
             {"value": 0.90, "label": "c"}]
    spaced = bmi._space_line_labels(lines, 0.045)
    assert [ln["label"] for ln in spaced] == ["a", "b", "c"]


def test_the_section_does_not_stack_two_baseline_labels_on_one_spot(tmp_path):
    crowded = pl.DataFrame(
        arm("gbmr4_k17_lcFalse", 0.34, 0.20, 0.12)
        + [row("foldseek", "-", 100, 0.40), row("foldseek", "-", 2000, 0.104),
           row("prostt5", "-", 100, 0.40), row("prostt5", "-", 2000, 0.100),
           row("hhblits", "-", 100, 0.40), row("hhblits", "-", 2000, 0.020)])
    plot, _ = build(tmp_path, crowded)
    lines = plot["pconfig"]["y_lines"]
    assert len(lines) == 3, "every method class still gets its own line"
    labels = [ln["label"] for ln in lines if ln["label"]]
    assert len(labels) == 2, "the two lines 1% apart share one label"
    assert any("prostt5" in t and "foldseek" in t for t in labels)
    assert sorted(round(ln["value"], 4) for ln in lines) == [
        round(0.020 / 0.40, 4), round(0.100 / 0.40, 4), round(0.104 / 0.40, 4)]
