"""The three claims the report's framing rests on, each tested against the shape of data
that would make it false.

1. Model-confidence regime. A grouped bar chart per pLDDT band cannot show whether a tool
   RISES with structure confidence or peaks in the middle, and those are different claims:
   one is "kmerseek degrades", the other is "kmerseek occupies a regime". The line form
   only helps if the x axis is real pLDDT, the peak is computed rather than asserted, and
   a run without three bands refuses to draw a shape.

2. Divergence retention. Retention is far/near, which flatters anything that starts low, so
   the absolute Fmax at both ends must travel with it. The k has to be chosen once per
   alphabet and on the selection half, because choosing it separately at each endpoint
   manufactures retention out of two different encodings.

3. Truth-set circularity. The report used to assert that a tool's Pfam-to-Swiss-Prot drop
   IS the size of Pfam's circularity with the profile methods. That only holds if the drop
   is specific to those methods. When every class drops by the same factor the ratio
   measures how much harder Swiss-Prot is, and the caption has to say so instead.

Plus the pinning mechanism, which exists because which kmerseek arm a figure draws depends
on that figure's own ranking metric and truth set.
"""
import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


@pytest.fixture(autouse=True)
def no_pin():
    """Every test starts unpinned. CANONICAL is module state rebound from the CLI."""
    bmi.CANONICAL = None
    yield
    bmi.CANONICAL = None


def band_row(tool, variant, stratum, fmax, n_proteins=500, n_instances=4_000):
    return {"truth_set": "swissprot", "tool": tool, "variant": variant, "species": "mouse",
            "split": "all", "stratum_axis": "plddt", "stratum": stratum, "fmax": fmax,
            "n_stratum_proteins": n_proteins, "n_truth_instances": n_instances}


def all_row(tool, variant, fmax):
    return {"truth_set": "swissprot", "tool": tool, "variant": variant, "species": "mouse",
            "split": "all", "stratum_axis": "all", "stratum": "all", "fmax": fmax,
            "n_stratum_proteins": 900, "n_truth_instances": 9_000}


# The shape the midi run has: the coarse-alphabet arm peaks in the middle band while every
# baseline climbs to the top one. Numbers are illustrative, the ORDERING is the fixture.
BANDS = {
    ("kmerseek", "polarity4_k16_lcFalse"): {"50-70": 0.08, "70-90": 0.17, "90-100": 0.15},
    ("foldseek", "-"): {"50-70": 0.10, "70-90": 0.13, "90-100": 0.28},
    ("hhblits", "-"): {"50-70": 0.16, "70-90": 0.17, "90-100": 0.25},
}


def plddt_metrics(bands=BANDS) -> pl.DataFrame:
    rows = []
    for (tool, variant), by_band in bands.items():
        rows.append(all_row(tool, variant, max(by_band.values())))
        for stratum, fmax in by_band.items():
            rows.append(band_row(tool, variant, stratum, fmax))
    return pl.DataFrame(rows)


def regime(tmp_path, bands=BANDS):
    bmi.section_plddt_regime(tmp_path, plddt_metrics(bands), "swissprot", 10)
    plot = tmp_path / "qfo_plddt_regime_mqc.json"
    table = tmp_path / "qfo_plddt_regime_table_mqc.json"
    return (json.loads(plot.read_text()) if plot.exists() else None,
            json.loads(table.read_text()) if table.exists() else None)


# --- 1. the model-confidence regime --------------------------------------------------

def test_band_midpoint_is_the_middle_of_the_band():
    assert bmi.band_midpoint("50-70") == 60
    assert bmi.band_midpoint("90-100") == 95
    assert bmi.band_midpoint("no_homolog") is None


def test_x_is_plddt_not_three_equal_slots(tmp_path):
    # Equal spacing would put 90-100 as far from 70-90 as 70-90 is from 50-70, which is
    # twice the pLDDT it spans -- and the crossover happens between exactly those two.
    plot, _ = regime(tmp_path)
    for series in plot["data"].values():
        assert sorted(float(x) for x in series) == [60.0, 80.0, 95.0]


def test_it_is_drawn_as_lines_because_the_shape_is_the_claim(tmp_path):
    plot, _ = regime(tmp_path)
    assert plot["plot_type"] == "linegraph"


def test_a_non_monotone_arm_is_named_in_the_caption(tmp_path):
    plot, _ = regime(tmp_path)
    assert "peak below the top band" in plot["description"]
    assert "polarity4_k16_lcFalse" in plot["description"]
    assert "foldseek" not in plot["description"].split("peak below the top band")[1][:400]


def test_when_every_arm_climbs_the_caption_says_there_is_no_regime(tmp_path):
    monotone = {k: {"50-70": 0.1, "70-90": 0.2, "90-100": 0.3} for k in BANDS}
    plot, _ = regime(tmp_path, monotone)
    assert "every arm peaks in the top band" in plot["description"]
    assert "no regime to claim" in plot["description"]


def test_the_peak_column_matches_the_lines(tmp_path):
    plot, table = regime(tmp_path)
    assert table["data"]["kmerseek polarity4_k16_lcFalse"]["peak"] == "70-90"
    assert table["data"]["foldseek"]["peak"] == "90-100"


def test_two_bands_refuse_to_draw_a_peak(tmp_path):
    # Two points are a slope. Drawing them here would be read as the shape the section is
    # named for, which is the failure the covariate bar chart had on the omega axis.
    two = {k: {"70-90": 0.2, "90-100": 0.3} for k in BANDS}
    plot, table = regime(tmp_path, two)
    assert plot["plot_type"] == "html"
    assert "peak needs at least 3" in plot["data"]
    assert table is None


def test_the_missing_low_band_is_stated_not_implied(tmp_path):
    # STRATA defines a 0-50 band. When it is empty, a fall-off at the left edge is a
    # fall-off in low-confidence structure, not in the disordered tail.
    plot, _ = regime(tmp_path)
    assert "The lowest band drawn is 50-70" in plot["description"]


def test_the_table_carries_the_denominator_and_says_there_is_no_error_bar(tmp_path):
    _, table = regime(tmp_path)
    assert "500 proteins" in table["description"]
    assert "sampling error" in table["description"]


# --- 2. divergence retention ---------------------------------------------------------

def div_row(tool, variant, mya, fmax, split="all", truth="pfam"):
    return {"truth_set": truth, "tool": tool, "variant": variant,
            "species": {100.0: "mouse", 2000.0: "ecoli"}.get(mya, f"sp{mya:.0f}"),
            "species_mya": float(mya), "split": split,
            "stratum_axis": "all", "stratum": "all", "fmax": fmax}


# polarity4 keeps 40% of its mouse score out to E. coli; protein20 keeps under 1%. Same
# engine, same index, same query set -- the alphabet is the only thing that differs.
RETAIN = pl.DataFrame([
    div_row("kmerseek", "polarity4_k16_lcFalse", 100, 0.28),
    div_row("kmerseek", "polarity4_k16_lcFalse", 2000, 0.112),
    div_row("kmerseek", "protein20_k11_lcFalse", 100, 0.29),
    div_row("kmerseek", "protein20_k11_lcFalse", 2000, 0.002),
    div_row("foldseek", "-", 100, 0.147),
    div_row("foldseek", "-", 2000, 0.137),
    div_row("hmmer3_phmmer", "-", 100, 0.184),
    div_row("hmmer3_phmmer", "-", 2000, 0.056),
])


def retention(tmp_path, metrics=RETAIN, truth="pfam"):
    bmi.section_alphabet_retention(tmp_path, metrics)
    plot = tmp_path / f"qfo_retention_{truth}_mqc.json"
    table = tmp_path / f"qfo_retention_table_{truth}_mqc.json"
    return (json.loads(plot.read_text()) if plot.exists() else None,
            json.loads(table.read_text()) if table.exists() else None)


def test_retention_is_far_over_near(tmp_path):
    plot, _ = retention(tmp_path)
    assert plot["data"]["polarity4"]["y"] == pytest.approx(0.112 / 0.28)
    assert plot["data"]["protein20"]["y"] == pytest.approx(0.002 / 0.29)


def test_x_is_the_alphabet_class_count_from_the_name(tmp_path):
    plot, _ = retention(tmp_path)
    assert plot["data"]["polarity4"]["x"] == 4
    assert plot["data"]["protein20"]["x"] == 20


def test_the_absolute_fmax_at_both_ends_travels_with_the_ratio(tmp_path):
    # A method that starts low and stays low retains 100%, so the ratio alone is a trap.
    _, table = retention(tmp_path)
    row = table["data"]["kmerseek polarity4 k16"]
    assert row["near"] == pytest.approx(0.28) and row["far"] == pytest.approx(0.112)
    assert "near" in table["headers"] and "far" in table["headers"]


def test_a_tool_that_starts_low_and_stays_low_is_not_hidden(tmp_path):
    flat = pl.concat([RETAIN, pl.DataFrame([
        div_row("reseek", "-", 100, 0.01), div_row("reseek", "-", 2000, 0.01)])])
    _, table = retention(tmp_path, flat)
    assert table["data"]["reseek"]["retention"] == pytest.approx(1.0)
    assert table["data"]["reseek"]["near"] == pytest.approx(0.01), (
        "100% retention of nothing has to be visible as nothing")


def test_baselines_are_reference_lines_and_table_rows(tmp_path):
    plot, table = retention(tmp_path)
    values = {round(ln["value"], 4) for ln in plot["pconfig"]["y_lines"]}
    assert round(0.137 / 0.147, 4) in values, "the best structure retainer is a line"
    assert "foldseek" in table["data"] and "hmmer3_phmmer" in table["data"]
    assert table["data"]["foldseek"]["classes"] is None, "a baseline has no alphabet"


def test_the_hp_family_is_marked_as_its_own_group(tmp_path):
    hp = pl.concat([RETAIN, pl.DataFrame([
        div_row("kmerseek", "hp_thomas_dill2_k25_lcFalse", 100, 0.20),
        div_row("kmerseek", "hp_thomas_dill2_k25_lcFalse", 2000, 0.10)])])
    plot, _ = retention(tmp_path, hp)
    assert plot["data"]["hp_thomas_dill2"]["group"] == "HP (hydrophobic / polar)"
    assert plot["data"]["protein20"]["group"] == "unreduced (20 classes)"
    assert plot["data"]["polarity4"]["group"] == "other reduced alphabet"


def test_one_k_per_alphabet_not_the_best_k_at_each_end(tmp_path):
    # k5 wins the near end and k30 wins the far end. Taking each endpoint's own best would
    # report 0.25 / 0.25 = 100% retention for an encoding that does not exist.
    two_ks = pl.DataFrame([
        div_row("kmerseek", "gbmr4_k5_lcFalse", 100, 0.25),
        div_row("kmerseek", "gbmr4_k5_lcFalse", 2000, 0.01),
        div_row("kmerseek", "gbmr4_k30_lcFalse", 100, 0.05),
        div_row("kmerseek", "gbmr4_k30_lcFalse", 2000, 0.25),
    ])
    plot, table = retention(tmp_path, two_ks)
    assert plot["data"]["gbmr4"]["y"] != pytest.approx(1.0), "0.25/0.25 is the fabrication"
    assert plot["data"]["gbmr4"]["y"] in (
        pytest.approx(0.01 / 0.25), pytest.approx(0.25 / 0.05))
    row = next(iter(table["data"].values()))
    assert (row["near"], row["far"]) in {(0.25, 0.01), (0.05, 0.25)}, (
        "both ends must come from the SAME encoding")


def test_the_k_is_chosen_on_the_selection_half(tmp_path):
    # Choosing k on the half the retention is reported on is what the split exists to
    # prevent. The selection half prefers k5; the heldout half would have preferred k30.
    split_rows = pl.DataFrame([
        div_row("kmerseek", "gbmr4_k5_lcFalse", 100, 0.40, split="selection"),
        div_row("kmerseek", "gbmr4_k5_lcFalse", 2000, 0.30, split="selection"),
        div_row("kmerseek", "gbmr4_k30_lcFalse", 100, 0.10, split="selection"),
        div_row("kmerseek", "gbmr4_k30_lcFalse", 2000, 0.05, split="selection"),
        div_row("kmerseek", "gbmr4_k5_lcFalse", 100, 0.20, split="heldout"),
        div_row("kmerseek", "gbmr4_k5_lcFalse", 2000, 0.02, split="heldout"),
        div_row("kmerseek", "gbmr4_k30_lcFalse", 100, 0.90, split="heldout"),
        div_row("kmerseek", "gbmr4_k30_lcFalse", 2000, 0.90, split="heldout"),
    ])
    _, table = retention(tmp_path, split_rows)
    assert list(table["data"]) == ["kmerseek gbmr4 k5"], "k30 was chosen on the wrong half"
    assert table["data"]["kmerseek gbmr4 k5"]["near"] == pytest.approx(0.20), (
        "and the number reported is the heldout one")


def test_an_arm_scoring_zero_at_the_near_end_says_so_rather_than_dividing(tmp_path):
    zeros = pl.DataFrame([
        div_row("kmerseek", "gbmr4_k5_lcFalse", 100, 0.0),
        div_row("kmerseek", "gbmr4_k5_lcFalse", 2000, 0.0),
    ])
    plot, table = retention(tmp_path, zeros)
    assert plot["plot_type"] == "html"
    assert "not a retention of zero" in plot["data"]
    assert table is None


def test_truth_sets_are_never_pooled(tmp_path):
    mixed = pl.concat([RETAIN, pl.DataFrame([
        div_row("kmerseek", "polarity4_k16_lcFalse", 100, 0.9, truth="swissprot"),
        div_row("kmerseek", "polarity4_k16_lcFalse", 2000, 0.9, truth="swissprot")])])
    bmi.section_alphabet_retention(tmp_path, mixed)
    pfam, _ = retention(tmp_path, mixed, truth="pfam")
    sprot, _ = retention(tmp_path, mixed, truth="swissprot")
    assert pfam["data"]["polarity4"]["y"] == pytest.approx(0.112 / 0.28)
    assert sprot["data"]["polarity4"]["y"] == pytest.approx(1.0)


def test_the_ceiling_species_is_not_an_endpoint(tmp_path):
    with_all = pl.concat([RETAIN, pl.DataFrame([
        {**div_row("hmmscan", "-", 100, 0.96), "species": "all", "species_mya": None}])],
        how="diagonal_relaxed")
    _, table = retention(tmp_path, with_all)
    assert "hmmscan" not in table["data"]


# --- 3. the truth-set caption --------------------------------------------------------

# Every class drops by roughly the same factor, which is what the midi run does. The ratio
# is then a statement about Swiss-Prot being harder, not about who wrote Pfam.
FLAT_DROP = {
    "kmerseek dayhoff6_k8_lcTrue": {"pfam": 0.461, "swissprot": 0.238, "pfamn": 0.027},
    "hmmer3_phmmer": {"pfam": 0.349, "swissprot": 0.193, "pfamn": 0.065},
    "hhblits": {"pfam": 0.327, "swissprot": 0.220, "pfamn": 0.063},
    "foldseek": {"pfam": 0.265, "swissprot": 0.151, "pfamn": 0.050},
}
FLAT_TOOLS = {"kmerseek dayhoff6_k8_lcTrue": "kmerseek", "hmmer3_phmmer": "hmmer3_phmmer",
              "hhblits": "hhblits", "foldseek": "foldseek"}


def test_a_drop_every_class_shares_is_not_circularity():
    text = bmi.circularity_bullet(FLAT_DROP, FLAT_TOOLS)
    assert "does not single out the methods Pfam was built from" in text
    assert "does not detect circularity" in text


def test_a_class_with_one_tool_in_it_does_not_read_as_separation():
    # A single-tool class is a point, not a range. A symmetric overlap test calls that
    # separation on its own, which is why the test here is directional instead.
    one_each = {"kmerseek dayhoff6_k8_lcTrue": {"pfam": 0.461, "swissprot": 0.238},
                "hmmer3_phmmer": {"pfam": 0.349, "swissprot": 0.193},
                "foldseek": {"pfam": 0.265, "swissprot": 0.151}}
    text = bmi.circularity_bullet(one_each, FLAT_TOOLS)
    assert "does not single out" in text, text


def test_a_drop_specific_to_the_profile_methods_is_called_what_it_is():
    steep = {**FLAT_DROP,
             "hmmer3_phmmer": {"pfam": 0.90, "swissprot": 0.10, "pfamn": 0.065},
             "hhblits": {"pfam": 0.88, "swissprot": 0.11, "pfamn": 0.063}}
    text = bmi.circularity_bullet(steep, FLAT_TOOLS)
    assert "does single out the methods Pfam was built from" in text
    assert "sequence alignment" in text


def test_the_ratios_are_reported_per_class_not_asserted():
    text = bmi.circularity_bullet(FLAT_DROP, FLAT_TOOLS)
    # 0.349 / 0.193 = 1.81 for phmmer, 0.327 / 0.220 = 1.49 for hhblits.
    assert "1.49-1.81" in text, text


def test_the_pfam_lead_is_measured_against_the_class_pfam_is_circular_with():
    text = bmi.pfam_lead_bullet(FLAT_DROP, FLAT_TOOLS)
    assert "beats every" in text and "sequence alignment" in text
    assert "0.461" in text and "0.349" in text


def test_a_higher_bar_from_another_class_is_not_swallowed():
    with_folddisco = {**FLAT_DROP,
                      "folddisco": {"pfam": 0.514, "swissprot": 0.439, "pfamn": 0.057}}
    tools = {**FLAT_TOOLS, "folddisco": "folddisco"}
    text = bmi.pfam_lead_bullet(with_folddisco, tools)
    assert "0.514" in text and "higher still" in text


def test_pfam_n_is_stated_rather_than_left_to_be_noticed():
    text = bmi.pfamn_bullet(FLAT_DROP, FLAT_TOOLS)
    assert "reverses the Pfam ordering" in text
    assert "ranks 3 of 3" in text, text
    assert "sequence alignment 0.065" in text, "the class it loses to is named"


def test_the_section_caption_no_longer_calls_the_gap_a_measure_of_circularity(tmp_path):
    rows = []
    for label, cells in FLAT_DROP.items():
        tool = FLAT_TOOLS[label]
        variant = label.split(" ", 1)[1] if tool == "kmerseek" else "-"
        for ts, fmax in cells.items():
            rows.append({"truth_set": ts, "tool": tool, "variant": variant,
                         "species": "mouse", "split": "all", "stratum_axis": "all",
                         "stratum": "all", "fmax": fmax})
    bmi.section_truthsets(tmp_path, pl.DataFrame(rows), 20)
    sec = json.loads((tmp_path / "qfo_truthsets_mqc.json").read_text())
    assert "is the size of that circularity" not in sec["description"]
    assert "cannot do on its own is measure circularity" in sec["description"]


# --- 4. pinning one arm across the report --------------------------------------------

def test_a_bare_variant_means_kmerseek():
    assert bmi.parse_canonical("hp_pbotc_1st_ed2_k25_lcFalse") == (
        "kmerseek", "hp_pbotc_1st_ed2_k25_lcFalse")
    assert bmi.parse_canonical("foldseek:-") == ("foldseek", "-")
    assert bmi.parse_canonical(None) is None


def test_nothing_is_marked_when_nothing_is_pinned():
    assert bmi.label_of("kmerseek", "polarity4_k16_lcFalse") == (
        "kmerseek polarity4_k16_lcFalse")


def test_the_pinned_arm_is_marked_wherever_it_is_drawn():
    bmi.CANONICAL = ("kmerseek", "polarity4_k16_lcFalse")
    assert bmi.label_of("kmerseek", "polarity4_k16_lcFalse").endswith(bmi.CANONICAL_MARK)
    assert not bmi.label_of("kmerseek", "protein20_k11_lcFalse").endswith(
        bmi.CANONICAL_MARK)


def test_the_pinned_arm_rides_into_a_board_it_did_not_rank_into():
    rows = []
    for i in range(8):
        rows.append(all_row("kmerseek", f"a{i}_k5_lcFalse", 0.9 - i / 100))
    rows.append(all_row("kmerseek", "hp_thomas_dill2_k25_lcFalse", 0.01))
    df = pl.DataFrame(rows)
    unpinned = bmi.best_variants(df, top_kmerseek=3)["variant"].to_list()
    assert "hp_thomas_dill2_k25_lcFalse" not in unpinned
    bmi.CANONICAL = ("kmerseek", "hp_thomas_dill2_k25_lcFalse")
    pinned = bmi.best_variants(df, top_kmerseek=3)["variant"].to_list()
    assert "hp_thomas_dill2_k25_lcFalse" in pinned
    assert len(pinned) == len(unpinned) + 1, "pinning adds a row, it does not reorder"


def test_no_pinned_section_when_nothing_is_pinned(tmp_path):
    bmi.section_canonical(tmp_path, plddt_metrics())
    assert not (tmp_path / "qfo_canonical_mqc.json").exists()


def test_a_pinned_arm_that_does_not_exist_is_reported_loudly(tmp_path):
    bmi.CANONICAL = ("kmerseek", "not_a_real_alphabet_k9_lcFalse")
    bmi.section_canonical(tmp_path, plddt_metrics())
    sec = json.loads((tmp_path / "qfo_canonical_mqc.json").read_text())
    assert "Pinned arm not found" in sec["data"]


def test_the_pinned_section_names_the_arm_and_its_score(tmp_path):
    bmi.CANONICAL = ("kmerseek", "polarity4_k16_lcFalse")
    bmi.section_canonical(tmp_path, plddt_metrics())
    sec = json.loads((tmp_path / "qfo_canonical_mqc.json").read_text())
    assert "polarity4_k16_lcFalse" in sec["data"]
    assert "swissprot" in sec["data"]
    assert "Off by default" in sec["description"]
    assert sec["data"].count(bmi.CANONICAL_MARK.strip()) == 1, (
        "the row key already carries the mark; printing both reads as '... ★ ★'")
