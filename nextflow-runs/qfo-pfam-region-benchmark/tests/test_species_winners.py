"""The per-species winners table: which kmerseek combo won each metric, per proteome.

The leaderboard ranks on a mean over the nine target species and the alphabet heatmap
averages too, so neither answers "which alphabet and ksize was best against zebrafish".
This section does, three deep per cell, because across ~400 combos a strict argmax
routinely wins in the fourth decimal.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def row(variant, species, mya, fmax, smin, split="selection", truth="pfam"):
    return {"tool": "kmerseek", "variant": variant, "species": species,
            "species_mya": mya, "truth_set": truth, "split": split,
            "stratum_axis": "all", "stratum": "all",
            "fmax": fmax, "smin": smin, "auprc": fmax / 2}


# Two species that disagree, which is the whole reason for the section.
METRICS = pl.DataFrame([
    row("hp_pbotc_1st_ed2_k19_lcFalse", "ecoli", 2000, 0.10, 9.0),
    row("dayhoff6_k12_lcTrue",          "ecoli", 2000, 0.30, 7.0),
    row("protein20_k5_lcFalse",         "ecoli", 2000, 0.20, 8.0),
    row("hp_pbotc_1st_ed2_k19_lcFalse", "mouse", 90,   0.90, 2.0),
    row("dayhoff6_k12_lcTrue",          "mouse", 90,   0.50, 4.0),
    row("protein20_k5_lcFalse",         "mouse", 90,   0.70, 1.0),
])


def build(tmp_path, metrics=METRICS):
    bmi.section_species_winners(tmp_path, metrics)
    out = tmp_path / "qfo_species_winners_pfam_mqc.json"
    return json.loads(out.read_text())["data"] if out.exists() else None


def test_each_species_gets_its_own_winner(tmp_path):
    data = build(tmp_path)
    assert data["ecoli · 1"]["fmax"].startswith("dayhoff6 k12 lcT")
    assert data["mouse · 1"]["fmax"].startswith("hp_pbotc_1st_ed2 k19 lcF")


def test_three_ranks_per_species_in_order(tmp_path):
    data = build(tmp_path)
    assert [k for k in data if k.startswith("ecoli")] == ["ecoli · 1", "ecoli · 2", "ecoli · 3"]
    assert data["ecoli · 2"]["fmax"].startswith("protein20 k5 lcF")
    assert data["ecoli · 3"]["fmax"].startswith("hp_pbotc_1st_ed2 k19 lcF")


def test_smin_is_ranked_the_other_way_round(tmp_path):
    # Semantic distance: the best combo is the SMALLEST. Ranking it like the rest would
    # crown the worst one and nothing about the table would look wrong.
    data = build(tmp_path)
    assert data["ecoli · 1"]["smin"].startswith("dayhoff6 k12 lcT"), "smin 7.0 is best"
    assert data["mouse · 1"]["smin"].startswith("protein20 k5 lcF"), "smin 1.0 is best"


def test_the_value_is_shown_beside_the_combo(tmp_path):
    assert "(0.300)" in build(tmp_path)["ecoli · 1"]["fmax"]


def test_species_are_ordered_by_divergence_not_alphabetically(tmp_path):
    # Same axis as the Divergence section; alphabetically ecoli would come first.
    assert list(build(tmp_path))[0].startswith("mouse")


def test_the_selection_split_is_preferred_over_heldout(tmp_path):
    # This table is for CHOOSING a combo, so it must not read the half kept for reporting.
    mixed = pl.concat([
        METRICS,
        pl.DataFrame([row("heldout_only_k9_lcFalse", "ecoli", 2000, 0.99, 0.1,
                          split="heldout")]),
    ])
    data = build(tmp_path, mixed)
    assert "heldout_only" not in json.dumps(data)


def test_a_truth_set_with_no_split_still_gets_a_table(tmp_path):
    # Swiss-Prot and Pfam-N are scored whole, so there is no selection half to prefer.
    whole = pl.DataFrame([
        row("a_k9_lcFalse", "ecoli", 2000, 0.4, 3.0, split="all", truth="swissprot"),
        row("b_k9_lcFalse", "ecoli", 2000, 0.6, 2.0, split="all", truth="swissprot"),
    ])
    bmi.section_species_winners(tmp_path, whole)
    data = json.loads((tmp_path / "qfo_species_winners_swissprot_mqc.json").read_text())
    assert data["data"]["ecoli · 1"]["fmax"].startswith("b k9 lcF")


def test_fewer_combos_than_ranks_leaves_the_extra_cells_blank(tmp_path):
    one = pl.DataFrame([row("only_k9_lcFalse", "ecoli", 2000, 0.4, 3.0)])
    data = build(tmp_path, one)
    assert data["ecoli · 1"]["fmax"].startswith("only k9 lcF")
    assert data["ecoli · 2"]["fmax"] == "" and data["ecoli · 3"]["fmax"] == ""


def test_non_kmerseek_tools_are_not_ranked_here(tmp_path):
    with_baseline = pl.concat([
        METRICS,
        pl.DataFrame([{**row("single_seq", "ecoli", 2000, 0.99, 0.1), "tool": "hhblits"}]),
    ])
    assert "single_seq" not in json.dumps(build(tmp_path, with_baseline))
