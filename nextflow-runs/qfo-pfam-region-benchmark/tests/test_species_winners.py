"""Per proteome: not which encoding won, but whether the win means anything.

This table used to be an argmax. For each proteome and each of ten headline metrics it
printed the three best combos with their scores, in cells reading "<alphabet> k<n> lcT
(0.137)". Every one of those strings is a winner drawn from a sweep of a couple of hundred
encodings, where the margin over the runner-up is routinely in the fourth decimal, and a
table of them reads as a result.

What replaced it reports the two numbers that say whether any of it was a result: the
margin over the next distinct encoding, and how many encodings score inside one standard
error of the best. When that count runs into the dozens the winner is not identifiable and
the section says so rather than leaving the table to imply otherwise.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def row(variant, species, mya, fmax, smin, split="selection", truth="pfam",
        reachable=10_000):
    return {"tool": "kmerseek", "variant": variant, "species": species,
            "species_mya": float(mya), "truth_set": truth, "split": split,
            "stratum_axis": "all", "stratum": "all",
            "n_reachable_instances": reachable,
            "fmax": fmax, "smin": smin, "auprc": fmax / 2}


# Two proteomes that disagree about the winner, which is the reason for a per-species
# section at all. mouse's winner is clear of the field; ecoli's is not.
METRICS = pl.DataFrame([
    row("hp_pbotc_1st_ed2_k19_lcFalse", "ecoli", 2000, 0.10, 9.0),
    row("dayhoff6_k12_lcTrue",          "ecoli", 2000, 0.30, 7.0),
    row("protein20_k5_lcFalse",         "ecoli", 2000, 0.20, 8.0),
    row("hp_pbotc_1st_ed2_k19_lcFalse", "mouse", 90,   0.90, 2.0),
    row("dayhoff6_k12_lcTrue",          "mouse", 90,   0.50, 4.0),
    row("protein20_k5_lcFalse",         "mouse", 90,   0.70, 1.0),
])


def build(tmp_path, metrics=METRICS, truth="pfam"):
    bmi.section_species_winners(tmp_path, metrics)
    out = tmp_path / f"qfo_species_winners_{truth}_mqc.json"
    return json.loads(out.read_text()) if out.exists() else None


def key_for(data, species):
    return next(k for k in data if k.startswith(species))


# --- the winner is named, once, not three deep across ten metrics --------------------

def test_each_proteome_names_its_best_encoding(tmp_path):
    data = build(tmp_path)["data"]
    assert data[key_for(data, "ecoli")]["encoding"] == "dayhoff6 k12"
    assert data[key_for(data, "mouse")]["encoding"] == "hp_pbotc_1st_ed2 k19"


def test_the_low_complexity_arm_is_not_part_of_the_encoding_name(tmp_path):
    # The filter is gone from the report: an encoding is an (alphabet, k) taken at its
    # better arm. Ranking raw combos would make the runner-up the same encoding with the
    # filter flipped, and the margin would be measuring the filter.
    assert "lcT" not in json.dumps(build(tmp_path)["data"])
    assert "lcF" not in json.dumps(build(tmp_path)["data"])


# --- the two columns the section exists for ------------------------------------------

def test_the_margin_over_the_runner_up_is_reported(tmp_path):
    data = build(tmp_path)["data"]
    assert data[key_for(data, "ecoli")]["margin"] == pytest_approx(0.30 - 0.20)
    assert data[key_for(data, "mouse")]["margin"] == pytest_approx(0.90 - 0.70)


def test_one_se_is_the_binomial_scale_at_the_winners_score(tmp_path):
    data = build(tmp_path)["data"]
    assert data[key_for(data, "ecoli")]["one_se"] == pytest_approx(
        bmi.one_se(0.30, 10_000))


def test_encodings_within_one_se_are_counted(tmp_path):
    # A tiny denominator makes the SE large, so the field closes up behind the winner and
    # it stops being identifiable. That is the state the caption has to call out. At n=20
    # the SE at 0.30 is 0.102, so 0.30 and 0.20 are inside the band and 0.10 is not.
    noisy = pl.DataFrame([{**r, "n_reachable_instances": 20}
                          for r in METRICS.to_dicts()])
    data = build(tmp_path, noisy)["data"]
    assert data[key_for(data, "ecoli")]["within_1se"] == 2


def test_a_clear_winner_stands_alone_inside_one_se(tmp_path):
    data = build(tmp_path)["data"]
    assert data[key_for(data, "mouse")]["within_1se"] == 1


def test_the_caption_says_when_the_winner_is_not_identifiable(tmp_path):
    noisy = pl.DataFrame([{**r, "n_reachable_instances": 20}
                          for r in METRICS.to_dicts()])
    text = build(tmp_path, noisy)["description"]
    assert "not identifiable" in text


def test_the_caption_does_not_claim_a_tie_when_there_is_none(tmp_path):
    text = build(tmp_path)["description"]
    assert "not identifiable" not in text


# --- the same guards the old table had, which still hold ------------------------------

def test_species_are_ordered_by_divergence_not_alphabetically(tmp_path):
    # Same axis as the Divergence section; alphabetically ecoli would come first.
    assert list(build(tmp_path)["data"])[0].startswith("mouse")


def test_the_selection_split_is_preferred_over_heldout(tmp_path):
    # This table is for CHOOSING an encoding, so it must not read the half kept for
    # reporting.
    mixed = pl.concat([
        METRICS,
        pl.DataFrame([row("heldout_only_k9_lcFalse", "ecoli", 2000, 0.99, 0.1,
                          split="heldout")]),
    ])
    assert "heldout_only" not in json.dumps(build(tmp_path, mixed)["data"])


def test_a_truth_set_with_no_split_still_gets_a_table(tmp_path):
    # Swiss-Prot and Pfam-N are scored whole, so there is no selection half to prefer.
    whole = pl.DataFrame([
        row("gbmr4_k9_lcFalse", "ecoli", 2000, 0.4, 3.0, split="all", truth="swissprot"),
        row("dayhoff6_k9_lcFalse", "ecoli", 2000, 0.6, 2.0, split="all",
            truth="swissprot"),
    ])
    bmi.section_species_winners(tmp_path, whole)
    data = json.loads(
        (tmp_path / "qfo_species_winners_swissprot_mqc.json").read_text())["data"]
    assert data[key_for(data, "ecoli")]["encoding"] == "dayhoff6 k9"


def test_a_lone_encoding_has_no_margin_rather_than_a_fabricated_one(tmp_path):
    one = pl.DataFrame([row("gbmr4_k9_lcFalse", "ecoli", 2000, 0.4, 3.0)])
    data = build(tmp_path, one)["data"]
    assert data[key_for(data, "ecoli")]["margin"] is None
    assert data[key_for(data, "ecoli")]["n_ranked"] == 1


def test_non_kmerseek_tools_are_not_ranked_here(tmp_path):
    with_baseline = pl.concat([
        METRICS,
        pl.DataFrame([{**row("single_seq", "ecoli", 2000, 0.99, 0.1),
                       "tool": "hhblits"}]),
    ])
    assert "single_seq" not in json.dumps(build(tmp_path, with_baseline)["data"])


def test_ciona_is_not_marked_on_a_family_labelled_truth_set(tmp_path):
    # Its Pfam annotation is complete and on that key it sits above fly and chicken. A mark
    # here would say the data is broken, which it is not.
    with_ciona = pl.concat([
        METRICS,
        pl.DataFrame([row("dayhoff6_k12_lcTrue", "ciona", 550, 0.2, 8.0),
                      row("protein20_k5_lcFalse", "ciona", 550, 0.1, 9.0)]),
    ])
    assert not any(bmi.CURATION_CAVEAT_MARK in k
                   for k in build(tmp_path, with_ciona)["data"])


def test_ciona_is_marked_on_the_curated_truth_set(tmp_path):
    # Swiss-Prot holds 23 curated Ciona proteins against E. coli's 2_999, so on that key a
    # species ranks partly by how much of it has been curated. The mark points at that
    # footnote; it does not say anything about the proteome.
    sprot = pl.DataFrame([
        row("dayhoff6_k12_lcTrue", "ciona", 550, 0.2, 8.0, split="all",
            truth="swissprot"),
        row("protein20_k5_lcFalse", "ciona", 550, 0.1, 9.0, split="all",
            truth="swissprot"),
        row("dayhoff6_k12_lcTrue", "mouse", 90, 0.9, 2.0, split="all", truth="swissprot"),
        row("protein20_k5_lcFalse", "mouse", 90, 0.7, 1.0, split="all",
            truth="swissprot"),
    ])
    sec = build(tmp_path, sprot, truth="swissprot")
    assert any(k.startswith("ciona" + bmi.CURATION_CAVEAT_MARK) for k in sec["data"])
    assert "23 curated proteins" in sec["description"]


def pytest_approx(value, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return other is not None and abs(other - value) < tol

        def __repr__(self):
            return f"~{value}"
    return _Approx()
