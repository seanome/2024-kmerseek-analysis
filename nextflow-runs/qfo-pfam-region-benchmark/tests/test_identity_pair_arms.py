"""A panel forced onto a fallback truth set must still draw the arms the report is about.

section_identity_vs_divergence falls back to Pfam when the primary truth set has no
identity axis, and then runs best_variants over Pfam rows. So the arms it drew were the
ones Pfam ranked, under a caption about the claim the primary truth set's leaderboard is
stated on. On the midi-plus run the two selections have almost no arms in common: Pfam
ranks near-exact matchers at 38-48 bits per k-mer, and the arms that hold their level out
to E. coli on the primary set sit at 23-34 bits and were not on the panel at all. Reading
"kmerseek is zero below 60% identity" off that figure reads it off a different tool.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402

MARK = bmi.PRIMARY_PICK_MARK

# Two arms that swap places between the truth sets, which is the situation the carry-over
# exists for: `sharp` wins on pfam and `broad` wins on swissprot.
ARMS = [("kmerseek", "protein20_k10_lcFalse"), ("kmerseek", "polarity4_k17_lcFalse"),
        ("hmmer3_phmmer", "-")]
FMAX = {
    ("pfam", "protein20_k10_lcFalse"): 0.40, ("swissprot", "protein20_k10_lcFalse"): 0.05,
    ("pfam", "polarity4_k17_lcFalse"): 0.06, ("swissprot", "polarity4_k17_lcFalse"): 0.30,
    ("pfam", "-"): 0.20, ("swissprot", "-"): 0.20,
}


def row(truth, tool, variant, species, mya, axis, stratum, fmax):
    return {"truth_set": truth, "tool": tool, "variant": variant, "species": species,
            "species_mya": mya, "split": "all", "stratum_axis": axis, "stratum": stratum,
            "fmax": fmax, "n_truth_instances": 500}


def metrics(identity_on=("pfam",)) -> pl.DataFrame:
    rows = []
    for truth in ("pfam", "swissprot"):
        for tool, variant in ARMS:
            f = FMAX[(truth, variant if tool == "kmerseek" else "-")]
            for species, mya in (("mouse", 100.0), ("ecoli", 2000.0)):
                rows.append(row(truth, tool, variant, species, mya, "all", "all", f))
            if truth in identity_on:
                for b, scale in (("20-30%", 0.2), ("30-40%", 0.5), ("60-100%", 1.0)):
                    rows.append(row(truth, tool, variant, "mouse", 100.0,
                                    "identity", b, f * scale))
    return pl.DataFrame(rows)


def panels(tmp_path, m, primary="swissprot", max_tools=2) -> dict:
    bmi.section_identity_vs_divergence(tmp_path, m, primary, max_tools)
    return {p.stem: json.loads(p.read_text())
            for p in tmp_path.glob("qfo_identity_pair_*_mqc.json")}


def series(tmp_path, **kw) -> set:
    p = panels(tmp_path, metrics(), **kw)
    return set(p["qfo_identity_pair_pident_mqc"]["data"])


def test_the_forced_panel_carries_the_primary_truth_sets_pick(tmp_path):
    # polarity4 is nowhere near the top on pfam, which is the only truth set with an
    # identity axis here, so nothing but the carry-over puts it on this figure.
    assert any(s.startswith("kmerseek polarity4_k17") for s in series(tmp_path))


def test_a_carried_arm_is_marked_so_the_legend_says_where_it_came_from(tmp_path):
    carried = [s for s in series(tmp_path) if s.endswith(MARK)]
    assert carried == [f"kmerseek polarity4_k17_lcFalse{MARK}"]


def test_the_caption_names_the_carried_arms_and_why(tmp_path):
    desc = panels(tmp_path, metrics())["qfo_identity_pair_pident_mqc"]["description"]
    assert "kmerseek polarity4_k17_lcFalse" in desc
    assert "only the selection is carried over" in desc


def test_the_pfam_pick_is_still_drawn_beside_it(tmp_path):
    assert any(s.startswith("kmerseek protein20_k10") for s in series(tmp_path))


def test_nothing_is_carried_when_the_primary_set_has_its_own_identity_axis(tmp_path):
    # The pipeline-side anchor fix is what puts an identity axis on swissprot. Once it has
    # one there is no fallback, no second selection, and no mark.
    p = panels(tmp_path, metrics(identity_on=("pfam", "swissprot")))
    labels = set(p["qfo_identity_pair_pident_mqc"]["data"])
    assert not [s for s in labels if s.endswith(MARK)]
    assert any(s.startswith("kmerseek polarity4_k17") for s in labels)


def test_an_arm_the_fallback_set_never_scored_is_not_carried_as_an_empty_line(tmp_path):
    m = metrics().filter(~((pl.col("truth_set") == "pfam")
                           & (pl.col("variant") == "polarity4_k17_lcFalse")))
    labels = set(panels(tmp_path, m)["qfo_identity_pair_pident_mqc"]["data"])
    assert not [s for s in labels if s.endswith(MARK)]


def test_the_skew_bullet_no_longer_answers_the_question_for_the_reader(tmp_path):
    # It used to close with "on this run it points at the former", a per-report verdict
    # read off whichever arms that selection happened to draw.
    desc = panels(tmp_path, metrics())["qfo_identity_pair_pident_mqc"]["description"]
    assert "points at the former" not in desc
    assert "read it off the line you mean to make the claim about" in desc
