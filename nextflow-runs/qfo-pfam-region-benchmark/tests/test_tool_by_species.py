"""Every tool against every target proteome, as points rather than as a mean.

The Divergence section covers three metrics and whichever variants topped the leaderboard,
which on a full sweep is mostly kmerseek combos. This section is one COLUMN per tool, with
one point per target proteome in it, so "does phmmer hold up further out than foldseek" is
readable directly and so is the thing a mean hides: an arm whose best proteome beats every
baseline's best while its average sits below them.

A line per tool was the wrong mark twice over. It drew the mean as the shape, and it
connected proteomes, which do not interpolate: there is no target between mouse and E. coli
for a point on that segment to be.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def row(tool, variant, species, mya, fmax, smin=5.0):
    return {"tool": tool, "variant": variant, "species": species, "species_mya": mya,
            "truth_set": "pfam", "split": "heldout", "stratum_axis": "all",
            "stratum": "all", "fmax": fmax, "smin": smin, "auprc": fmax / 2,
            "recall_reachable": fmax}


# phmmer holds its level out to E. coli; foldseek falls away. That contrast is the
# question this section exists to answer.
METRICS = pl.DataFrame([
    row("hmmer3_phmmer", "default", "mouse", 90, 0.60),
    row("hmmer3_phmmer", "default", "ecoli", 2000, 0.55),
    row("foldseek", "3di_aa", "mouse", 90, 0.70),
    row("foldseek", "3di_aa", "ecoli", 2000, 0.10),
    row("kmerseek", "hp_pbotc_1st_ed2_k19_lcFalse", "mouse", 90, 0.40),
    row("kmerseek", "hp_pbotc_1st_ed2_k19_lcFalse", "ecoli", 2000, 0.38),
    # a second kmerseek combo, which must NOT get its own line
    row("kmerseek", "dayhoff6_k12_lcTrue", "mouse", 90, 0.30),
    row("kmerseek", "dayhoff6_k12_lcTrue", "ecoli", 2000, 0.05),
])


def build(tmp_path, metrics=METRICS, name="qfo_tool_by_species_pfam"):
    bmi.section_tool_by_species(tmp_path, metrics)
    f = tmp_path / f"{name}_mqc.json"
    return json.loads(f.read_text()) if f.exists() else None


def tools_on_x(sec):
    return sec["pconfig"]["categories"]


def point(sec, panel, tool, species):
    return next(v for k, v in sec["data"][panel].items()
                if k.startswith(tool) and f" · {species} · " in k)


def test_every_tool_gets_a_column(tmp_path):
    sec = build(tmp_path)
    assert {label.split()[0] for label in tools_on_x(sec)} == {
        "hmmer3_phmmer", "foldseek", "kmerseek"}


def test_only_one_column_per_tool(tmp_path):
    # Not the leaderboard's top-N kmerseek combos: those would crowd out the baselines.
    sec = build(tmp_path)
    assert len([k for k in tools_on_x(sec) if k.startswith("kmerseek")]) == 1


def test_the_plot_is_points_not_a_line(tmp_path):
    # A line between two proteomes asserts a target that does not exist between them.
    assert build(tmp_path)["plot_type"] == "scatter"


def test_one_point_per_proteome_and_no_aggregation_in_it(tmp_path):
    sec = build(tmp_path)
    assert point(sec, 0, "foldseek", "mouse")["y"] == 0.70
    assert point(sec, 0, "foldseek", "ecoli")["y"] == 0.10


def test_the_mean_is_one_mark_inside_the_column_not_the_column(tmp_path):
    sec = build(tmp_path)
    mean = next(v for k, v in sec["data"][0].items()
                if k.startswith("foldseek") and " · mean · " in k)
    assert mean["y"] == pytest_approx((0.70 + 0.10) / 2)
    assert mean["marker_symbol"] == "line-ew"


def test_every_point_in_a_tool_shares_that_tools_x(tmp_path):
    sec = build(tmp_path)
    xs = {v["x"] for k, v in sec["data"][0].items() if k.startswith("foldseek")}
    assert len(xs) == 1


def test_the_shape_that_answers_the_question_survives(tmp_path):
    # foldseek beats phmmer on mouse and loses badly on ecoli. If the section flattened
    # species away, this would be unreadable.
    sec = build(tmp_path)
    assert point(sec, 0, "foldseek", "mouse")["y"] > point(sec, 0, "hmmer3_phmmer",
                                                           "mouse")["y"]
    assert point(sec, 0, "foldseek", "ecoli")["y"] < point(sec, 0, "hmmer3_phmmer",
                                                           "ecoli")["y"]


def test_no_proteome_is_drawn_as_though_its_data_were_broken(tmp_path):
    """ciona is not an annotation failure.

    Its Pfam annotation is complete -- 20_234 domain rows over 10_658 proteins in 5_542
    families -- and on the Pfam truth set it reaches 82.9% of the answer key, above fly and
    chicken. What understates it is the SWISS-PROT key, which is manually curated and holds
    23 Ciona proteins against E. coli's 2_999. That is a property of the answer key, so it
    belongs in a footnote on the Swiss-Prot panels and nowhere else. An open marker, a
    hatched bar or a mean drawn a second time without it would all say the data is broken.
    """
    with_ciona = pl.concat([
        METRICS,
        pl.DataFrame([row("hmmer3_phmmer", "default", "ciona", 550, 0.02),
                      row("foldseek", "3di_aa", "ciona", 550, 0.02),
                      row("kmerseek", "hp_pbotc_1st_ed2_k19_lcFalse", "ciona", 550,
                          0.02),
                      row("kmerseek", "dayhoff6_k12_lcTrue", "ciona", 550, 0.02)]),
    ])
    sec = build(tmp_path, with_ciona)
    assert point(sec, 0, "foldseek", "ciona")["marker_symbol"] == "circle"
    assert not any("mean-nodiag" in k for k in sec["data"][0])
    # pfam truth, so no curation footnote and no mark on the tick label either
    assert "curated" not in sec["description"]
    assert not any(bmi.CURATION_CAVEAT_MARK in g
                   for g in (v["group"] for v in sec["data"][0].values()))


def test_the_swissprot_panel_footnotes_ciona_as_a_curation_depth_caveat(tmp_path):
    sprot = pl.DataFrame([
        {**r, "truth_set": "swissprot", "split": "all"}
        for r in pl.concat([
            METRICS,
            pl.DataFrame([row("hmmer3_phmmer", "default", "ciona", 550, 0.02),
                          row("foldseek", "3di_aa", "ciona", 550, 0.02),
                          row("kmerseek", "hp_pbotc_1st_ed2_k19_lcFalse", "ciona", 550,
                              0.02),
                          row("kmerseek", "dayhoff6_k12_lcTrue", "ciona", 550, 0.02)]),
        ]).to_dicts()])
    sec = build(tmp_path, sprot, name="qfo_tool_by_species_swissprot")
    assert "23 curated proteins" in sec["description"]
    assert "annotation failure" not in sec["description"]
    assert any(bmi.CURATION_CAVEAT_MARK in v["group"] for v in sec["data"][0].values())


def test_every_headline_metric_present_gets_its_own_panel(tmp_path):
    sec = build(tmp_path)
    names = [d["name"] for d in sec["pconfig"]["data_labels"]]
    assert len(sec["data"]) == len(names) >= 4


def test_the_table_carries_the_same_numbers(tmp_path):
    bmi.section_tool_by_species(tmp_path, METRICS)
    tbl = json.loads((tmp_path / "qfo_tool_by_species_table_pfam_mqc.json").read_text())
    keys = list(tbl["data"])
    assert keys[0].startswith("mouse") and keys[1].startswith("ecoli"), "by divergence"
    ecoli = tbl["data"][keys[1]]
    assert ecoli["Mya"] == 2000
    fold_col = next(k for k in ecoli if k.startswith("foldseek"))
    assert ecoli[fold_col] == 0.10


def test_the_table_ends_with_a_mean_row_that_names_its_aggregation(tmp_path):
    bmi.section_tool_by_species(tmp_path, METRICS)
    tbl = json.loads((tmp_path / "qfo_tool_by_species_table_pfam_mqc.json").read_text())
    last = list(tbl["data"])[-1]
    assert last == "mean over 2 proteomes"
    fold_col = next(k for k in tbl["data"][last] if k.startswith("foldseek"))
    assert tbl["data"][last][fold_col] == pytest_approx((0.70 + 0.10) / 2)


def pytest_approx(value, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return other is not None and abs(other - value) < tol

        def __repr__(self):
            return f"~{value}"
    return _Approx()


def test_a_run_with_no_divergence_column_is_skipped(tmp_path):
    bmi.section_tool_by_species(tmp_path, METRICS.drop("species_mya"))
    assert not list(tmp_path.glob("qfo_tool_by_species*"))
