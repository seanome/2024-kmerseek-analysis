"""Every tool against every target proteome, on every headline metric.

The Divergence section covers three metrics and whichever variants topped the leaderboard,
which on a full sweep is mostly kmerseek combos. This section is one line per tool so
"does phmmer hold up further out than foldseek" is readable directly.
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


def test_every_tool_gets_a_line(tmp_path):
    first = build(tmp_path)["data"][0]
    assert {label.split()[0] for label in first} == {"hmmer3_phmmer", "foldseek", "kmerseek"}


def test_only_one_line_per_tool(tmp_path):
    # Not the leaderboard's top-N kmerseek combos: those would crowd out the baselines.
    first = build(tmp_path)["data"][0]
    assert len([k for k in first if k.startswith("kmerseek")]) == 1


def test_points_are_keyed_by_divergence_time(tmp_path):
    first = build(tmp_path)["data"][0]
    phmmer = next(v for k, v in first.items() if k.startswith("hmmer3_phmmer"))
    assert set(phmmer) == {"90", "2000"}, "keyed by Mya, the axis the question is about"
    assert phmmer["2000"] == 0.55


def test_the_shape_that_answers_the_question_survives(tmp_path):
    # foldseek beats phmmer on mouse and loses badly on ecoli. If the section flattened
    # species away, this would be unreadable.
    first = build(tmp_path)["data"][0]
    fold = next(v for k, v in first.items() if k.startswith("foldseek"))
    phm = next(v for k, v in first.items() if k.startswith("hmmer3_phmmer"))
    assert fold["90"] > phm["90"] and fold["2000"] < phm["2000"]


def test_every_headline_metric_present_gets_its_own_panel(tmp_path):
    sec = build(tmp_path)
    names = [d["name"] for d in sec["pconfig"]["data_labels"]]
    assert len(sec["data"]) == len(names) >= 4


def test_the_table_carries_the_same_numbers(tmp_path):
    bmi.section_tool_by_species(tmp_path, METRICS)
    tbl = json.loads((tmp_path / "qfo_tool_by_species_table_pfam_mqc.json").read_text())
    assert list(tbl["data"]) == ["mouse", "ecoli"], "ordered by divergence"
    assert tbl["data"]["ecoli"]["Mya"] == 2000
    fold_col = next(k for k in tbl["data"]["ecoli"] if k.startswith("foldseek"))
    assert tbl["data"]["ecoli"][fold_col] == 0.10


def test_a_run_with_no_divergence_column_is_skipped(tmp_path):
    bmi.section_tool_by_species(tmp_path, METRICS.drop("species_mya"))
    assert not list(tmp_path.glob("qfo_tool_by_species*"))
