"""The first screen, the "level with" rule, and the two closing sections.

The head of the report is written by the builder (intro_text in report_header.yaml)
because the answer carries numbers. Three things have to hold: the numbers in the head
are the ones the conclusions compute; a gap narrower than the between-proteome SD is
called a tie and not a loss; and every flag name lives in "Reproducing this report" and
nowhere a reader who has not seen the pipeline would meet it.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402
from test_conclusions_section import metrics  # noqa: E402


def setup_function():
    bmi.CANONICAL = None
    bmi.QUERY_SET = None
    bmi.NOT_IN_RUN.clear()


def test_within_noise_calls_a_gap_inside_both_sds_a_tie():
    arm = {"v": 0.123, "sd": 0.074}
    assert bmi.within_noise(arm, [{"v": 0.140, "sd": 0.038}, {"v": 0.131, "sd": 0.027}])
    # One gap wider than the smaller SD of its pair: not a tie.
    assert not bmi.within_noise(arm, [{"v": 0.140, "sd": 0.038}, {"v": 0.200, "sd": 0.027}])
    # No SD on the arm itself: never a tie, because nothing says how noisy it is.
    assert not bmi.within_noise({"v": 0.123, "sd": None}, [{"v": 0.124, "sd": 0.05}])


def test_the_divergence_conclusion_says_level_with_when_the_gap_is_inside_the_noise():
    # ecoli rows are fmax * 0.9 for every arm, so at 2_000 Mya hhblits 0.162 and phmmer
    # 0.130 sit above kmerseek's 0.125. The between-proteome SD of each arm (three
    # proteomes at fmax, fmax, 0.9 fmax) is about 0.06 x fmax, so the phmmer gap (0.005)
    # is inside the noise and the hhblits gap (0.037) is not.
    text = bmi.conclusion_divergence(metrics(), "pfam")
    assert "behind" in text and "level with" not in text
    assert "between-proteome SD" in text
    assert "at least one of those gaps is wider than the noise" in text


def test_the_first_screen_carries_the_answer_the_conclusions_compute(tmp_path):
    m = metrics()
    f = bmi.overview_facts(m, 964)
    html = bmi.first_screen_html(m, f, "pfam")
    assert "What kmerseek is" in html
    assert "The answer, on the Pfam answer key" in html
    # The most distant proteome and the best baseline there, as the conclusions have them.
    assert "ecoli" in html and "hhblits" in html
    assert "ranks" in html and "of 5 arms on the leaderboard" in html
    # The glossary defines the words the sections use, with this run's alphabets.
    assert "Words used on every page" in html
    assert "polarity4" in html and "wwmj5" in html
    assert "<b>Reachable.</b>" in html
    # No flag names on the first screen.
    assert "--" not in html.replace("&mdash;", "")
    # Without --query-set nothing claims a subset.
    assert "test subset" not in html and "not the whole proteome" not in html


def test_the_query_set_label_reaches_the_head_and_the_header(tmp_path):
    bmi.QUERY_SET = "human proteins from chromosome 6, a test subset of the proteome"
    m = metrics()
    bmi.section_overview(tmp_path, m, 964, "pfam")
    header = (tmp_path / "report_header.yaml").read_text()
    assert header.startswith("intro_text: ")
    assert "not the whole proteome" in header
    assert '"Query": "human proteins from chromosome 6, a test subset of the proteome: 964 in the FASTA' in header
    overview = json.loads((tmp_path / "qfo_overview_mqc.json").read_text())
    assert "Query: human proteins from chromosome 6" in overview["data"]


def test_the_marked_arm_is_named_once_in_the_glossary_without_a_double_mark():
    bmi.CANONICAL = ("kmerseek", "polarity4_k17_lcFalse")
    m = metrics()
    html = bmi.first_screen_html(m, bmi.overview_facts(m, 964), "pfam")
    star = bmi.CANONICAL_MARK.strip()
    assert f"<b>{star}</b> marks <code>kmerseek polarity4_k17_lcFalse</code>" in html


def test_not_in_run_is_one_section_written_only_when_something_is_missing(tmp_path):
    bmi.section_not_in_run(tmp_path)
    assert not (tmp_path / "qfo_not_in_run_mqc.json").exists()
    bmi.not_in_run("The frontier", "no timing records")
    bmi.not_in_run("Selective pressure", "one bin")
    bmi.section_not_in_run(tmp_path)
    sec = json.loads((tmp_path / "qfo_not_in_run_mqc.json").read_text())
    assert sec["data"].count("Not built:") == 2
    assert "The frontier" in sec["data"] and "Selective pressure" in sec["data"]


def test_a_run_without_timings_lists_cost_once_instead_of_three_stub_sections(tmp_path):
    m = metrics()
    empty = pl.DataFrame()
    bmi.section_frontier(tmp_path, m, empty, 964, "pfam")
    bmi.section_resources(tmp_path, empty, 964)
    assert not (tmp_path / "qfo_frontier_mqc.json").exists()
    assert not (tmp_path / "qfo_resources_missing_mqc.json").exists()
    whats = [w for w, _ in bmi.NOT_IN_RUN]
    assert any("frontier" in w for w in whats)
    assert any("Resource usage" in w for w in whats)


def test_reproduce_holds_every_flag_and_file_name(tmp_path):
    bmi.CANONICAL = ("kmerseek", "polarity4_k17_lcFalse")
    bmi.section_reproduce(tmp_path, "qfo_pfam_region.2026-09-13.trace.txt", 12)
    sec = json.loads((tmp_path / "qfo_reproduce_mqc.json").read_text())
    for name in ["all_domain_metrics.parquet", "--report_trace", "make multiqc",
                 "storeDir", "*.timings.jsonl", "--canonical-variant kmerseek:polarity4_k17_lcFalse",
                 "attach_identity", 'STRATA["disorder_fine"]', "MIN_STRATUM_PROTEINS",
                 "evaluate_domain_calls.py", "UNFLOORED_AXES", "bin/gene_sets.py",
                 "notebooks 210-216", "bin/cafa_metrics.py", "--mobidb_cache",
                 "--query-set", "12 stored kmerseek timing records"]:
        assert name in sec["data"], name


def test_ordinal():
    assert [bmi.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st"]


def test_the_divergence_conclusion_calls_a_tie_a_tie():
    # kmerseek 0.123 against hhblits 0.140 at the far proteome, both with an SD of about
    # 0.07 across the three proteomes: the 0.017 gap is inside the noise.
    rows = []
    for tool, variant, vals in (("kmerseek", "wwmj5_k11_lcFalse", [0.05, 0.19, 0.123]),
                                ("hhblits", "default", [0.07, 0.21, 0.140]),
                                ("hmmer3_phmmer", "default", [0.02, 0.10, 0.060])):
        for (species, mya), v in zip([("mouse", 90.0), ("chicken", 320.0), ("ecoli", 2000.0)], vals):
            rows.append(dict(truth_set="pfam", tool=tool, variant=variant, species=species,
                             species_mya=mya, split="heldout", stratum_axis="all",
                             stratum="all", fmax=v, precision=0.1, recall_reachable=0.1,
                             auprc=0.1))
    text = bmi.conclusion_divergence(pl.DataFrame(rows), "pfam")
    assert "level with the tools above it" in text
    assert "this is a tie, not a loss" in text
    assert "ahead of the other 1" in text
