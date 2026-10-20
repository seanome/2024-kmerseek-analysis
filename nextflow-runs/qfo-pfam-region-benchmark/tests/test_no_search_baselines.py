"""The no-search baselines: notebook 231's scan copied unchanged, runs, SEG intervals,
the frequency table, the disorder truth parser, and the applicability marks."""

import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(PIPE / "bin"))
sys.path.insert(0, str(PIPE.parents[1] / "notebooks"))

import build_sprot_disorder_truth as bdt  # noqa: E402
import no_search_baselines as nsb  # noqa: E402

APPLIES = PIPE / "assets" / "no_search_baselines.tsv"


def random_protein(rng, n):
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n))


def has_tie_window(seq, window=19):
    """A window whose mean is exactly 1.6: the one place the exact test and notebook 231's
    floating-point test may disagree."""
    t = [round(nsb.KYTE_DOOLITTLE.get(c, 0.0) * 10) for c in seq]
    return any(sum(t[i:i + window]) == 16 * window for i in range(len(t) - window + 1))


def test_kd_matches_notebook_231_away_from_ties():
    """Notebook 231's kd_transmem_calls, with the label a parameter: the same calls on every
    protein that has no window of mean exactly 1.6."""
    scu = pytest.importorskip("swissprot_control_utils")
    rng = random.Random(231)
    seqs = {f"P{i}": random_protein(rng, rng.randint(10, 400)) for i in range(200)}
    # Hydrophobic stretches, so there are calls to compare.
    seqs.update({f"TM{i}": random_protein(rng, 40) + "L" * 22 + random_protein(rng, 40)
                 for i in range(20)})
    seqs = {k: v for k, v in seqs.items() if not has_tie_window(v)}
    assert len(seqs) > 150
    ours = nsb.kd_calls(seqs, "TRANSMEM").rename({"query": "query_acc",
                                                  "label": "pfam_id"})
    theirs = scu.kd_transmem_calls(seqs)
    cols = ["query_acc", "pfam_id", "qstart", "qend"]
    assert ours.height > 20
    assert ours.select(cols).sort(cols).equals(theirs.select(cols).sort(cols))
    j = ours.join(theirs, on=cols)
    assert (j["score"] - j["score_right"]).abs().max() < 1e-9
    assert set(nsb.kd_calls(seqs, "INTRAMEM")["label"]) == {"INTRAMEM"}


def test_kd_window_of_mean_exactly_threshold_is_not_called():
    # In tenths: 13 A (13 x 18) + 2 I (2 x 45) + 3 G (3 x -4) + 1 S (-8) = 304 = 16 x 19,
    # a mean of exactly 1.6. Swapping the S for a fourth G gives 308, mean 1.62. D (-35)
    # on both sides keeps every other window far below.
    tie = "D" * 25 + "A" * 13 + "II" + "GGG" + "S" + "D" * 25
    above = "D" * 25 + "A" * 13 + "II" + "GGGG" + "D" * 25
    assert nsb.kd_calls({"X": above}, "TRANSMEM").height == 1
    assert nsb.kd_calls({"X": tie}, "TRANSMEM").height == 0


def test_kd_call_spans_whole_windows():
    seq = "D" * 30 + "L" * 19 + "D" * 30
    c = nsb.kd_calls({"X": seq}, "TRANSMEM")
    # L is 3.8 and D is -3.5, so a window with up to 5 D residues still averages above 1.6
    # ((14 x 3.8 - 5 x 3.5) / 19 = 1.88; with 6 D it is 1.49). Those windows start at
    # residues 26 to 36, and the call covers them whole: 26 to 36 + 18 = 54.
    assert c.select("qstart", "qend").rows() == [(26, 54)]


def test_runs_strictly_above_cutoff_and_min_length():
    s = np.array([0.6] * 20 + [0.5] + [0.9] * 19 + [0.1] + [0.7] * 25)
    c = nsb.run_calls({"A": s}, ["REGION_DISORDERED"], cutoff=0.5, min_length=20)
    # 0.5 is not above 0.5; the 19-residue run is too short.
    assert c.select("qstart", "qend").rows() == [(1, 20), (42, 66)]
    assert c["score"][0] == pytest.approx(0.6)


def test_read_residue_scores_rejects_gaps(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text("A\t1\t0.1\nA\t3\t0.2\n")
    with pytest.raises(SystemExit):
        nsb.read_residue_scores(p)


def test_segmasker_intervals_are_zero_based(tmp_path):
    # segmasker -outfmt interval writes 0-based inclusive ends; checked against its
    # -outfmt fasta lower-casing on MYMX_HUMAN (first masked residue at 0-based 3).
    p = tmp_path / "seg.txt"
    p.write_text(">sp|A0A1B0GTQ4|MYMX_HUMAN Protein myomixer\n3 - 39\n>sp|Q1|X\n0 - 11\n"
                 "20 - 31\n")
    c = nsb.seg_calls(nsb.read_segmasker_intervals(p), ["COMPBIAS"])
    assert c.select("query", "qstart", "qend", "score").rows() == [
        ("A0A1B0GTQ4", 4, 40, 37.0), ("Q1", 1, 12, 12.0), ("Q1", 21, 32, 12.0)]


def test_frequency_excludes_queries_and_counts_proteins():
    ann = pl.DataFrame({
        "accession": ["R1", "R1", "R1", "R2", "R3", "Q1", "Q1", "Q2"],
        "pfam_id":   ["PF1", "PF1", "PF1", "PF2", "PF2", "PF9", "PF9", "PF9"],
    })
    lengths = {"R1": 150, "R2": 150, "R3": 150, "Q1": 150, "Q2": 150}
    t = nsb.frequency_table(ann, lengths, exclude={"Q1", "Q2"})
    row = t.filter(pl.col("bin") == "100-199").row(0, named=True)
    # PF1 has three instances but one protein; PF2 is held by two proteins.
    assert (row["pfam_id"], row["n_proteins_family"], row["n_proteins_bin"]) == ("PF2", 2, 3)
    calls = nsb.frequency_calls({"Q1": "A" * 150, "Q3": "A" * 50}, t)
    assert calls.select("query", "label", "qstart", "qend").rows() == [
        ("Q1", "PF2", 1, 150)]   # no reference protein in Q3's bin, so no call


def test_length_bins():
    assert [nsb.length_bin(n) for n in (0, 99, 100, 1199, 1200, 9000)] == [
        "0-99", "0-99", "100-199", "800-1199", "1200+", "1200+"]


DAT = """\
ID   AAA_HUMAN               Reviewed;         300 AA.
AC   P11111; Q99999;
FT   REGION          1..44
FT                   /note="Disordered"
FT                   /evidence="ECO:0000256|SAM:MobiDB-lite"
FT   REGION          50..80
FT                   /note="Interaction with X"
FT   COMPBIAS        10..30
FT                   /note="Polar residues"
FT                   /evidence="ECO:0000256|SAM:MobiDB-lite"
FT   REGION          <100..120
FT                   /note="Disordered"
FT   TRANSMEM        200..220
//
ID   BBB_HUMAN               Reviewed;         100 AA.
AC   P22222;
FT   REGION          1..20
FT                   /note="Disordered"
//
"""


def test_disorder_truth_keeps_disordered_regions_and_compbias(tmp_path):
    dat = tmp_path / "s.dat"
    dat.write_text(DAT)
    rows = bdt.parse(dat, {"P11111"})
    got = sorted((r["accession"], r["pfam_id"], r["domain_start"], r["domain_end"],
                  r["protein_length"]) for r in rows)
    assert got == [("P11111", "COMPBIAS", 10, 30, 300),
                   ("P11111", "REGION_DISORDERED", 1, 44, 300)]


def test_applicability_table_names_every_call_file():
    a = nsb.read_applies(APPLIES)
    pairs = set(zip(a["tool"], a["variant"]))
    assert pairs == {
        ("kyte_doolittle", "w19_t1.6_transmem"), ("kyte_doolittle", "w19_t1.6_intramem"),
        ("iupred", "long_t0.5_min20_disordered"), ("iupred", "long_t0.5_min20_compbias"),
        ("iupred", "long_t0.5_min20_region"),
        ("metapredict", "v3_t0.5_min20_disordered"), ("metapredict", "v3_t0.5_min20_compbias"),
        ("metapredict", "v3_t0.5_min20_region"),
        ("seg", "segmasker_default_compbias"), ("seg", "segmasker_default_repeat"),
        ("pfam_family_frequency", "length_bin"),
    }
    # Each label in the table is a label its truth set holds.
    for ts, label in (("swissprot_disordered", "REGION_DISORDERED"),
                      ("swissprot_compbias", "COMPBIAS")):
        assert set(a.filter(pl.col("truth_set") == ts)["label"]) == {label}


def test_mark_applicable():
    m = pl.DataFrame({
        "tool": ["kyte_doolittle"] * 3 + ["pfam_family_frequency"] * 2,
        "variant": ["w19_t1.6_transmem"] * 3 + ["length_bin"] * 2,
        "truth_set": ["swissprot"] * 3 + ["pfam", "pfam"],
        "stratum_axis": ["feature_type", "feature_type", "all", "all", "plddt"],
        "stratum": ["TRANSMEM", "DOMAIN", "all", "all", "70-90"],
    })
    out = nsb.mark_applicable(m, nsb.read_applies(APPLIES))
    assert out["applicable"].to_list() == [True, False, False, True, True]
    assert set(out["tool_kind"]) == {"no search"}


def test_disorder_truth_cli_writes_one_table_per_label(tmp_path):
    import json
    import subprocess

    dat = tmp_path / "s.dat"
    dat.write_text(DAT)
    fa = tmp_path / "q.fasta"
    fa.write_text(">sp|P11111|AAA_HUMAN\nMAAA\n>sp|P33333|CCC_HUMAN\nMCCC\n")
    out = {k: tmp_path / f"{k}.parquet" for k in ("disordered", "compbias")}
    subprocess.run([sys.executable, str(PIPE / "bin" / "build_sprot_disorder_truth.py"),
                    "--sprot-dat", dat, "--query-fasta", fa,
                    "--disordered-out", out["disordered"], "--compbias-out", out["compbias"],
                    "--summary-out", tmp_path / "s.json"], check=True, capture_output=True)
    d, c = pl.read_parquet(out["disordered"]), pl.read_parquet(out["compbias"])
    assert d.select("accession", "pfam_id", "domain_start", "domain_end").rows() == [
        ("P11111", "REGION_DISORDERED", 1, 44)]
    assert c["pfam_id"].to_list() == ["COMPBIAS"]
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["by_label"]["REGION_DISORDERED"]["n_mobidb_lite"] == 1
