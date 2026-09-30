"""scripts/reduce_elm_cover.py on a hand-built arm whose answers are worked out below.

Query Q1 is 200 residues with one motif at [40, 50) (0-based, end-exclusive): 10 residues,
so a call covers it with 8 (80%). Calls, kmerseek arm, ranked by region_mean_idf:

  target  region     score  covers?                      p_place
  T3      [41, 48)   10     no: 7 residues
  T2      [0, 200)    9     yes: the whole protein       1 of 1 window positions = 1.0
  T1      [38, 48)    5     yes: 8 residues              s in [38, 42] = 5 of 191 = 0.026

T1 is Q1's chicken ortholog. Target ranks by best call: T3 1, T2 2, T1 3.
Q2 has a motif and no call at all.

The motif projects onto T1 at [100, 110). The kmerseek T1 call's target side is [0, 10), so
it is on the ortholog but NOT on position. The phmmer T1 call's target side is 1-based
101..108, [100, 108): 8 of 10 projected residues, so it IS on position, and only after the
start is shifted (read as [101, 108) it holds 7).
"""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "reduce_elm_cover.py"
BIN = Path(__file__).resolve().parents[1] / "bin"


def kmerseek_rows(rows):
    return pl.DataFrame(
        {"query_name": [f"sp|{q}|{q}_HUMAN" for q, *_ in rows],
         "target_name": [f"tr|{t}|{t}_CHICK" for _, t, *_ in rows],
         "region_start": [r[2] for r in rows], "region_end": [r[3] for r in rows],
         "target_start": [r[5] for r in rows], "target_end": [r[5] + r[3] - r[2] for r in rows],
         "region_mean_idf": [float(r[4]) for r in rows]},
        schema_overrides={"region_start": pl.Int64, "region_end": pl.Int64,
                          "target_start": pl.Int64, "target_end": pl.Int64})


@pytest.fixture
def workdir(tmp_path):
    res = tmp_path / "results"
    (res / "kmerseek").mkdir(parents=True)
    (res / "regions" / "hmmer3_phmmer").mkdir(parents=True)
    (res / "regions" / "foldseek").mkdir(parents=True)
    # Foldseek counts along the AlphaFold model. Q1's model has 210 residues against 200 in
    # the FASTA, so Q1 is not scoreable for Foldseek; Q2's model matches.
    with gzip.open(res / "regions" / "foldseek" / "human_vs_chicken.foldseek.tsv.gz", "wt") as fh:
        fh.write("Q1\tT1\t41\t50\t101\t110\t300\t1e-9\n")
    st = tmp_path / "structures"
    st.mkdir()
    for acc, n in [("Q1", 210), ("Q2", 60)]:
        (st / f"AF-{acc}-F1-model_v6.cif").write_text("".join(
            f"ATOM {i} C CA . ALA A 1 {i} ? 0 0 0 1 90 1 ALA A CA 1\n" for i in range(1, n + 1)))
    kmerseek_rows([("Q1", "T3", 41, 48, 10, 0), ("Q1", "T2", 0, 200, 9, 0), ("Q1", "T1", 38, 48, 5, 0)]
                  ).write_parquet(res / "kmerseek" / "human_vs_chicken.toy.k5.lcfalse.regions.parquet")
    (res / "kmerseek" / "human_vs_chicken.toy.k6.lcfalse.regions.parquet").write_bytes(b"")
    # phmmer, 1-based inclusive: 43..50 is [42, 50), 8 motif residues, so it covers.
    # Read without the shift it would be [43, 50), 7 residues, and would not.
    with gzip.open(res / "regions" / "hmmer3_phmmer" / "human_vs_chicken.hmmer3_phmmer.tsv.gz", "wt") as fh:
        fh.write("sp|Q1|Q1_HUMAN\ttr|T1|T1_CHICK\t43\t50\t101\t108\t50.0\t1e-5\n")
    (tmp_path / "instances.tsv").write_text(
        "elm_instance\telm_class\taccession\tstart\tend\tmotif_length\tlength_bin\tin_midi_plus\tregex\n"
        "E1\tLIG_toy\tQ1\t40\t50\t10\t7-10 aa\tfalse\tX\n"
        "E2\tLIG_toy\tQ2\t5\t11\t6\t3-6 aa\tfalse\tX\n")
    (tmp_path / "orthologs.tsv").write_text("accession\tspecies\tortholog\nQ1\tchicken\tT1\nQ1\tmouse\tM1\n")
    (tmp_path / "projections.tsv").write_text(
        "elm_instance\taccession\tspecies\tortholog\tproj_start\tproj_end\n"
        "E1\tQ1\tchicken\tT1\t100\t110\nE1\tQ1\tmouse\tM1\t0\t10\n")
    (tmp_path / "q.fasta").write_text(">sp|Q1|Q1_HUMAN x\n" + "A" * 200 + "\n>sp|Q2|Q2_HUMAN y\n" + "A" * 60 + "\n")
    return tmp_path


def run(workdir, *extra, ok=True):
    cmd = [sys.executable, str(SCRIPT), "--results", workdir / "results", "--species", "chicken",
           "--instances", workdir / "instances.tsv", "--orthologs", workdir / "orthologs.tsv",
           "--projections", workdir / "projections.tsv", "--query-fasta", workdir / "q.fasta",
           "--structures", workdir / "structures", "--qfo-bin", BIN, "--out-dir", workdir / "out",
           *extra]
    proc = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, cwd=workdir)
    assert (proc.returncode == 0) == ok, proc.stderr
    return proc


def write_hhblits(workdir, rows):
    d = workdir / "results" / "regions" / "hhblits"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "human_vs_chicken.hhblits.tsv.gz", "wt") as fh:
        fh.writelines(rows)


def row(workdir, arm, inst):
    t = pl.read_parquet(workdir / "out" / f"{arm}.instances.parquet")
    return t.filter(pl.col("elm_instance") == inst).row(0, named=True)


def test_ranks_placement_and_ortholog(workdir):
    run(workdir, "--arm", "toy.k5.lcfalse")
    r = row(workdir, "toy.k5.lcfalse", "E1")
    assert r["best_rank"] == 2                 # T2, the whole-protein call
    assert r["best_rank_placed"] == 3          # T2 fails the placement null, T1 passes
    assert r["ortholog_rank"] == 3 and r["ortholog_rank_placed"] == 3
    assert r["n_covering_calls"] == 2 and r["n_covering_targets"] == 2
    assert r["min_p_place"] == pytest.approx(5 / 191)
    assert r["has_ortholog"] and r["has_projection"]
    assert r["ortholog_rank_on_position"] is None   # target side [0, 10), motif at [100, 110)


def test_uncovered_instance_is_a_row_with_no_rank(workdir):
    run(workdir, "--arm", "toy.k5.lcfalse")
    r = row(workdir, "toy.k5.lcfalse", "E2")
    assert r["best_rank"] is None and r["n_covering_calls"] == 0 and not r["has_ortholog"]


def test_list_length_cuts_targets_before_cover(workdir):
    run(workdir, "--arm", "toy.k5.lcfalse", "--list-length", "2")
    r = row(workdir, "toy.k5.lcfalse", "E1")
    assert r["best_rank"] == 2 and r["best_rank_placed"] is None and r["ortholog_rank"] is None


def test_comparator_starts_are_shifted_to_zero_based(workdir):
    run(workdir, "--arm", "hmmer3_phmmer")
    r = row(workdir, "hmmer3_phmmer", "E1")
    assert r["best_rank"] == 1 and r["ortholog_rank"] == 1
    assert r["ortholog_rank_on_position"] == 1


def test_empty_arm_is_missing_not_zero(workdir):
    run(workdir, "--arm", "toy.k6.lcfalse")
    s = json.loads((workdir / "out" / "toy.k6.lcfalse.summary.json").read_text())
    assert s["status"] == "empty"
    assert not (workdir / "out" / "toy.k6.lcfalse.instances.parquet").exists()


def test_tasks_split_the_arms_and_rerun_skips(workdir):
    names = set()
    for task in range(2):
        out = run(workdir, "--task", str(task), "--n-tasks", "2").stdout
        names |= {line.split(":")[0] for line in out.splitlines() if line and not line.startswith("task")}
    assert names == {"toy.k5.lcfalse", "toy.k6.lcfalse", "hmmer3_phmmer", "foldseek"}
    assert "already scored" in run(workdir).stdout


def test_structure_tool_skips_queries_whose_model_differs(workdir):
    run(workdir, "--arm", "foldseek")
    r = row(workdir, "foldseek", "E1")
    assert r["query_scoreable"] is False and r["best_rank"] is None   # Q1's model is 210, not 200
    assert row(workdir, "foldseek", "E2")["query_scoreable"] is True
    s = json.loads((workdir / "out" / "foldseek.summary.json").read_text())
    assert s["structure_status"] == {"differs": 1, "same": 1} and s["n_instances_scoreable"] == 1


def test_sequence_tools_score_every_query(workdir):
    run(workdir, "--arm", "hmmer3_phmmer")
    assert row(workdir, "hmmer3_phmmer", "E1")["query_scoreable"] is True


def test_a_rare_malformed_row_is_dropped_and_counted(workdir):
    # 150 good covering calls on 150 targets, and one row ending at 2_555 on a 200-aa query.
    good = [f"sp|Q1|Q1_HUMAN\ttr|X{i}|X{i}_CHICK\t41\t50\t1\t10\t{500 - i}\t1e-5\n" for i in range(150)]
    write_hhblits(workdir, good + ["sp|Q1|Q1_HUMAN\ttr|Y|Y_CHICK\t41\t2555\t1\t10\t900\t1e-9\n"])
    proc = run(workdir, "--arm", "hhblits")
    assert "dropped 1 covering calls" in proc.stderr
    s = json.loads((workdir / "out" / "hhblits.summary.json").read_text())
    assert s["n_covering_calls_past_query_end"] == 1
    assert row(workdir, "hhblits", "E1")["best_rank"] == 2   # Y ranked first, then dropped


def test_many_calls_past_the_end_stop_the_arm(workdir):
    write_hhblits(workdir, ["sp|Q1|Q1_HUMAN\ttr|Y|Y_CHICK\t41\t2555\t1\t10\t900\t1e-9\n"])
    proc = run(workdir, "--arm", "hhblits", ok=False)
    assert "coordinate convention" in proc.stderr


def test_kmerseek_subdir_reads_a_cut_store(workdir):
    store = workdir / "results" / "kmerseek_top1000"
    store.mkdir()
    (workdir / "results" / "kmerseek" / "human_vs_chicken.toy.k5.lcfalse.regions.parquet").rename(
        store / "human_vs_chicken.toy.k5.lcfalse.regions.parquet")
    out = run(workdir, "--kmerseek-subdir", "kmerseek_top1000", "--list-arms").stdout
    assert "toy.k5.lcfalse" in out and "toy.k6.lcfalse" not in out
