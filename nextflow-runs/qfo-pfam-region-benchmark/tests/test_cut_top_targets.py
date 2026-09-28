"""bin/cut_top_targets.py: each query keeps its best N targets, ties at the cut included.

Query Q1 has four targets. Their best regions score 9 (A), 7 (B), 7 (C) and 2 (D); A has a
second, weaker region. With --top 2 the floor is Q1's second-best target score, 7, so A, B
and C stay (B and C tie at the cut) and D goes. Q2 has one target and keeps it.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parents[1] / "bin"


def run(tmp_path, top):
    rows = [("Q1", "A", 9.0), ("Q1", "A", 1.0), ("Q1", "B", 7.0), ("Q1", "C", 7.0),
            ("Q1", "D", 2.0), ("Q2", "E", 0.5)]
    pl.DataFrame(rows, schema=["query_name", "target_name", "region_mean_idf"], orient="row").write_parquet(
        tmp_path / "in.parquet")
    proc = subprocess.run([sys.executable, str(BIN / "cut_top_targets.py"), "--in", tmp_path / "in.parquet",
                           "--out", tmp_path / "out.parquet", "--top", str(top)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return pl.read_parquet(tmp_path / "out.parquet"), proc.stdout


def test_ties_at_the_cut_stay_and_every_region_of_a_kept_target_stays(tmp_path):
    out, log = run(tmp_path, 2)
    kept = sorted(set(zip(out["query_name"], out["target_name"])))
    assert kept == [("Q1", "A"), ("Q1", "B"), ("Q1", "C"), ("Q2", "E")]
    assert out.filter(pl.col("target_name") == "A").height == 2
    assert "kept 5 of 6 regions" in log


def test_top_one_keeps_only_the_best_target(tmp_path):
    out, _ = run(tmp_path, 1)
    assert sorted(set(zip(out["query_name"], out["target_name"]))) == [("Q1", "A"), ("Q2", "E")]


def test_a_large_top_keeps_everything(tmp_path):
    out, _ = run(tmp_path, 1_000)
    assert out.height == 6
