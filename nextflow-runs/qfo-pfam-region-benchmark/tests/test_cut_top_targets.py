"""bin/cut_top_targets.py: each query keeps its best N targets under each ranking, ties at the
cut included, and the union over rankings.

Query Q1 has four targets, scored by their best region:

  target  region_mean_idf (higher better)  region_evalue (lower better)
  A       9 (a second region scores 1)     50
  B       7                                 5
  C       7                                 0.1
  D       2                                 0.01

By mean IDF with --top 2 the edge is 7, so A, B and C stay (B and C tie) and D goes. By
E-value with --top 2 the edge is 0.1, so D and C stay. The union keeps all four. Q2 has one
target and keeps it under every setting.
"""

import math
import subprocess
import sys
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parents[1] / "bin"
ROWS = [("Q1", "A", 9.0, 50.0), ("Q1", "A", 1.0, 900.0), ("Q1", "B", 7.0, 5.0),
        ("Q1", "C", 7.0, 0.1), ("Q1", "D", 2.0, 0.01), ("Q2", "E", 0.5, 3.0)]


def run(tmp_path, top, *rank_by, rows=ROWS):
    pl.DataFrame(rows, schema=["query_name", "target_name", "region_mean_idf", "region_evalue"],
                 orient="row").write_parquet(tmp_path / "in.parquet")
    cmd = [sys.executable, str(BIN / "cut_top_targets.py"), "--in", tmp_path / "in.parquet",
           "--out", tmp_path / "out.parquet", "--top", str(top)]
    for r in rank_by:
        cmd += ["--rank-by", r]
    proc = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = pl.read_parquet(tmp_path / "out.parquet")
    return sorted(set(zip(out["query_name"], out["target_name"]))), out, proc.stdout


def test_ties_at_the_cut_stay_and_every_region_of_a_kept_target_stays(tmp_path):
    kept, out, log = run(tmp_path, 2)
    assert kept == [("Q1", "A"), ("Q1", "B"), ("Q1", "C"), ("Q2", "E")]
    assert out.filter(pl.col("target_name") == "A").height == 2
    assert "kept 5 of 6 regions" in log


def test_lower_is_better_for_min(tmp_path):
    kept, _, _ = run(tmp_path, 2, "region_evalue:min")
    assert kept == [("Q1", "C"), ("Q1", "D"), ("Q2", "E")]


def test_union_of_two_rankings(tmp_path):
    kept, _, log = run(tmp_path, 2, "region_mean_idf:max", "region_evalue:min")
    assert kept == [("Q1", "A"), ("Q1", "B"), ("Q1", "C"), ("Q1", "D"), ("Q2", "E")]
    assert "region_mean_idf (highest) or region_evalue (lowest)" in log


def test_an_arm_without_e_values_is_cut_by_the_other_ranking_only(tmp_path):
    # A refused Karlin-Altschul fit writes region_evalue = inf on every row. Counted, every
    # target would tie at the cut and D would stay.
    rows = [(q, t, idf, math.inf) for q, t, idf, _ in ROWS]
    kept, _, _ = run(tmp_path, 2, "region_mean_idf:max", "region_evalue:min", rows=rows)
    assert kept == [("Q1", "A"), ("Q1", "B"), ("Q1", "C"), ("Q2", "E")]


def test_a_large_top_keeps_everything(tmp_path):
    _, out, _ = run(tmp_path, 1_000, "region_mean_idf:max", "region_evalue:min")
    assert out.height == 6
