"""Notebook 287's sum statistic against NCBI BLAST's own numbers, and the chain search on
small hand-made cases."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import sum_statistics_utils as ss  # noqa: E402

# BLAST's tabulated Prob(r, s) (blast_stat.c, s_BlastSumP, kTab2 and kTab4), read
# 2026-10-13. Entry i of the table is at s = (size - i - 4r) / 2, size = len(table) - 1.
KTAB2 = [0.01669, 0.0249, 0.03683, 0.05390, 0.07794, 0.1111, 0.1559, 0.2146, 0.2890,
         0.3794, 0.4836, 0.5965, 0.7092, 0.8114, 0.8931, 0.9490, 0.9806, 0.9944, 0.9989]
KTAB4 = [2.658e-07, 4.064e-07, 6.203e-07, 9.450e-07, 1.437e-06, 2.181e-06, 3.302e-06,
         4.990e-06, 7.524e-06, 1.132e-05, 1.698e-05, 2.541e-05, 3.791e-05, 5.641e-05,
         8.368e-05, 0.0001237, 0.0001823, 0.0002677, 0.0003915, 0.0005704, 0.0008275,
         0.001195, 0.001718, 0.002457, 0.003494, 0.004942, 0.006948, 0.009702, 0.01346,
         0.01853, 0.02532, 0.03431, 0.04607, 0.06128, 0.08068, 0.1051, 0.1352, 0.1719,
         0.2157, 0.2669, 0.3254, 0.3906, 0.4612, 0.5355, 0.6110, 0.6849, 0.7544, 0.8168,
         0.8699, 0.9127, 0.9451, 0.9679, 0.9827, 0.9915, 0.9963]


@pytest.mark.parametrize("r,table", [(2, KTAB2), (4, KTAB4)])
def test_log_sum_p_matches_blast_table(r, table):
    size = len(table) - 1
    for i, want in enumerate(table):
        s = (size - i - 4 * r) / 2
        got = math.exp(ss.log_sum_p(r, s))
        assert got == pytest.approx(want, rel=0.02, abs=2e-4), (r, s)


@pytest.mark.parametrize("r", [2, 3, 5, 8])
def test_integral_meets_large_s_form(r):
    """Just below the switch to BLAST's large-s form, the integral agrees with it."""
    s = r * r + r - 1 - 1e-9
    big = math.log(r) + (r - 1) * math.log(s) - s - 2 * math.lgamma(r + 1)
    assert ss.log_sum_p(r, s) == pytest.approx(big, abs=0.01)


def test_one_call_keeps_its_own_evalue():
    # y = ln(N / (n_t E)) -> E back
    n_db, n_t, e = 1.05e7, 239.0, 1954.6
    y = math.log(n_db / (n_t * e))
    assert ss.chain_evalue(y, 1, n_db, n_t) == pytest.approx(e)


def test_two_strong_calls_beat_either_alone():
    n_db, n_t = 1.05e7, 239.0
    e1, e2 = 1e-3, 1e-2
    y1, y2 = (math.log(n_db / (n_t * e)) for e in (e1, e2))
    assert ss.chain_evalue(y1 + y2, 2, n_db, n_t) < e1


def test_best_chain_order_overlap_gap_and_setting():
    rules = ss.ChainRules(max_gap=50, across_settings=True)
    # call 0: q 0-20 t 100-120; call 1: q 30-50 t 140-160 (in order, gaps 10 and 20);
    # call 2 overlaps call 1 on the query; call 3 is out of order on the target
    qs = np.array([0, 30, 45, 60]); qe = np.array([20, 50, 65, 80])
    ts = np.array([100, 140, 170, 10]); te = np.array([120, 160, 190, 30])
    y = np.array([3.0, 2.0, 5.0, 4.0]); st = np.array(["a", "b", "a", "a"])
    out = {r: (s, m) for r, s, m in ss.best_chain(qs, qe, ts, te, y, st, rules)}
    assert out[1] == (5.0, [2])
    assert out[2] == (8.0, [0, 2])  # 0 then 2: q gap 25, t gap 50, no overlap
    assert 3 not in out  # 0,1,2 is not allowed: 2 overlaps 1 on the query
    same = ss.best_chain(qs, qe, ts, te, y, st, ss.ChainRules(max_gap=50, across_settings=False))
    assert {r: m for r, _, m in same}[2] == [0, 2]  # both in setting "a"
    tight = ss.best_chain(qs, qe, ts, te, y, st, ss.ChainRules(max_gap=20))
    assert {r: m for r, _, m in tight}[2] == [0, 1]


def test_strong_chain_keeps_a_finite_log_evalue():
    """Two calls at E = 1e-300 each: the chain's E underflows a double, its log must not."""
    n_db, n_t = 5e5, 400.0
    y = math.log(n_db / n_t) - math.log(1e-300)
    log_e = ss.chain_log_evalue(2 * y, 2, n_db, n_t)
    assert math.isfinite(log_e) and log_e < math.log(1e-300)
