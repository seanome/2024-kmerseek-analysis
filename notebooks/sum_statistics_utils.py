"""Combine several ungapped kmerseek calls on one query-target pair into one E-value, the
way BLAST combines high-scoring segments with sum statistics.

Used by scripts/287_chain_scores.py (notebook 241's three queries), scripts/287_pfam998_chains.py
(the Pfam 998 all-against-all) and scripts/287_shuffle_chains.py (dipeptide-shuffled copies),
so all three come from one function.

Sources
-------
Karlin, S., and S. F. Altschul. "Applications and Statistics for Multiple High-Scoring
Segments in Molecular Sequences." *Proceedings of the National Academy of Sciences of the
United States of America* 90, no. 12 (1993): 5873-77. https://doi.org/10.1073/pnas.90.12.5873

Altschul, S. F., and W. Gish. "Local Alignment Statistics." *Methods in Enzymology* 266
(1996): 460-80. https://doi.org/10.1016/s0076-6879(96)66029-7

NCBI BLAST, which implements both: ncbi-cxx-toolkit-public, src/algo/blast/core/blast_stat.c
(s_BlastSumPCalc, s_BlastSumP, BLAST_LargeGapSumE, BLAST_GapDecayDivisor) and
include/algo/blast/core/blast_parameters.h (BLAST_GAP_DECAY_RATE 0.5 for ungapped search),
read 2026-10-13.

The formula
-----------
A call i in setting s (an alphabet and k) has raw score S_i, its own lambda_i, the setting's
K_s, and kmerseek's E-value E_i = K_s m N_s exp(-lambda_i S_i), with m the query length and
N_s the number of k-mers in the database ("total k-mers found" in the index log). Its
normalised score for this pair is

    y_i = lambda_i S_i - ln K_s - ln(m n_t) = ln(N_s / (n_t E_i))

with n_t the target length: the score in nats (natural-log units) less the size of this
one pair's search. y_i has the same chance distribution whatever the alphabet, which is why
calls from different alphabets can be added.

For r calls in the same order on both proteins (Altschul & Gish's "consistent ordering",
BLAST_LargeGapSumE):

    s = sum(y_i) + ln(r!)
    P = Prob(r, s), the chance that r unrelated segments sum to at least s
        (Karlin & Altschul 1993, equation 4 and the paragraph after it):

                  r^(r-2)        inf             inf
      Prob(r,s) = ------------  integral e^(-y) integral x^(r-2) exp(-e^(x - y/r)) dx dy
                  (r-1)!(r-2)!   y=s             x=0

        which for large s is close to e^(-s) s^(r-1) / (r! (r-1)!).
    E_pair = -ln(1 - P), the expected number of such chains on this pair by chance
    E      = E_pair * N / n_t, the same chain over the whole database

and BLAST divides E by its gap-decay divisor (1 - d) d^(r-1), d = 0.5, so a set of r calls
pays 2^(r-1) for having been picked out of all possible sets. We use the factor relative
to a single call, 2^(r-1), so that a chain of one call keeps kmerseek's own E-value.
The ln(r!) is the ordering rule: of the r! orders r chance segments can fall in, only one
is the same on both proteins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

GAP_DECAY_RATE = 0.5  # BLAST_GAP_DECAY_RATE for ungapped search, blast_parameters.h
E_FLOOR = 1e-300  # smallest E-value used; kmerseek can report 0 when it underflows
MAX_CHAIN = 8  # longest chain considered (BLAST has no cap); 40 of 934_198 Pfam 998 pairs reach it
_LN_FACT = [math.lgamma(r + 1) for r in range(MAX_CHAIN + 2)]


def _log_sum_p_integral(r: int, s: float, n_y: int = 801, n_x: int = 401) -> float:
    """ln Prob(r, s) from Karlin & Altschul's double integral (r >= 2), by the trapezoid
    rule: y from s to s + 50, x from 0 to max(y/r, 0) + 8, past which
    exp(-e^(x - y/r)) < e^-2980. The factor e^-s is taken out so a small P keeps its digits.
    """
    y = np.linspace(s, s + 50.0, n_y)
    hi = np.maximum(y / r, 0.0) + 8.0
    x = np.linspace(0.0, 1.0, n_x)[None, :] * hi[:, None]
    with np.errstate(divide="ignore"):
        log_x = np.log(x) if r > 2 else np.zeros_like(x)
    f = np.exp(
        (r - 2) * np.where(x > 0, log_x, -np.inf if r > 2 else 0.0)
        - np.exp(x - y[:, None] / r)
    )
    inner = np.trapezoid(f, x, axis=1)
    outer = np.trapezoid(np.exp(-(y - s)) * inner, y)
    const = (r - 2) * math.log(r) - _LN_FACT[r - 1] - _LN_FACT[r - 2]
    return min(0.0, const - s + math.log(outer))


_TABLE_STEP = 0.02
_TABLE: dict[int, tuple[float, np.ndarray]] = {}


def _table(r: int) -> tuple[float, np.ndarray]:
    """ln Prob(r, s) on a grid of s from BLAST's "certainly 1" bound (mean - 4 sd) to the
    start of the large-s form (r^2 + r - 1), step 0.02, built on first use."""
    if r not in _TABLE:
        lo = (
            r * (1 - math.log(r)) - 0.5 - 4 * math.sqrt(r)
        )  # BLAST's mean - 4 sd of the sum
        hi = r * r + r - 1
        grid = np.arange(lo, hi + _TABLE_STEP, _TABLE_STEP)
        _TABLE[r] = (lo, np.array([_log_sum_p_integral(r, s) for s in grid]))
    return _TABLE[r]


def log_sum_p(r: int, s: float) -> float:
    """ln Prob(r, s): ln of the chance that r unrelated segments' normalised scores, added,
    reach at least s (Karlin & Altschul 1993, equation 4). r = 1 is the Gumbel tail
    1 - exp(-e^-s). For r >= 2: 0 (P = 1) below BLAST's bound mean - 4 sd; above
    s = r^2 + r - 1, BLAST's large-s form r s^(r-1) e^-s / (r!)^2; in between, the double
    integral read off a table at step 0.02 (checked against BLAST's own tables in
    tests/test_sum_statistics_utils.py)."""
    if r == 1:
        return math.log(-math.expm1(-math.exp(-s))) if s > -50 else 0.0
    if s >= r * r + r - 1:
        return math.log(r) + (r - 1) * math.log(s) - s - 2 * _LN_FACT[r]
    lo, tab = _table(r)
    if s <= lo:
        return 0.0
    i = (s - lo) / _TABLE_STEP
    k = min(int(i), len(tab) - 2)
    a = i - k
    return float((1 - a) * tab[k] + a * tab[k + 1])


def chain_log_evalue(y_sum: float, r: int, n_db: float, n_t: float) -> float:
    """ln of the database E-value of r calls in the same order on both proteins whose
    pair-level normalised scores add up to y_sum (BLAST_LargeGapSumE, then the gap-decay
    factor relative to a single call). In log form so that strong chains do not all
    underflow to E = 0 and tie. For r = 1 this is ln of the call's own E-value."""
    if r == 1:
        return -y_sum + math.log(n_db / n_t)
    lp = log_sum_p(r, y_sum + _LN_FACT[r])
    # E_pair = -ln(1 - P); for small P it is P itself
    log_e_pair = (
        lp if lp < -20 else math.log(-math.log1p(-math.exp(lp))) if lp < 0 else math.inf
    )
    return log_e_pair + math.log(n_db / n_t) - (r - 1) * math.log(GAP_DECAY_RATE)


def chain_evalue(y_sum: float, r: int, n_db: float, n_t: float) -> float:
    """exp(chain_log_evalue): the chain's E-value over the whole database."""
    return math.exp(chain_log_evalue(y_sum, r, n_db, n_t))


@dataclass(frozen=True)
class ChainRules:
    """What may join a chain. Fixed in notebook 287's first cell before any run.

    max_gap           residues allowed between one call's end and the next call's start, on
                      each protein separately
    across_settings   True lets calls from different alphabets and k join one chain
    settings_penalty  ln of the number of settings searched, taken off every call's y
                      (Bonferroni: each call was the best of that many searches); 0 for none
    """

    max_gap: int = 200
    across_settings: bool = True
    settings_penalty: float = 0.0


def add_normalised_score(calls: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Columns qs, qe, ts, te (0-based, end-exclusive) and y = ln(N_s / (n_t E)).
    Needs region_start, region_end, target_start, region_evalue, n_db (N_s for the call's
    setting) and n_t (target length)."""
    return (
        calls.lazy()
        .filter(pl.col("region_evalue").is_finite())
        .with_columns(
            qs=pl.col("region_start").cast(pl.Int64),
            qe=pl.col("region_end").cast(pl.Int64),
            ts=pl.col("target_start").cast(pl.Int64),
        )
        .with_columns(te=pl.col("ts") + pl.col("qe") - pl.col("qs"))
        # A difference of logs: n_db / (n_t E) overflows for E near 1e-320. kmerseek
        # reports E = 0 when exp(-lambda S) underflows (16 calls in the Pfam 998 run), so
        # E is floored at 1e-300 (E_FLOOR) rather than dropped.
        .with_columns(
            y=pl.col("n_db").log()
            - pl.col("n_t").log()
            - pl.max_horizontal(pl.col("region_evalue"), pl.lit(E_FLOOR)).log()
        )
    )


def best_chain(qs, qe, ts, te, y, setting, rules: ChainRules):
    """Best chain of each length on one pair. Returns (r, sum of y, member indices into
    the arrays given) for every r up to MAX_CHAIN that has a chain. A call may follow
    another when it starts after that call ends on both proteins (no overlap), at most
    max_gap residues later on each, and, unless across_settings, in the same setting.
    Dynamic programme over calls sorted by query start:
    best[j, r] = y_j + max over allowed predecessors i of best[i, r - 1]."""
    n = len(qs)
    order = np.lexsort((ts, qs))
    qs, qe, ts, te, y, setting = (a[order] for a in (qs, qe, ts, te, y, setting))
    best = np.full((n, MAX_CHAIN + 1), -np.inf)
    back = np.full((n, MAX_CHAIN + 1), -1, dtype=np.int64)
    best[:, 1] = y
    # a predecessor ends at or before qs[j], so it started at least its own length earlier;
    # only calls starting at or after qs[j] - max_gap - longest call can qualify.
    longest = int((qe - qs).max()) if n else 0
    first = np.searchsorted(qs, qs - rules.max_gap - longest, side="left")
    for j in range(n):
        lo = first[j]
        if lo >= j:
            continue
        ok = (
            (qe[lo:j] <= qs[j])
            & (te[lo:j] <= ts[j])
            & (qs[j] - qe[lo:j] <= rules.max_gap)
            & (ts[j] - te[lo:j] <= rules.max_gap)
        )
        if not rules.across_settings:
            ok &= setting[lo:j] == setting[j]
        idx = lo + np.flatnonzero(ok)
        if idx.size == 0:
            continue
        cand = best[idx, 1:MAX_CHAIN]  # chains of length 1..MAX_CHAIN-1 ending at i
        k = np.argmax(cand, axis=0)
        top = cand[k, np.arange(cand.shape[1])]
        fin = np.isfinite(top)
        best[j, 2:][fin] = y[j] + top[fin]
        back[j, 2:][fin] = idx[k[fin]]
    out = []
    for r in range(1, MAX_CHAIN + 1):
        j = int(np.argmax(best[:, r]))
        if not np.isfinite(best[j, r]):
            break
        members, jj, rr = [], j, r
        while jj >= 0 and rr >= 1:
            members.append(int(order[jj]))
            jj, rr = back[jj, rr], rr - 1
        out.append((r, float(best[j, r]), members[::-1]))
    return out


def score_pairs(
    calls: pl.DataFrame, rules: ChainRules, keys=("query_name", "target_name")
) -> pl.DataFrame:
    """One row per query-target pair: the chain with the lowest E-value (chain_e and its
    natural log chain_log_e, which ranks without ties at underflow; its
    length chain_r and its members as row numbers of `calls`) and the lowest single-call
    E-value (single_e, single_log_e, its row single_row), both with the settings penalty
    applied.
    `calls` comes from add_normalised_score(), collected, with columns `setting` and
    `n_db`. Only calls with y > 0 (a positive score for this pair in their own setting)
    are used: BLAST links only segments above its cutoff, and kmerseek's chain_regions
    only positive ones."""
    c = calls.with_row_index("row").filter(pl.col("y") > 0).sort(*keys, "qs")
    if c.height == 0:
        return pl.DataFrame()
    pair_id = c.select(pl.struct(*keys).rle_id()).to_series().to_numpy()
    starts = np.flatnonzero(np.r_[True, pair_id[1:] != pair_id[:-1]])
    ends = np.r_[starts[1:], len(pair_id)]
    cols = {
        k: c[k].to_numpy() for k in ("qs", "qe", "ts", "te", "y", "n_t", "n_db", "row")
    }
    setting = c["setting"].to_numpy()
    key_vals = [c[k].to_numpy() for k in keys]
    y_all = cols["y"] - rules.settings_penalty
    log_e_single_all = -y_all + np.log(cols["n_db"] / cols["n_t"])
    rows = []
    for a, b in zip(starts, ends):
        sl = slice(a, b)
        n_t = float(cols["n_t"][a])
        n_db = cols["n_db"][sl]
        chains = best_chain(
            cols["qs"][sl],
            cols["qe"][sl],
            cols["ts"][sl],
            cols["te"][sl],
            y_all[sl],
            setting[sl],
            rules,
        )
        if not chains:
            raise ValueError(
                f"no chain for pair {[v[a] for v in key_vals]}: y = {y_all[sl]}"
            )
        best = None
        for r, y_sum, members in chains:
            # the largest database size among the members: settings differ by under 15%
            e = chain_log_evalue(y_sum, r, float(n_db[members].max()), n_t)
            if best is None or e < best[0]:
                best = (e, r, members)
        i_single = int(np.argmin(log_e_single_all[sl]))
        rws = cols["row"][sl]
        rows.append(
            {k: v[a] for k, v in zip(keys, key_vals)}
            | dict(
                n_calls=b - a,
                chain_log_e=best[0],
                chain_e=math.exp(best[0]),
                chain_r=best[1],
                chain_rows=[int(rws[m]) for m in best[2]],
                single_log_e=float(log_e_single_all[a + i_single]),
                single_e=float(np.exp(log_e_single_all[a + i_single])),
                single_row=int(rws[i_single]),
            )
        )
    return pl.DataFrame(rows, schema_overrides={"chain_rows": pl.List(pl.Int64)})
