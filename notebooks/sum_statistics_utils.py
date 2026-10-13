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
MAX_CHAIN = 8  # longest chain considered; BLAST has no cap, we never saw 8 used
_LN_FACT = [math.lgamma(r + 1) for r in range(MAX_CHAIN + 2)]


def log_sum_p(r: int, s: float) -> float:
    """ln Prob(r, s): ln of the chance that r unrelated segments' normalised scores, added,
    reach at least s (Karlin & Altschul 1993). Exact double integral, in log form so a
    small P does not underflow; above s = r^2 + r - 1 BLAST's large-s form is used, which
    agrees with the integral to under 1% there (tests/test_sum_statistics_utils.py)."""
    if r == 1:
        # Prob = 1 - exp(-e^-s)
        return math.log(-math.expm1(-math.exp(-s))) if s > -50 else 0.0
    if s >= r * r + r - 1:
        return math.log(r) + (r - 1) * math.log(s) - s - 2 * _LN_FACT[r]
    mean, sd = r * (1 - math.log(r)) - 0.5, math.sqrt(
        r
    )  # BLAST's mean and sd of the sum
    if s <= mean - 4 * sd:
        return 0.0
    # outer variable y from s to s + 60 (e^-60 of the integrand's start), inner x from 0 to
    # max(y/r, 0) + 8, past which exp(-e^(x - y/r)) < e^-2980.
    y = np.linspace(s, s + 60.0, 3001)
    hi = np.maximum(y / r, 0.0) + 8.0
    t = np.linspace(0.0, 1.0, 2001)
    x = t[None, :] * hi[:, None]
    with np.errstate(over="ignore", divide="ignore"):
        log_inner_f = (r - 2) * np.log(np.where(x > 0, x, 1e-300)) - np.exp(
            x - y[:, None] / r
        )
    if r == 2:
        log_inner_f = -np.exp(x - y[:, None] / r)
    inner = np.trapezoid(np.exp(log_inner_f), x, axis=1)
    # integrand e^-(y - s) * inner, factor e^-s taken out
    outer = np.trapezoid(np.exp(-(y - s)) * inner, y)
    const = (r - 2) * math.log(r) - _LN_FACT[r - 1] - _LN_FACT[r - 2]
    return min(0.0, const - s + math.log(outer))


def chain_evalue(y_sum: float, r: int, n_db: float, n_t: float) -> float:
    """Database E-value of r calls in the same order on both proteins whose pair-level
    normalised scores add up to y_sum (BLAST_LargeGapSumE, then the gap-decay factor
    relative to a single call). For r = 1 this is the call's own E-value."""
    if r == 1:
        return math.exp(-y_sum) * n_db / n_t
    lp = log_sum_p(r, y_sum + _LN_FACT[r])
    p = math.exp(lp)
    e_pair = -math.log1p(-p) if p < 1 else math.inf
    return e_pair * n_db / n_t / GAP_DECAY_RATE ** (r - 1)


@dataclass(frozen=True)
class ChainRules:
    """What may join a chain. Fixed in notebook 287's first cell before any run.

    max_gap        residues allowed between one call's end and the next call's start, on
                   each protein separately
    across_settings  True lets calls from different alphabets and k join one chain
    settings_penalty  ln of the number of settings searched, taken off every call's y
                   (Bonferroni: the call was the best of that many searches); 0 for none
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
        .filter(pl.col("region_evalue").is_finite() & (pl.col("region_evalue") > 0))
        .with_columns(
            qs=pl.col("region_start").cast(pl.Int64),
            qe=pl.col("region_end").cast(pl.Int64),
            ts=pl.col("target_start").cast(pl.Int64),
        )
        .with_columns(te=pl.col("ts") + pl.col("qe") - pl.col("qs"))
        .with_columns(
            y=(pl.col("n_db") / (pl.col("n_t") * pl.col("region_evalue"))).log()
        )
    )


def best_chain(qs, qe, ts, te, y, setting, rules: ChainRules):
    """Best chain on one pair. Arrays are the pair's calls with y > 0 (y already less the
    settings penalty is NOT required to be positive; the eligibility filter is the caller's).
    Returns (r, y_sum, member indices) for every chain length r that has a chain, best
    sum of y for each r. Calls must start after the previous call ends on both proteins
    (no overlap), within max_gap on each, and, when not across_settings, share a setting.
    Dynamic programme over calls in query order: best[j][r] = y_j + max best[i][r-1]."""
    n = len(qs)
    order = np.lexsort((ts, qs))
    qs, qe, ts, te, y, setting = (a[order] for a in (qs, qe, ts, te, y, setting))
    best = np.full((n, MAX_CHAIN + 1), -np.inf)
    back = np.full((n, MAX_CHAIN + 1), -1, dtype=np.int64)
    best[:, 1] = y
    for j in range(n):
        ok = (
            (qe[:j] <= qs[j])
            & (te[:j] <= ts[j])
            & (qs[j] - qe[:j] <= rules.max_gap)
            & (ts[j] - te[:j] <= rules.max_gap)
        )
        if not rules.across_settings:
            ok &= setting[:j] == setting[j]
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        for r in range(2, MAX_CHAIN + 1):
            cand = best[idx, r - 1]
            k = int(np.argmax(cand))
            if np.isfinite(cand[k]):
                best[j, r] = y[j] + cand[k]
                back[j, r] = idx[k]
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
    """One row per query-target pair: the chain with the lowest E-value (chain_e, its
    length r and members as row numbers of `calls`) and the lowest single-call E-value
    (single_e), both with the settings penalty applied. `calls` comes from
    add_normalised_score(), collected, with columns `setting` and `n_db`; only calls with
    y > 0 (a positive score for this pair in their own setting) are used, as BLAST links
    only segments above its cutoff and kmerseek's chain_regions only positive ones."""
    c = calls.with_row_index("row").filter(pl.col("y") > 0)
    rows = []
    for key, g in c.group_by(list(keys), maintain_order=True):
        n_t = float(g["n_t"][0])
        n_db = g["n_db"].to_numpy().astype(float)
        y = g["y"].to_numpy() - rules.settings_penalty
        chains = best_chain(
            g["qs"].to_numpy(),
            g["qe"].to_numpy(),
            g["ts"].to_numpy(),
            g["te"].to_numpy(),
            y,
            g["setting"].to_numpy(),
            rules,
        )
        best = None
        for r, y_sum, members in chains:
            # the largest database size among the members: settings differ by under 15%
            e = chain_evalue(y_sum, r, float(n_db[members].max()), n_t)
            if best is None or e < best[0]:
                best = (e, r, members)
        single_e = float(np.min(np.exp(-y) * n_db / n_t))
        rws = g["row"].to_numpy()
        rows.append(
            dict(zip(keys, key))
            | dict(
                n_calls=g.height,
                chain_e=best[0],
                chain_r=best[1],
                chain_rows=[int(rws[m]) for m in best[2]],
                single_e=single_e,
                single_row=int(rws[int(np.argmax(y))]),
            )
        )
    return pl.DataFrame(rows, schema_overrides={"chain_rows": pl.List(pl.Int64)})
