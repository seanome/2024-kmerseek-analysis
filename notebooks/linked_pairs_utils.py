"""Linked word pairs (strobemers) over an HP alphabet, scored on known alignments (notebook 294).

A linked pair is two short exact words, called anchors, plus the number of residues between
them, called the gap. Sahlin (2021) calls the construct a strobemer. Here both sequences of
an aligned pair are encoded in a 2-letter hydrophobic/polar alphabet and a key is shared
when both anchors match on the alignment's own diagonal (the same rule notebook 232 uses for
its exact seeds) and the gap rule of the setting holds:

* ``exact``: the query gap and the target gap are equal (the gap is part of the key);
* ``binned3``: the two gaps fall in the same 3-residue bin (gap // 3 is part of the key);
  ``binned7`` (section 4d) is the same with 7-residue bins;
* ``none``: the gap is not part of the key; both gaps only have to lie in [0, gap_max].

Two ways to choose which anchor pairs become keys:

* ``all``: every pair of anchors inside the window is a key (many keys per position);
* ``rand``: one key per position. Each sequence picks, on its own, the second anchor that
  minimises (h(w1) + h(w2)) mod p over the window (Sahlin 2021's randstrobe rule), so a
  key is shared only when both sequences pick the same aligned partner.

Each shared key is scored with notebook 232's extension (``seed_extend_utils``): +1 per
matching column, -penalty per mismatch, a gap column ends a segment. A key's score is the
best total of at most two non-overlapping gap-free segments that together contain both
anchors; when both anchors sit in one gap-free stretch that can be the one segment spanning
both. A pair's score is the best key score, and its key count is the number of shared keys.
"""

from __future__ import annotations

import numpy as np

from hp_conservation_utils import GAP, TABLES
from seed_extend_utils import best_segment_containing_runs, column_scores

ALPHABET = "hp_thomas_dill2"
PENALTY = 2.0
ANCHOR_LENGTHS = [6, 7, 8, 10]
GAP_MAXES = [8, 16, 32]
GAP_BIN = 3
TREATMENTS = ["exact", "binned3", "none"]
MODES = ["all", "rand"]
BASELINE_SEEDS = {"B1": 10, "B2": 8}
# B3: B2's 8-letter seed plus a second extended segment within this many residues
B3_SEED, B3_GAP_MAX = 8, 32
# Randstrobe rule of Sahlin (2021): partner = argmin over the window of (h(w1) + h(w2)) mod p.
RANDSTROBE_PRIME = 997
HASH_SEED = 0x5EED


def setting_name(mode: str, anchor: int, gap_max: int, treatment: str) -> str:
    return f"{mode}_a{anchor}_g{gap_max}_{treatment}"


ALL_SETTINGS = [
    (m, a, g, t)
    for m in MODES
    for a in ANCHOR_LENGTHS
    for g in GAP_MAXES
    for t in TREATMENTS
]


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser: a fixed pseudo-random hash of each word code."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64) + np.uint64(HASH_SEED) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def word_codes(cls: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer code of the a-letter HP word starting at each position of a raw (ungapped)
    class array, and whether the word is valid (no residue outside the two classes)."""
    n = cls.size
    if n < a:
        return np.empty(0, np.int64), np.empty(0, bool)
    bad = cls == GAP
    x = np.where(bad, 0, cls).astype(np.int64)
    w = np.lib.stride_tricks.sliding_window_view(x, a)
    codes = w @ (np.int64(1) << np.arange(a - 1, -1, -1, dtype=np.int64))
    ok = ~np.lib.stride_tricks.sliding_window_view(bad, a).any(axis=1)
    return codes, ok


def randstrobe_partner(cls: np.ndarray, a: int, gap_max: int) -> np.ndarray:
    """For each start i of the first anchor, the start of the second anchor chosen by the
    randstrobe rule among starts i + a .. i + a + gap_max (-1 if none is valid). Ties go to
    the leftmost start."""
    codes, ok = word_codes(cls, a)
    n = codes.size
    sel = np.full(cls.size, -1, dtype=np.int64)
    if n == 0:
        return sel
    hp = (_mix64(codes) % np.uint64(RANDSTROBE_PRIME)).astype(np.int64)
    width = gap_max + 1
    big = np.int64(1 << 40)
    pad_h = np.concatenate([hp, np.zeros(a + width, np.int64)])
    pad_ok = np.concatenate([ok, np.zeros(a + width, bool)])
    j = np.arange(n)[:, None] + a + np.arange(width)[None, :]
    score = (hp[:, None] + pad_h[j]) % RANDSTROBE_PRIME
    score = np.where(pad_ok[j], score, big)
    best = score.argmin(axis=1)
    has = score[np.arange(n), best] < big
    pick = np.where(has & ok, np.arange(n) + a + best, -1)
    sel[:n] = pick
    return sel


def _key_scores(c1, c2, a, P, left, right, gapcount, max_inner, a2=None):
    """Score of each key: a first anchor of length a at column c1 and a second anchor of
    length a2 (default a) at column c2 >= c1 + a; see module docstring."""
    a2 = a if a2 is None else a2
    s1 = a + left[c1] + right[c1 + a]
    s2 = a2 + left[c2] + right[c2 + a2]
    out = s1 + s2
    same = gapcount[c2 + a2 - 1] == gapcount[c1]
    if same.any():
        b1, b2 = c1[same], c2[same]
        e1 = b1 + a
        span = P[b2 + a2] - P[b1] + left[b1] + right[b2 + a2]
        d = np.arange(max_inner + 1)[None, :]
        m = np.minimum(e1[:, None] + d, b2[:, None])
        pm = P[m]
        lb = np.maximum.accumulate(pm, axis=1) - P[e1][:, None]
        rb = P[b2][:, None] - np.minimum.accumulate(pm[:, ::-1], axis=1)[:, ::-1]
        disjoint = (lb + rb).max(axis=1) + a + a2 + left[b1] + right[b2 + a2]
        out[same] = np.maximum(span, disjoint)
    return out


def seed_plus_second_segment(qpos, tpos, same, seed, gap_max, P, left, right, gapcount):
    """B3 control: the best total of at most two non-overlapping gap-free segments, one
    containing an exact seed of `seed` matching columns and the other containing any
    matching column within `gap_max` residues of the seed (before or after it, in both
    sequences). This is the linked-pair score with the second anchor shortened to one
    column, so it separates "a second extended segment" from "a second exact word"."""
    n = same.size
    if n < seed:
        return 0.0
    S = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(same, seed).all(axis=1))
    M1 = np.flatnonzero(same)
    if S.size == 0 or M1.size == 0:
        return 0.0
    best = 0.0
    for c_first, a_first, c_second, a_second in [(S, seed, M1, 1), (M1, 1, S, seed)]:
        Q1, T1 = qpos[c_first], tpos[c_first]
        Q2, T2 = qpos[c_second], tpos[c_second]
        dq = Q2[None, :] - Q1[:, None] - a_first
        dt = T2[None, :] - T1[:, None] - a_first
        ok = (dq >= 0) & (dt >= 0) & (dq <= gap_max) & (dt <= gap_max)
        i1, i2 = np.nonzero(ok)
        if i1.size:
            sc = _key_scores(
                c_first[i1],
                c_second[i2],
                a_first,
                P,
                left,
                right,
                gapcount,
                gap_max,
                a2=a_second,
            )
            best = max(best, float(sc.max()))
    return best


def _residue_maps(aln_bytes: np.ndarray):
    """Per column: residue index (-1 at '-'); per residue: its column."""
    isres = aln_bytes != ord("-")
    pos = np.where(isres, np.cumsum(isres) - 1, -1)
    col_of = np.flatnonzero(isres)
    return isres, pos, col_of


def score_pair(
    qaln: str, taln: str, rng: np.random.Generator, n_shuffle: int = 2
) -> list[dict]:
    """One row per comparison (the real pair and n_shuffle shuffled targets), with every
    linked-pair setting's extended score and key count, plus the B1/B2 baselines.

    The shuffles reproduce notebook 232 exactly: the same rng consumed in the same order
    (hp_thomas_dill2 was the first alphabet 232 scored)."""
    tab = TABLES[ALPHABET]
    qb = np.frombuffer(qaln.encode(), dtype=np.uint8)
    tb = np.frombuffer(taln.encode(), dtype=np.uint8)
    qa, ta = tab[qb], tab[tb]
    t_res_idx = np.flatnonzero(ta != GAP)
    targets = [("real", ta)]
    for i in range(n_shuffle):
        ts = ta.copy()
        ts[t_res_idx] = ta[rng.permutation(t_res_idx)]
        targets.append((f"null{i}", ts))

    q_isres, qpos, qcol_of = _residue_maps(qb)
    t_isres, tpos, tcol_of = _residue_maps(tb)
    q_raw = qa[q_isres]
    q_sel = {
        (a, g): randstrobe_partner(q_raw, a, g)
        for a in ANCHOR_LENGTHS
        for g in GAP_MAXES
    }

    rows = []
    for kind, tt in targets:
        sc, same = column_scores(qa, tt, PENALTY)
        both = (qa != GAP) & (tt != GAP)
        P = np.concatenate(([0.0], np.cumsum(sc)))
        left = P - np.minimum.accumulate(P)
        right = np.maximum.accumulate(P[::-1])[::-1] - P
        gapcount = np.cumsum(~both)
        row = {"kind": kind}
        for name, s in BASELINE_SEEDS.items():
            row[f"{name}_ext"] = best_segment_containing_runs(sc, same, s)
        row["B3_ext"] = seed_plus_second_segment(
            qpos, tpos, same, B3_SEED, B3_GAP_MAX, P, left, right, gapcount
        )
        t_raw = tt[t_isres]
        for a in ANCHOR_LENGTHS:
            if same.size >= a:
                M = np.flatnonzero(
                    np.lib.stride_tricks.sliding_window_view(same, a).all(axis=1)
                )
            else:
                M = np.empty(0, np.int64)
            row[f"seed{a}_count"] = M.size
            is_anchor = np.zeros(same.size + 1, bool)
            is_anchor[M] = True
            Q, T = qpos[M], tpos[M]
            dq = Q[None, :] - Q[:, None] - a
            dt = T[None, :] - T[:, None] - a
            gmax = max(GAP_MAXES)
            ok = (dq >= 0) & (dt >= 0) & (dq <= gmax) & (dt <= gmax)
            i1, i2 = np.nonzero(ok)
            c1, c2 = M[i1], M[i2]
            kq, kt = dq[i1, i2], dt[i1, i2]
            sc_all = (
                _key_scores(c1, c2, a, P, left, right, gapcount, gmax)
                if c1.size
                else np.empty(0)
            )
            for g in GAP_MAXES:
                inwin = (kq <= g) & (kt <= g)
                for tr in TREATMENTS:
                    keep = inwin & _gap_rule(kq, kt, tr)
                    nm = setting_name("all", a, g, tr)
                    row[f"{nm}_ext"] = float(sc_all[keep].max()) if keep.any() else 0.0
                    row[f"{nm}_count"] = int(keep.sum())
            # randstrobe: one key per first-anchor position, partner picked per sequence
            t_sel = {g: randstrobe_partner(t_raw, a, g) for g in GAP_MAXES}
            for g in GAP_MAXES:
                rq = q_sel[(a, g)][Q] if M.size else np.empty(0, np.int64)
                rt = t_sel[g][T] if M.size else np.empty(0, np.int64)
                valid = (rq >= 0) & (rt >= 0)
                c2q = np.where(valid, qcol_of[np.clip(rq, 0, None)], -1)
                c2t = np.where(valid, tcol_of[np.clip(rt, 0, None)], -2)
                on_diag = valid & (c2q == c2t)
                on_diag &= is_anchor[np.clip(c2q, 0, same.size)]
                k1, k2 = M[on_diag], c2q[on_diag]
                gq = rq[on_diag] - Q[on_diag] - a
                gt = rt[on_diag] - T[on_diag] - a
                ks = (
                    _key_scores(k1, k2, a, P, left, right, gapcount, g)
                    if k1.size
                    else np.empty(0)
                )
                for tr in TREATMENTS:
                    keep = _gap_rule(gq, gt, tr)
                    nm = setting_name("rand", a, g, tr)
                    row[f"{nm}_ext"] = float(ks[keep].max()) if keep.any() else 0.0
                    row[f"{nm}_count"] = int(keep.sum())
        rows.append(row)
    return rows


def _gap_rule(gq: np.ndarray, gt: np.ndarray, treatment: str) -> np.ndarray:
    """Whether a key's query gap gq and target gap gt agree under the gap rule:
    ``exact`` (equal), ``binned<w>`` (same bin, gap // w; binned3 and binned7 are used) or
    ``none`` (always)."""
    if treatment == "exact":
        return gq == gt
    if treatment.startswith("binned"):
        w = int(treatment[len("binned") :])
        return (gq // w) == (gt // w)
    if treatment == "none":
        return np.ones(gq.shape, bool)
    raise ValueError(f"unknown gap rule {treatment!r}")


# ---------------------------------------------------------------------------
# Section 4d: key designs compared at their chance cost.
# ---------------------------------------------------------------------------

# Gap 0..32 in bins of 7 residues: 0-6, 7-13, 14-20, 21-27, 28-32 (gap // 7). Two gaps in
# one bin differ by at most 6, so the key tolerates an indel of up to 6 residues.
GAP_BIN7 = 7
DESIGN_GAP_MAX = 32
DESIGN_ANCHORS = [6, 7, 8]
SINGLE_WORDS = [8, 10, 12, 14, 16]
# (mode, length, gap_max, gap rule). "word" is one k-letter word with extension; k = 8
# and 10 are B2 and B1. The last two are the chosen key and the randstrobe of section 4b.
DESIGNS = (
    [("word", k, 0, "single") for k in SINGLE_WORDS]
    + [("all", a, DESIGN_GAP_MAX, "exact") for a in DESIGN_ANCHORS]
    + [("all", a, DESIGN_GAP_MAX, f"binned{GAP_BIN7}") for a in DESIGN_ANCHORS]
    + [("all", 6, DESIGN_GAP_MAX, "none"), ("rand", 6, DESIGN_GAP_MAX, "none")]
)


def design_name(mode: str, length: int, gap_max: int, treatment: str) -> str:
    if mode == "word":
        return f"word{length}"
    return setting_name(mode, length, gap_max, treatment)


def _shuffled_targets(ta: np.ndarray, rng: np.random.Generator, n_shuffle: int):
    """The real target row and n_shuffle shuffles of its residues (gaps stay put), drawn
    from rng in the order notebook 232 drew them."""
    t_res_idx = np.flatnonzero(ta != GAP)
    targets = [("real", ta)]
    for i in range(n_shuffle):
        ts = ta.copy()
        ts[t_res_idx] = ta[rng.permutation(t_res_idx)]
        targets.append((f"null{i}", ts))
    return targets


def _extension_arrays(sc: np.ndarray, both: np.ndarray):
    """Prefix sums, best left and right extension from each column, and the running count
    of gap columns, as score_pair builds them for _key_scores."""
    P = np.concatenate(([0.0], np.cumsum(sc)))
    left = P - np.minimum.accumulate(P)
    right = np.maximum.accumulate(P[::-1])[::-1] - P
    return P, left, right, np.cumsum(~both)


def longest_match_run(same: np.ndarray) -> int:
    """Length of the longest run of aligned columns whose HP classes match."""
    padded = np.concatenate(([False], same, [False])).astype(np.int8)
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max()) if starts.size else 0


def score_pair_designs(
    qaln: str, taln: str, rng: np.random.Generator, n_shuffle: int = 2
) -> list[dict]:
    """Section 4d's extra scores, one row per comparison, on the same shuffles as
    score_pair (same rng, same order):

    * ``word<k>_ext``: one k-letter seed with extension (best_segment_containing_runs);
    * ``all_a<a>_g32_<rule>_ext`` and ``_count`` for every anchor pair up to 32 apart with
      the gap stored exactly or in 7-residue bins, scored with _key_scores;
    * ``longest_run``: the longest run of matching HP classes on the alignment, so a pair
      carries an exact k-letter match when longest_run >= k.
    """
    tab = TABLES[ALPHABET]
    qb = np.frombuffer(qaln.encode(), dtype=np.uint8)
    tb = np.frombuffer(taln.encode(), dtype=np.uint8)
    qa, ta = tab[qb], tab[tb]
    targets = _shuffled_targets(ta, rng, n_shuffle)
    _, qpos, _ = _residue_maps(qb)
    _, tpos, _ = _residue_maps(tb)
    G = DESIGN_GAP_MAX
    rows = []
    for kind, tt in targets:
        sc, same = column_scores(qa, tt, PENALTY)
        both = (qa != GAP) & (tt != GAP)
        P, left, right, gapcount = _extension_arrays(sc, both)
        row = {"kind": kind, "longest_run": longest_match_run(same)}
        for k in SINGLE_WORDS:
            row[f"word{k}_ext"] = best_segment_containing_runs(sc, same, k)
        for a in DESIGN_ANCHORS:
            if same.size >= a:
                M = np.flatnonzero(
                    np.lib.stride_tricks.sliding_window_view(same, a).all(axis=1)
                )
            else:
                M = np.empty(0, np.int64)
            Q, T = qpos[M], tpos[M]
            dq = Q[None, :] - Q[:, None] - a
            dt = T[None, :] - T[:, None] - a
            ok = (dq >= 0) & (dt >= 0) & (dq <= G) & (dt <= G)
            i1, i2 = np.nonzero(ok)
            c1, c2 = M[i1], M[i2]
            kq, kt = dq[i1, i2], dt[i1, i2]
            sc_all = (
                _key_scores(c1, c2, a, P, left, right, gapcount, G)
                if c1.size
                else np.empty(0)
            )
            for tr in ["exact", f"binned{GAP_BIN7}"]:
                keep = _gap_rule(kq, kt, tr)
                nm = setting_name("all", a, G, tr)
                row[f"{nm}_ext"] = float(sc_all[keep].max()) if keep.any() else 0.0
                row[f"{nm}_count"] = int(keep.sum())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Recall at a null false-positive rate, with notebook 232's threshold rule, and a
# pair-level bootstrap.
# ---------------------------------------------------------------------------

BINS = ["<20%", "20-30%", "30-40%", "40-60%", ">=60%"]


def score_arrays(df, cols: list[str]):
    """Real scores (pairs x cols), null scores (pairs x n_null x cols) and identity-bin
    index per pair, from the long table written by scripts/linked_pairs_on_alignments.py
    (one row per comparison, an ``identity_bin`` column added)."""
    import polars as pl

    d = df.sort("pair_index", "kind")
    kinds = d["kind"].unique().sort().to_list()  # null0, null1, real
    n_null = len(kinds) - 1
    pairs = d.filter(pl.col("kind") == "real")
    real = pairs.select(cols).to_numpy().astype(float)
    null = np.stack(
        [d.filter(pl.col("kind") == k).select(cols).to_numpy() for k in kinds[:n_null]],
        axis=1,
    ).astype(float)
    bin_idx = np.array([BINS.index(b) for b in pairs["identity_bin"].cast(str)])
    return real, null, bin_idx, pairs["pair_index"].to_numpy()


def threshold_and_recall(real, null, bin_idx, fpr: float = 1e-3):
    """Notebook 232's rule (seed_extend_utils.recall_at_null_quantile): the threshold is the
    (1 - fpr) quantile of the pooled null scores, taking the higher of two neighbouring
    values, and recall is the share of real pairs scoring strictly above it.

    Returns threshold (cols), achieved null rate (cols), recall (bins x cols)."""
    flat = null.reshape(-1, null.shape[-1])
    thr = np.quantile(flat, 1 - fpr, axis=0, method="higher")
    null_rate = (flat > thr).mean(axis=0)
    above = real > thr
    rec = np.stack([above[bin_idx == b].mean(axis=0) for b in range(len(BINS))])
    return thr, null_rate, rec


def bootstrap_recall(real, null, bin_idx, fpr=1e-3, n_boot=1000, seed=0):
    """Resample pairs with replacement (each real pair travels with its own shuffled
    targets), recompute the threshold and the recall per bin. Returns (n_boot, bins, cols).
    """
    rng = np.random.default_rng(seed)
    n = real.shape[0]
    out = np.empty((n_boot, len(BINS), real.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        _, _, out[i] = threshold_and_recall(real[idx], null[idx], bin_idx[idx], fpr)
    return out


def best_key(qaln: str, taln: str, mode: str, a: int, gap_max: int, treatment: str):
    """The highest-scoring shared key of one real pair under one setting, for drawing:
    dict with the two anchor columns, both gaps (in residues) and the key score."""
    tab = TABLES[ALPHABET]
    qb = np.frombuffer(qaln.encode(), dtype=np.uint8)
    tb = np.frombuffer(taln.encode(), dtype=np.uint8)
    qa, ta = tab[qb], tab[tb]
    sc, same = column_scores(qa, ta, PENALTY)
    both = (qa != GAP) & (ta != GAP)
    P = np.concatenate(([0.0], np.cumsum(sc)))
    left = P - np.minimum.accumulate(P)
    right = np.maximum.accumulate(P[::-1])[::-1] - P
    gapcount = np.cumsum(~both)
    _, qpos, qcol_of = _residue_maps(qb)
    t_isres, tpos, tcol_of = _residue_maps(tb)
    q_isres = qb != ord("-")
    M = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(same, a).all(axis=1))
    Q, T = qpos[M], tpos[M]
    if mode == "all":
        dq = Q[None, :] - Q[:, None] - a
        dt = T[None, :] - T[:, None] - a
        ok = (dq >= 0) & (dt >= 0) & (dq <= gap_max) & (dt <= gap_max)
        i1, i2 = np.nonzero(ok)
        c1, c2, gq, gt = M[i1], M[i2], dq[i1, i2], dt[i1, i2]
    else:
        rq = randstrobe_partner(qa[q_isres], a, gap_max)[Q]
        rt = randstrobe_partner(ta[t_isres], a, gap_max)[T]
        valid = (rq >= 0) & (rt >= 0)
        c2q = np.where(valid, qcol_of[np.clip(rq, 0, None)], -1)
        c2t = np.where(valid, tcol_of[np.clip(rt, 0, None)], -2)
        is_anchor = np.zeros(same.size + 1, bool)
        is_anchor[M] = True
        keep = valid & (c2q == c2t) & is_anchor[np.clip(c2q, 0, same.size)]
        c1, c2 = M[keep], c2q[keep]
        gq, gt = rq[keep] - Q[keep] - a, rt[keep] - T[keep] - a
    keep = _gap_rule(gq, gt, treatment)
    c1, c2, gq, gt = c1[keep], c2[keep], gq[keep], gt[keep]
    if c1.size == 0:
        return None
    s = _key_scores(c1, c2, a, P, left, right, gapcount, gap_max)
    k = int(np.argmax(s))
    return {
        "c1": int(c1[k]),
        "c2": int(c2[k]),
        "gap_query": int(gq[k]),
        "gap_target": int(gt[k]),
        "score": float(s[k]),
        "n_keys": int(c1.size),
    }


def recall_at_exact_fpr(real, null, bin_idx, fpr: float = 1e-3):
    """Recall at exactly `fpr`, read off the line between the two thresholds on either side
    of it. Scores are whole numbers, so "strictly above t" (notebook 232's rule, null rate
    at or below fpr) and "at or above t" (null rate above fpr) bracket the target; recall is
    interpolated by where fpr falls between those two null rates. This equals breaking ties
    at t at random so that exactly fpr of the null is called.

    A NaN null score (a null target that could not be built, section 4f) is left out: the
    threshold and the null rates are taken over the remaining null comparisons. Without
    NaNs the result is the same as before this rule was added.

    Returns threshold t (cols), weight of the "at or above" side (cols), recall (bins x cols).
    """
    flat = null.reshape(-1, null.shape[-1])
    if np.isnan(flat).any():
        n_valid = (~np.isnan(flat)).sum(axis=0)
        thr = np.nanquantile(flat, 1 - fpr, axis=0, method="higher")
        f_hi = (flat > thr).sum(axis=0) / n_valid
        f_lo = (flat >= thr).sum(axis=0) / n_valid
    else:
        thr = np.quantile(flat, 1 - fpr, axis=0, method="higher")
        f_hi = (flat > thr).mean(axis=0)
        f_lo = (flat >= thr).mean(axis=0)
    w = np.where(f_lo > f_hi, (fpr - f_hi) / np.where(f_lo > f_hi, f_lo - f_hi, 1), 0.0)
    above, at_or_above = real > thr, real >= thr
    rec = np.stack(
        [
            above[bin_idx == b].mean(axis=0)
            + w
            * (
                at_or_above[bin_idx == b].mean(axis=0)
                - above[bin_idx == b].mean(axis=0)
            )
            for b in range(len(BINS))
        ]
    )
    return thr, w, rec


def bootstrap_recall_by_group(
    real, null, bin_idx, groups, fpr=1e-3, n_boot=1000, seed=0, exact=True
):
    """Bootstrap that resamples whole groups (Pfam families) with replacement, each true
    pair travelling with its own shuffled targets, and recomputes threshold and recall.
    Pairs from one family share sequence, so resampling pairs one by one would make the
    interval too narrow. Returns (n_boot, bins, cols)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    counts = np.diff(np.r_[starts, g_sorted.size])
    n_groups = starts.size
    fn = recall_at_exact_fpr if exact else threshold_and_recall
    out = np.empty((n_boot, len(BINS), real.shape[1]))
    for i in range(n_boot):
        g = rng.integers(0, n_groups, n_groups)
        c = counts[g]
        offs = np.repeat(starts[g] - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(
            c.sum()
        )
        idx = order[offs]
        out[i] = fn(real[idx], null[idx], bin_idx[idx], fpr)[2]
    return out


# ---------------------------------------------------------------------------
# Section 4e: the seed ceiling and the scoring ceiling.
# ---------------------------------------------------------------------------

# Joint HP class of an aligned column, (query class, target class), class 0 = H
# (ACFILMVWY) and 1 = P (DEGHKNPQRST) in hp_thomas_dill2. NOT_SCORED where either
# sequence has a gap or a letter outside the two classes (X): such a column is skipped.
JOINT_LABELS = ["HH", "HP", "PH", "PP"]
NOT_SCORED = -1


def joint_codes(qa: np.ndarray, ta: np.ndarray) -> np.ndarray:
    """Per aligned column: 2 * query class + target class, or NOT_SCORED."""
    ok = (qa != GAP) & (ta != GAP)
    code = 2 * qa.astype(np.int16) + ta.astype(np.int16)
    return np.where(ok, code, NOT_SCORED).astype(np.int8)


def comparison_joint_codes(
    qaln: str, taln: str, rng: np.random.Generator, n_shuffle: int = 2
) -> list[tuple[str, np.ndarray]]:
    """Joint class codes of the real pair and of its n_shuffle shuffled targets, drawn from
    rng in notebook 232's order (the shuffles score_pair and score_pair_designs use)."""
    tab = TABLES[ALPHABET]
    qa = tab[np.frombuffer(qaln.encode(), dtype=np.uint8)]
    ta = tab[np.frombuffer(taln.encode(), dtype=np.uint8)]
    return [
        (kind, joint_codes(qa, tt))
        for kind, tt in _shuffled_targets(ta, rng, n_shuffle)
    ]


def joint_counts(codes: np.ndarray, comp: np.ndarray, n_comp: int) -> np.ndarray:
    """(n_comp, 4) counts of HH, HP, PH, PP columns per comparison. codes is one long
    array of joint codes, and comp gives the comparison each column belongs to."""
    ok = codes != NOT_SCORED
    flat = comp[ok].astype(np.int64) * 4 + codes[ok]
    return np.bincount(flat, minlength=4 * n_comp).reshape(n_comp, 4)


def logodds_matrix(counts: np.ndarray) -> dict:
    """2x2 log-odds scores in bits from pooled joint counts (HH, HP, PH, PP).

    Pr(a, b) is made symmetric (the HP and PH shares averaged), so a score does not depend
    on which protein is the query. share(a) is the class share over the same columns, both
    sequences pooled. s(a, b) = log2(Pr(a, b) / (share(a) * share(b))).

    Returns the raw and the symmetric joint probabilities, the two shares, the score of
    each joint code (HH, HP, PH, PP), the relative entropy sum Pr(a, b) * s(a, b) (the
    mean score per column of the columns the matrix was fitted on) and
    sum share(a) * share(b) * s(a, b) (the mean score per column of two unrelated
    sequences with these shares)."""
    c = np.asarray(counts, dtype=float).reshape(4)
    pr = c / c.sum()
    sym = pr.copy()
    sym[1] = sym[2] = (pr[1] + pr[2]) / 2
    share_h = sym[0] + sym[1]
    share = np.array([share_h, 1 - share_h])
    indep = np.outer(share, share).reshape(4)
    s = np.log2(sym / indep)
    return {
        "pr": pr,
        "pr_sym": sym,
        "share_h": float(share[0]),
        "share_p": float(share[1]),
        "score": s,
        "relative_entropy": float((sym * s).sum()),
        "expected_unrelated": float((indep * s).sum()),
    }


def best_stretch_scores(
    codes: np.ndarray, comp: np.ndarray, n_comp: int, score: np.ndarray
) -> np.ndarray:
    """Per comparison, the highest total score of a run of consecutive scored columns (no
    gap or X column inside it, any start and end), or 0 if no column scores above 0: a
    Smith-Waterman on the fixed alignment with no gaps allowed. codes and comp as in
    joint_counts, each comparison's columns contiguous and in alignment order."""
    import polars as pl

    ok = codes != NOT_SCORED
    # a new stretch starts at every unscored column and at every new comparison
    new_comp = np.r_[True, comp[1:] != comp[:-1]]
    seg = np.cumsum(new_comp | ~ok)
    sc = np.asarray(score, dtype=float)[np.clip(codes, 0, 3)]
    df = pl.DataFrame({"comp": comp[ok], "seg": seg[ok], "sc": sc[ok]})
    best = (
        df.with_columns(pl.col("sc").cum_sum().over("seg").alias("P"))
        .with_columns(
            pl.min_horizontal(pl.col("P").cum_min().over("seg"), pl.lit(0.0)).alias("M")
        )
        .group_by("comp")
        .agg((pl.col("P") - pl.col("M")).max().alias("best"))
    )
    out = np.zeros(n_comp)
    out[best["comp"].to_numpy()] = np.maximum(best["best"].to_numpy(), 0.0)
    return out


# ---------------------------------------------------------------------------
# Section 4f: a null that keeps local HP patterns (an unrelated real protein), and the
# report half used for kmerseek's own chaining.
# ---------------------------------------------------------------------------

SPLIT_SEED = 292  # the seed of the selection/report split, kept from the first run


def select_half_ids(pair_index: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """Pair indices of the selection half, as notebook 294's section 2 draws them: a
    permutation of the unique pair indices with default_rng(seed), first half, sorted.
    Every other pair is in the report half."""
    perm = np.random.default_rng(seed).permutation(np.unique(pair_index))
    return np.sort(perm[: perm.size // 2])


def pfam_clans(seed_path) -> dict[str, str]:
    """Pfam family accession (no version, e.g. ``PF00244``) -> its clan (``CL0020``),
    read from the ``#=GF AC`` and ``#=GF CL`` lines of Pfam-A.seed(.gz). A family with no
    clan counts as its own clan, so its value is its own accession."""
    import gzip

    opener = gzip.open if str(seed_path).endswith(".gz") else open
    out: dict[str, str] = {}
    acc = clan = None
    with opener(seed_path, "rb") as fh:
        for line in fh:
            if line.startswith(b"#=GF AC"):
                acc = line.split()[2].decode().split(".")[0]
            elif line.startswith(b"#=GF CL"):
                clan = line.split()[2].decode()
            elif line.startswith(b"//"):
                if acc is not None:
                    out[acc] = clan or acc
                acc = clan = None
    return out


def unrelated_candidates(aln_df, clans: dict[str, str]):
    """Every distinct seed sequence of the alignment table (query and target rows, gaps
    stripped), with its family's clan, sorted by length then name so draws are
    reproducible. Returns (names, sequences, clans, lengths) as numpy arrays."""
    import polars as pl

    rows = pl.concat(
        [
            aln_df.select("family", pl.col(n).alias("name"), pl.col(a).alias("aln"))
            for n, a in [("query", "qaln"), ("target", "taln")]
        ]
    )
    cand = (
        rows.unique(["family", "name"])
        .with_columns(pl.col("aln").str.replace_all("-", "", literal=True).alias("seq"))
        .with_columns(
            pl.col("seq").str.len_bytes().alias("length"),
            pl.col("family").replace_strict(clans).alias("clan"),
        )
        .sort("length", "name", "family")
    )
    return (
        cand["name"].to_numpy(),
        cand["seq"].to_numpy(),
        cand["clan"].to_numpy(),
        cand["length"].to_numpy(),
    )


def draw_unrelated(
    rng: np.random.Generator,
    length: int,
    clan: str,
    cand_clans: np.ndarray,
    cand_lengths: np.ndarray,
    max_tries: int = 10_000,
):
    """Pick a candidate from another clan at least ``length`` residues long, uniformly
    among those, and a uniform start offset into it. Returns (candidate index, offset), or
    (None, reason) when no candidate qualifies: ``"longer than any candidate"`` or
    ``"no other-clan candidate long enough"``. cand_lengths must be sorted ascending."""
    start = int(np.searchsorted(cand_lengths, length, side="left"))
    n = cand_lengths.size
    if start == n:
        return None, "longer than any candidate"
    for _ in range(max_tries):
        j = int(rng.integers(start, n))
        if cand_clans[j] != clan:
            return j, int(rng.integers(0, cand_lengths[j] - length + 1))
    other = np.flatnonzero(cand_clans[start:] != clan)
    if other.size == 0:
        return None, "no other-clan candidate long enough"
    j = start + int(other[rng.integers(0, other.size)])
    return j, int(rng.integers(0, cand_lengths[j] - length + 1))


def place_in_gaps(taln: str, residues: str) -> str:
    """The target's alignment row with its scored residues replaced, in order, by
    ``residues`` (as many letters as the row has columns with an H or P residue). Gap
    columns, and columns holding X or another letter outside the two classes (which
    score_pair treats as a gap), stay put, so the new row has the true target's gaps."""
    row = np.frombuffer(taln.encode(), dtype=np.uint8).copy()
    res = np.flatnonzero(TABLES[ALPHABET][row] != GAP)
    new = np.frombuffer(residues.encode(), dtype=np.uint8)
    if new.size != res.size:
        raise ValueError(f"{new.size} residues for {res.size} residue columns")
    row[res] = new
    return row.tobytes().decode()


def shuffled_target_rows(taln: str, seed: int, n_shuffle: int = 2) -> list[str]:
    """Notebook 232's shuffled targets as residue rows: the same permutations that
    score_pair applies to the target's HP classes (default_rng(seed), drawn in the same
    order), applied to its residues instead. Gaps stay put."""
    rng = np.random.default_rng(seed)
    tb = np.frombuffer(taln.encode(), dtype=np.uint8)
    t_res_idx = np.flatnonzero(TABLES[ALPHABET][tb] != GAP)
    out = []
    for _ in range(n_shuffle):
        ts = tb.copy()
        ts[t_res_idx] = tb[rng.permutation(t_res_idx)]
        out.append(ts.tobytes().decode())
    return out


# kmerseek v0.4.0 `search` CSV columns, in order: the fields of SearchResultCsv in
# src/rust/search.rs at tag v0.4.0 (serde writes the field names as the header).
KMERSEEK_040_CSV_COLUMNS = [
    "query_name", "query_md5", "target_name", "target_md5", "containment",
    "n_intersecting_hashes", "ksize", "scaled", "moltype", "remove_low_complexity",
    "jaccard", "max_containment", "average_abund", "median_abund", "std_abund",
    "containment_target_in_query", "f_weighted_target_in_query", "query_tfidf",
    "mean_matched_kmer_freq", "sum_matched_kmer_freq", "query_expected_shared_kmers",
    "query_enrichment", "joint_kmer_freq", "query_poisson_pvalue", "region_search_space",
    "db_n_targets", "db_n_kmers", "run_n_queries", "region_start", "region_end",
    "region_subseq", "target_start", "target_end", "target_subseq", "moltype_seq",
    "region_length", "region_n_shared_kmers", "region_expected_shared_kmers",
    "region_poisson_score", "region_tail_probability", "region_enrichment", "region_tfidf",
    "region_mean_idf", "region_n_mismatches", "region_ka_bits", "region_evalue",
    "region_n_chained",
]  # fmt: skip


def read_kmerseek_best_evalue(path):
    """Best (smallest) region_evalue per (query_name, target_name) in a kmerseek search CSV
    (gzipped or not), with the number of regions chained into that best row. Checks the
    header against kmerseek v0.4.0's column layout."""
    import gzip

    import polars as pl

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
    if header != KMERSEEK_040_CSV_COLUMNS:
        raise ValueError(f"{path}: header is not kmerseek v0.4.0's search layout")
    d = pl.read_csv(
        path,
        columns=["query_name", "target_name", "region_evalue", "region_n_chained"],
        schema_overrides={"region_evalue": pl.Float64, "region_n_chained": pl.Int64},
    )
    return (
        d.sort("query_name", "target_name", "region_evalue", "region_n_chained")
        .group_by("query_name", "target_name", maintain_order=True)
        .first()
    )


def evalue_cutoff_recall(
    own_e, null_pair, null_e, share, n_targets, fpr=1e-3, mult=None
):
    """Recall at a database-scale false-positive rate, from E-values.

    own_e (pairs): each pair's best E-value of its query against its own target (inf if
    the search reported no row). null_pair, null_e: one entry per reported (pair, shuffled
    target) comparison, the pair being the one whose query was searched, sorted by null_e
    ascending. share (pairs): the number of pairs with that pair's query, so a query
    searched once counts once. A null comparison not reported counts as not passing; there
    are n_targets null comparisons per query.

    The cutoff is the largest null E-value at which the weighted share of null comparisons
    with E at or below it is at most fpr. A pair is found when its own E is at or below the
    cutoff. mult (pairs): bootstrap multiplicity of each pair, default 1.

    Raises ValueError when every reported null comparison fits under fpr, because the
    unreported ones (above the run's E filter) could then hold the cutoff.

    Returns cutoff, achieved false-positive rate, recall."""
    own_e = np.asarray(own_e, float)
    null_e = np.asarray(null_e, float)
    if null_e.size == 0:
        raise ValueError("no reported null comparisons")
    if (np.diff(null_e) < 0).any():
        raise ValueError("null_e must be sorted ascending")
    m = np.ones(own_e.size) if mult is None else np.asarray(mult, float)
    per_query = m / np.asarray(share, float)
    n_null = per_query.sum() * n_targets
    cw = np.cumsum(per_query[null_pair])
    last = np.r_[null_e[1:] != null_e[:-1], True]  # last entry of each distinct E-value
    ok = np.flatnonzero(last & (cw <= fpr * n_null))
    if ok.size and ok[-1] == null_e.size - 1:
        raise ValueError(
            "the reported null comparisons run out before the false-positive rate is reached"
        )
    if ok.size:
        cutoff, achieved = float(null_e[ok[-1]]), float(cw[ok[-1]] / n_null)
    else:  # even the smallest null E-value is too frequent: only E below it passes
        cutoff, achieved = float(np.nextafter(null_e[0], -np.inf)), 0.0
    recall = float((m * (own_e <= cutoff)).sum() / m.sum())
    return cutoff, achieved, recall


def group_multiplicities(groups, n_boot, seed):
    """Yield, n_boot times, how often each item is drawn when whole groups are resampled
    with replacement (the family bootstrap of section 4b, as counts per item)."""
    groups = np.asarray(groups)
    codes, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        g = rng.integers(0, codes.size, codes.size)
        yield np.bincount(g, minlength=codes.size)[inv]


# ---------------------------------------------------------------------------
# Notebook 346: the same four scores on every diagonal of two whole proteins, with no
# alignment given (a database search).
#
# B2, B3, the chosen linked pair (every pair of 6-letter anchors whose starts are at most
# 32 apart, gap not stored) and its randstrobe are scored as score_pair scores them on an
# alignment, with one change forced by the missing alignment: the two anchors of a key may
# sit on different diagonals (the gap is not part of the key, so the query gap and the
# target gap may differ). A key's two extended segments then may not overlap in either
# protein, the rule a gapped alignment with one indel between them would follow. When an
# allowed-cell mask is given (the cells of a known alignment), every other cell is a wall
# and the scores are score_pair's on that alignment (gates 1 and 2 of notebook 346).
#
# Coordinates: cell (i, j) pairs query residue i with target residue j; the diagonal of a
# cell is j - i. A boundary (i, j) sits just before cell (i, j) on its diagonal.
# ---------------------------------------------------------------------------

DB_WALL = -100_000  # one wall outweighs any run of matches between two proteins
DB_ANCHOR = 6  # anchor length of the chosen linked pair (all_a6_g32_none)
DB_WINDOW = 32  # gap_max of the chosen linked pair, and B3's window
DB_SEED = BASELINE_SEEDS["B2"]  # 8: B2's seed, and B3's
DB_PAD = DB_WINDOW + DB_SEED + 2  # wall border, so no window reads past the matrix
DB_SCORES = ["B2", "linked", "randstrobe", "B3"]
_NEG = np.int64(-(1 << 40))  # "no key here", below any real score


def _db_score_matrix(q: np.ndarray, t: np.ndarray, allowed=None) -> np.ndarray:
    """Padded int32 score matrix: +1 where the HP classes match, -PENALTY where they differ,
    DB_WALL where either residue is outside the two classes, outside an allowed mask, or in
    the DB_PAD border."""
    nq, nt = q.size, t.size
    sc = np.full((nq + 2 * DB_PAD, nt + 2 * DB_PAD), DB_WALL, np.int32)
    valid = (q != GAP)[:, None] & (t != GAP)[None, :]
    if allowed is not None:
        valid &= allowed
    inner = np.where(q[:, None] == t[None, :], 1, -int(PENALTY)).astype(np.int32)
    sc[DB_PAD : DB_PAD + nq, DB_PAD : DB_PAD + nt] = np.where(valid, inner, DB_WALL)
    return sc


def _diagonal_extensions(sc: np.ndarray):
    """Best leftward and rightward extension from every boundary, along its diagonal:
    left[i, j] = best total of cells (i-y, j-y) .. (i-1, j-1), y >= 0, and right[i, j] =
    best total of cells (i, j) .. (i+y-1, j+y-1); both 0 at least (the empty extension).
    Both arrays have shape (R + 1, C + 1) for an R x C matrix. The diagonals are laid out
    as rows through strided views, so numpy's row-wise cumsum and running min/max apply.
    """
    from numpy.lib.stride_tricks import as_strided

    R, C = sc.shape
    if R * -DB_WALL >= 2**31:
        raise ValueError("query too long for int32 prefix sums")
    n_diag = R + C - 1
    w = C + 2 * (R - 1)
    buf = np.full((R, w), DB_WALL, np.int32)
    buf[:, R - 1 : R - 1 + C] = sc
    s = buf.itemsize
    sheared = as_strided(buf, shape=(n_diag, R), strides=(s, (w + 1) * s))
    P = np.zeros((n_diag, R + 1), np.int32)
    np.cumsum(sheared, axis=1, out=P[:, 1:])
    out = []
    for ext in (
        P - np.minimum.accumulate(P, axis=1),
        np.maximum.accumulate(P[:, ::-1], axis=1)[:, ::-1] - P,
    ):
        bb = np.zeros((R + 1, w + 1), np.int32)
        as_strided(bb, shape=(n_diag, R + 1), strides=(s, (w + 2) * s))[...] = ext
        out.append(bb[:, R - 1 : R + C])
    return out[0], out[1]


def _diagonal_windows(match: np.ndarray, a: int) -> np.ndarray:
    """True at (i, j) when cells (i, j) .. (i+a-1, j+a-1) all match: an exact shared
    a-letter word starting at query residue i and target residue j."""
    out = match.copy()
    R, C = match.shape
    for y in range(1, a):
        out[: R - y, : C - y] &= match[y:, y:]
        out[R - y :, :] = False
        out[:, C - y :] = False
    return out


def _forward_max(x: np.ndarray, w: int, axis: int) -> np.ndarray:
    """out[i] = max of x[i .. i+w-1] along axis (positions past the end count as _NEG),
    from running maxima within blocks of w (van Herk 1992; Gil and Werman 1993)."""
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]
    nb = -(-(n + w - 1) // w)
    pad = np.full((nb * w,) + x.shape[1:], _NEG, np.int64)
    pad[:n] = x
    blocks = pad.reshape((nb, w) + x.shape[1:])
    prefix = np.maximum.accumulate(blocks, axis=1).reshape(pad.shape)
    suffix = np.maximum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].reshape(pad.shape)
    return np.moveaxis(np.maximum(suffix[:n], prefix[w - 1 : w - 1 + n]), 0, axis)


def _cumsum2(x: np.ndarray) -> np.ndarray:
    cs = np.zeros((x.shape[0] + 1, x.shape[1] + 1), np.int64)
    cs[1:, 1:] = x.cumsum(0).cumsum(1)
    return cs


def _rect_sum(cs, i0, i1, j0, j1):
    """Sums over rectangles [i0, i1] x [j0, j1] (inclusive, arrays) of the array cs sums."""
    return cs[i1 + 1, j1 + 1] - cs[i0, j1 + 1] - cs[i1 + 1, j0] + cs[i0, j0]


def _db_key_scores(sc, left, right, i1, j1, a1, i2, j2, a2) -> np.ndarray:
    """Exact score of each key: a first anchor of a1 matching cells from (i1, j1) and a
    second of a2 from (i2, j2), with i2 >= i1 + a1 and j2 >= j1 + a1. The best total of
    one segment holding both anchors (same diagonal, no wall between them) or two gap-free
    segments, one through each anchor, that do not overlap in either protein."""
    i1, j1, i2, j2 = (np.asarray(v, np.int64) for v in (i1, j1, i2, j2))
    out1 = left[i1, j1].astype(np.int64)
    out2 = right[i2 + a2, j2 + a2].astype(np.int64)
    L = np.minimum(i2 - i1 - a1, j2 - j1 - a1)
    y = np.arange(DB_WINDOW + 1)
    after = sc[(i1 + a1)[:, None] + y, (j1 + a1)[:, None] + y].astype(np.int64)
    before = sc[(i2 - 1)[:, None] - y, (j2 - 1)[:, None] - y].astype(np.int64)
    zero = np.zeros((i1.size, 1), np.int64)
    r1 = np.maximum.accumulate(np.hstack([zero, after.cumsum(1)]), axis=1)
    l2 = np.maximum.accumulate(np.hstack([zero, before.cumsum(1)]), axis=1)
    x = np.arange(DB_WINDOW + 2)[None, :]
    rest = L[:, None] - x
    pair = np.take_along_axis(r1, x.repeat(i1.size, 0), 1) + np.take_along_axis(
        l2, np.clip(rest, 0, None), 1
    )
    inner = np.where(rest >= 0, pair, _NEG).max(axis=1)
    best = a1 + a2 + out1 + out2 + inner
    same_diag = (j1 - i1) == (j2 - i2)
    if same_diag.any():
        k = np.flatnonzero(same_diag)
        n_span = (i2[k] + a2 - i1[k])[:, None]
        z = np.arange(2 * DB_WINDOW + DB_SEED + 2)[None, :]
        cells = sc[
            np.clip(i1[k][:, None] + z, 0, sc.shape[0] - 1),
            np.clip(j1[k][:, None] + z, 0, sc.shape[1] - 1),
        ].astype(np.int64)
        span = np.where(z < n_span, cells, 0).sum(axis=1) + out1[k] + out2[k]
        best[k] = np.maximum(best[k], span)
    return best


def _best_over_rectangles(
    sc,
    left,
    right,
    i1,
    j1,
    a1,
    bound,
    partners,
    a2,
    row0,
    col0,
    best=0,
    off_diag=False,
    anchor_u=None,
    partner_u=None,
):
    """Highest exact key score over first anchors (i1, j1), each with its partners in the
    33 x 33 rectangle starting at (i1 + row0, j1 + col0) of the boolean array partners
    (with off_diag, the anchor's own diagonal is left out). Anchors are visited in chunks
    from the highest upper bound down, and the search stops when no remaining bound can
    beat the best key found (starting from `best`). With anchor_u (per anchor) and
    partner_u (per cell), the unconstrained segment scores, a key is scored exactly only
    when their sum, an upper bound on its score, can beat the best."""
    order = np.argsort(-bound, kind="stable")
    y = np.arange(DB_WINDOW + 1)
    own = np.eye(DB_WINDOW + 1, dtype=bool)[None] if off_diag else None
    pos, chunk = 0, 16
    while pos < order.size and bound[order[pos]] > best:
        k = order[pos : pos + chunk]
        k = k[bound[k] > best]
        pos += chunk
        r0, c0 = i1[k] + row0, j1[k] + col0
        block = partners[
            r0[:, None, None] + y[None, :, None], c0[:, None, None] + y[None, None, :]
        ]
        if own is not None:
            block &= ~own
        which, di, dj = np.nonzero(block)
        if which.size == 0:
            continue
        p_i, p_j = r0[which] + di, c0[which] + dj
        f_i, f_j = i1[k][which], j1[k][which]
        if anchor_u is not None:
            hi = anchor_u[k][which] + partner_u[p_i, p_j] > best
            if not hi.any():
                continue
            p_i, p_j, f_i, f_j = p_i[hi], p_j[hi], f_i[hi], f_j[hi]
        if row0 > 0:  # partners come after the anchor
            s = _db_key_scores(sc, left, right, f_i, f_j, a1, p_i, p_j, a2)
        else:  # partners come before it: they are the first anchor of the key
            s = _db_key_scores(sc, left, right, p_i, p_j, a2, f_i, f_j, a1)
        best = max(best, int(s.max()))
    return best


def _same_diagonal_seed_cell(sc, left, right, si, sj, seed, after: bool) -> np.ndarray:
    """B3 keys whose matching cell sits on the seed's own diagonal, at most DB_WINDOW cells
    after the seed's end (after=True) or before its start: the best key score per seed.

    On one diagonal the two segments are the seed's and the cell's, and the cells between
    them (g of them) are split into a prefix kept by the first segment and a suffix kept by
    the second; the best split keeps all g cells minus their lowest-scoring run (possibly
    empty, which is the one segment holding both). A wall in between is part of that
    lowest run, so it is never kept."""
    if si.size == 0:
        return np.empty(0, np.int64)
    y = np.arange(DB_WINDOW + 1)
    if after:
        ci, cj = (si + seed)[:, None] + y, (sj + seed)[:, None] + y
    else:
        ci, cj = (si - 1)[:, None] - y, (sj - 1)[:, None] - y
    cells = sc[ci, cj].astype(np.int64)  # cells[:, g]: the cell g steps away
    n, w = cells.shape
    total = np.zeros(n, np.int64)
    run_min = np.zeros(n, np.int64)  # lowest run ending just before step g
    low = np.zeros(n, np.int64)  # lowest run anywhere in the first g cells
    keep = np.zeros((n, w), np.int64)  # best kept total of the first g cells
    for g in range(w):
        keep[:, g] = total - low
        total += cells[:, g]
        run_min = np.minimum(run_min + cells[:, g], 0)
        low = np.minimum(low, run_min)
    if after:
        outer = left[si, sj][:, None] + right[ci + 1, cj + 1]
    else:
        outer = left[ci, cj] + right[si + seed, sj + seed][:, None]
    score = outer.astype(np.int64) + seed + 1 + keep
    return np.where(cells == 1, score, _NEG).max(axis=1)


def diagonal_scan(
    q: np.ndarray,
    t: np.ndarray,
    q_partner: np.ndarray,
    t_partner: np.ndarray,
    allowed: np.ndarray | None = None,
) -> dict:
    """B2, the chosen linked pair, its randstrobe and B3 for one protein pair, each the best
    key score over every diagonal, plus how many keys each finds.

    q, t: HP class arrays (TABLES[ALPHABET] of the residues, GAP for letters outside the two
    classes). q_partner, t_partner: randstrobe_partner(q, DB_ANCHOR, DB_WINDOW) and the
    same for t. allowed: optional (len(q), len(t)) mask of the cells that may be used.

    Returns a dict with ``<score>`` and ``<score>_count`` for each of DB_SCORES, plus
    ``words6`` and ``words8``, the number of exact shared 6- and 8-letter HP words."""
    sc = _db_score_matrix(q, t, allowed)
    left, right = _diagonal_extensions(sc)
    match = sc == 1
    A = _diagonal_windows(match, DB_ANCHOR)
    S = _diagonal_windows(match, DB_SEED)
    out = {"words6": int(A.sum()), "words8": int(S.sum())}
    span = DB_WINDOW + 1

    def rect_max(v, lo):
        """Max of v over rows i+lo .. i+lo+32 and columns j+lo .. j+lo+32, at (i, j)."""
        m = _forward_max(_forward_max(v, span, 0), span, 1)
        res = np.full(v.shape, _NEG, np.int64)
        if lo >= 0:
            res[: v.shape[0] - lo, : v.shape[1] - lo] = m[lo:, lo:]
        else:
            res[-lo:, -lo:] = m[: v.shape[0] + lo, : v.shape[1] + lo]
        return res

    # B2: the best segment holding an exact 8-letter word
    si, sj = np.nonzero(S)
    u8 = DB_SEED + left[si, sj].astype(np.int64) + right[si + DB_SEED, sj + DB_SEED]
    out["B2"] = int(u8.max()) if si.size else 0
    out["B2_count"] = int(si.size)

    # linked pair: every pair of 6-letter anchors, starts at most 32 apart, gap not stored
    ai, aj = np.nonzero(A)
    a = DB_ANCHOR
    if ai.size:
        cs = _cumsum2(A)
        n_keys = _rect_sum(cs, ai + a, ai + a + DB_WINDOW, aj + a, aj + a + DB_WINDOW)
        out["linked_count"] = int(n_keys.sum())
        RR, CC = left.shape[0] - 1, left.shape[1] - 1
        ua = np.full(A.shape, _NEG, np.int64)
        ua[ai, aj] = (
            a + left[ai, aj] + right[np.minimum(ai + a, RR), np.minimum(aj + a, CC)]
        )
        bound = ua[ai, aj] + rect_max(ua, a)[ai, aj]
        has = n_keys > 0
        out["linked"] = _best_over_rectangles(
            sc,
            left,
            right,
            ai[has],
            aj[has],
            a,
            bound[has],
            A,
            a,
            a,
            a,
            anchor_u=ua[ai[has], aj[has]],
            partner_u=ua,
        )
        # randstrobe: one partner per first anchor, chosen in each protein on its own
        pq = q_partner[ai - DB_PAD]
        pt = t_partner[aj - DB_PAD]
        ok = (pq >= 0) & (pt >= 0)
        ok[ok] = A[pq[ok] + DB_PAD, pt[ok] + DB_PAD]
        out["randstrobe_count"] = int(ok.sum())
        out["randstrobe"] = (
            int(
                _db_key_scores(
                    sc,
                    left,
                    right,
                    ai[ok],
                    aj[ok],
                    a,
                    pq[ok] + DB_PAD,
                    pt[ok] + DB_PAD,
                    a,
                ).max()
            )
            if ok.any()
            else 0
        )
    else:
        out.update(linked=0, linked_count=0, randstrobe=0, randstrobe_count=0)

    # B3: the 8-letter seed plus any matching cell within 32 residues, before or after it
    if si.size:
        cm = _cumsum2(match)
        s8 = DB_SEED
        n_after = _rect_sum(
            cm, si + s8, si + s8 + DB_WINDOW, sj + s8, sj + s8 + DB_WINDOW
        )
        n_before = _rect_sum(cm, si - span, si - 1, sj - span, sj - 1)
        out["B3_count"] = int((n_after + n_before).sum())
        mi, mj = np.nonzero(match)
        u1 = np.full(match.shape, _NEG, np.int64)
        u1[mi, mj] = 1 + left[mi, mj] + right[mi + 1, mj + 1]
        best = 0
        for after, row0 in ((True, s8), (False, -span)):
            same = _same_diagonal_seed_cell(sc, left, right, si, sj, s8, after)
            best = max(best, int(same.max()))
            bound = u8 + rect_max(u1, row0)[si, sj]
            best = _best_over_rectangles(
                sc,
                left,
                right,
                si,
                sj,
                s8,
                bound,
                match,
                1,
                row0,
                row0,
                best,
                True,
                anchor_u=u8,
                partner_u=u1,
            )
        out["B3"] = max(best, 0)
    else:
        out.update(B3=0, B3_count=0)
    return out


def alignment_cells(qaln: str, taln: str):
    """The residues of an alignment's two rows (gaps removed) and the mask of the cells the
    alignment pairs, for running diagonal_scan restricted to that alignment."""
    qb = np.frombuffer(qaln.encode(), np.uint8)
    tb = np.frombuffer(taln.encode(), np.uint8)
    _, qpos, _ = _residue_maps(qb)
    _, tpos, _ = _residue_maps(tb)
    q_seq = qb[qb != ord("-")].tobytes().decode()
    t_seq = tb[tb != ord("-")].tobytes().decode()
    both = (qpos >= 0) & (tpos >= 0)
    allowed = np.zeros((len(q_seq), len(t_seq)), bool)
    allowed[qpos[both], tpos[both]] = True
    return q_seq, t_seq, allowed


def hp_classes(seq: str) -> np.ndarray:
    """HP classes of a residue string in hp_thomas_dill2, GAP for any other letter."""
    return TABLES[ALPHABET][np.frombuffer(seq.encode(), np.uint8)]


def scan_sequences(query: str, target: str, allowed=None) -> dict:
    """diagonal_scan on two residue strings, randstrobe partners computed here."""
    q, t = hp_classes(query), hp_classes(target)
    return diagonal_scan(
        q,
        t,
        randstrobe_partner(q, DB_ANCHOR, DB_WINDOW),
        randstrobe_partner(t, DB_ANCHOR, DB_WINDOW),
        allowed,
    )


def alignment_rows_from_pairs(seq_q: str, seq_t: str, pos_q, pos_t) -> tuple[str, str]:
    """Two gapped alignment rows from aligned residue pairs (1-based positions, both
    increasing), from the first aligned pair to the last. Residues between two aligned
    pairs that are not themselves aligned become gap columns: the query's first, then the
    target's."""
    pos_q, pos_t = list(pos_q), list(pos_t)
    if any(b <= a for a, b in zip(pos_q, pos_q[1:])) or any(
        b <= a for a, b in zip(pos_t, pos_t[1:])
    ):
        raise ValueError("aligned positions must increase in both sequences")
    qrow, trow = [seq_q[pos_q[0] - 1]], [seq_t[pos_t[0] - 1]]
    for (q0, t0), (q1, t1) in zip(zip(pos_q, pos_t), zip(pos_q[1:], pos_t[1:])):
        gap_q = seq_q[q0 : q1 - 1]
        gap_t = seq_t[t0 : t1 - 1]
        qrow.append(gap_q + "-" * len(gap_t))
        trow.append("-" * len(gap_q) + gap_t)
        qrow.append(seq_q[q1 - 1])
        trow.append(seq_t[t1 - 1])
    return "".join(qrow), "".join(trow)


def _walk_extension(sc, i, j, step, limit=None):
    """Cells from (i, j) along the diagonal in direction step (+1 or -1) until a wall:
    the length of the shortest walk reaching the best running total, and that total."""
    best, best_len, tot, y = 0, 0, 0, 0
    while limit is None or y < limit:
        c = int(sc[i + step * y, j + step * y])
        if c <= DB_WALL // 2:
            break
        tot += c
        y += 1
        if tot > best:
            best, best_len = tot, y
    return best_len, best


def key_segments(sc, i1, j1, a1, i2, j2, a2):
    """The segments of one key's best score (padded coordinates, as _db_key_scores): a list
    of (query start, query end exclusive, target start), one segment or two, and the total.
    The total equals _db_key_scores for the key."""
    out_l, best_l = _walk_extension(sc, i1 - 1, j1 - 1, -1)
    out_r, best_r = _walk_extension(sc, i2 + a2, j2 + a2, +1)
    L = min(i2 - i1 - a1, j2 - j1 - a1)
    after = [int(sc[i1 + a1 + y, j1 + a1 + y]) for y in range(L)]
    before = [int(sc[i2 - 1 - y, j2 - 1 - y]) for y in range(L)]
    pre = np.r_[0, np.cumsum(after)] if L else np.zeros(1, int)
    suf = np.r_[0, np.cumsum(before)] if L else np.zeros(1, int)
    r1 = np.maximum.accumulate(pre)
    l2 = np.maximum.accumulate(suf)
    x = int(np.argmax([r1[x] + l2[L - x] for x in range(L + 1)]))
    y1 = int(np.argmax(pre[: x + 1] == r1[x]))  # shortest walk reaching the best
    y2 = int(np.argmax(suf[: L - x + 1] == l2[L - x]))
    inner = int(r1[x] + l2[L - x])
    total = a1 + a2 + best_l + best_r + inner
    q1s, t1s = i1 - out_l, j1 - out_l
    q1e = i1 + a1 + y1
    q2s, t2s = i2 - y2, j2 - y2
    q2e = i2 + a2 + out_r
    same = (j1 - i1) == (j2 - i2)
    if same and q1e == q2s:
        segs = [(q1s, q2e, t1s)]
    else:
        segs = [(q1s, q1e, t1s), (q2s, q2e, t2s)]
    if same:  # one segment over everything between the anchors, if no wall in it
        span_cells = sc[np.arange(i1, i2 + a2), np.arange(j1, j2 + a2)]
        if (span_cells > DB_WALL // 2).all():
            span = best_l + best_r + int(span_cells.sum())
            if span > total:
                segs, total = [(q1s, q2e, t1s)], span
    return [(a - DB_PAD, b - DB_PAD, c - DB_PAD) for a, b, c in segs], total


def best_keys_with_segments(query: str, target: str, allowed=None, chunk: int = 20_000):
    """For drawing: the best chosen-linked-pair key and the best B3 key of two sequences
    (every diagonal, or only the allowed cells), each as a dict with both anchors (query
    and target start, unpadded), the anchor lengths, the score and its segments. Among keys
    with the best score the first found is kept: first anchors in (query start, target
    start) order, and for B3 the keys with the matching residue after the 8-letter word
    before those with it before the word. None where a score has no key.
    """
    q, t = hp_classes(query), hp_classes(target)
    sc = _db_score_matrix(q, t, allowed)
    left, right = _diagonal_extensions(sc)
    match = sc == 1
    w = DB_WINDOW
    out = {}
    for name, first, a1, second, a2, orders in [
        ("linked", _diagonal_windows(match, DB_ANCHOR), DB_ANCHOR,
         _diagonal_windows(match, DB_ANCHOR), DB_ANCHOR, ["after"]),
        ("B3", _diagonal_windows(match, DB_SEED), DB_SEED, match, 1, ["after", "before"]),
    ]:  # fmt: skip
        fi, fj = np.nonzero(first)
        si, sj = np.nonzero(second)
        best = None
        for order in orders:
            step = max(1, chunk // 100)  # first anchors per batch
            for c in range(0, fi.size, step):
                bi, bj = fi[c : c + step], fj[c : c + step]
                if order == "after":
                    di = si[None, :] - bi[:, None] - a1
                    dj = sj[None, :] - bj[:, None] - a1
                else:
                    di = bi[:, None] - si[None, :] - a2
                    dj = bj[:, None] - sj[None, :] - a2
                ok = (di >= 0) & (dj >= 0) & (di <= w) & (dj <= w)
                k1, k2 = np.nonzero(ok)
                if k1.size == 0:
                    continue
                if order == "after":
                    key = (bi[k1], bj[k1], a1, si[k2], sj[k2], a2)
                else:
                    key = (si[k2], sj[k2], a2, bi[k1], bj[k1], a1)
                s = _db_key_scores(sc, left, right, *key)
                m = int(np.argmax(s))
                cand = (int(s[m]), [int(v[m]) if np.ndim(v) else int(v) for v in key])
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            out[name] = None
            continue
        score, (i1, j1, aa1, i2, j2, aa2) = best
        segs, total = key_segments(sc, i1, j1, aa1, i2, j2, aa2)
        assert total == score, (name, total, score)
        out[name] = {
            "q1": i1 - DB_PAD, "t1": j1 - DB_PAD, "a1": aa1,
            "q2": i2 - DB_PAD, "t2": j2 - DB_PAD, "a2": aa2,
            "score": score, "segments": segs,
        }  # fmt: skip
    return out
