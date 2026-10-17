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

    Returns threshold t (cols), weight of the "at or above" side (cols), recall (bins x cols).
    """
    flat = null.reshape(-1, null.shape[-1])
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
