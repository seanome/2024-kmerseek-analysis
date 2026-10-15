"""Brute-force checks of the linked-pair scoring in notebooks/linked_pairs_utils.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import linked_pairs_utils as lp  # noqa: E402
from hp_conservation_utils import GAP, TABLES  # noqa: E402
from seed_extend_utils import column_scores  # noqa: E402


def _seg(sc, i, j):
    """Score of columns [i, j), or -inf if it crosses a gap column."""
    s = sc[i:j]
    return -np.inf if (s <= -1000).any() else s.sum()


def _brute_key(sc, c1, c2, a):
    """Best total of <= 2 disjoint gap-free segments containing [c1,c1+a) and [c2,c2+a)."""
    n = sc.size
    best = -np.inf
    # one segment containing both
    for i in range(0, c1 + 1):
        for j in range(c2 + a, n + 1):
            best = max(best, _seg(sc, i, j))
    # two disjoint segments
    for i1 in range(0, c1 + 1):
        for j1 in range(c1 + a, c2 + 1):
            s1 = _seg(sc, i1, j1)
            for i2 in range(j1, c2 + 1):
                for j2 in range(c2 + a, n + 1):
                    best = max(best, s1 + _seg(sc, i2, j2))
    return best


def test_key_scores_match_brute_force():
    rng = np.random.default_rng(1)
    for trial in range(120):
        n = int(rng.integers(14, 24))
        qa = rng.integers(0, 2, n).astype(np.uint8)
        ta = qa.copy()
        flip = rng.random(n) < 0.25
        ta[flip] = 1 - ta[flip]
        gaps = rng.random(n) < 0.06
        ta[gaps] = GAP
        sc, same = column_scores(qa, ta, 2.0)
        both = (qa != GAP) & (ta != GAP)
        P = np.concatenate(([0.0], np.cumsum(sc)))
        left = P - np.minimum.accumulate(P)
        right = np.maximum.accumulate(P[::-1])[::-1] - P
        gapcount = np.cumsum(~both)
        a = int(rng.integers(2, 4))
        M = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(same, a).all(1))
        pairs = [(x, y) for x in M for y in M if y >= x + a]
        if not pairs:
            continue
        c1 = np.array([p[0] for p in pairs])
        c2 = np.array([p[1] for p in pairs])
        got = lp._key_scores(c1, c2, a, P, left, right, gapcount, n)
        want = np.array([_brute_key(sc, x, y, a) for x, y in pairs])
        np.testing.assert_allclose(got, want)


def test_randstrobe_partner_window_and_determinism():
    rng = np.random.default_rng(2)
    cls = rng.integers(0, 2, 200).astype(np.uint8)
    for a, g in [(6, 8), (8, 32)]:
        sel = lp.randstrobe_partner(cls, a, g)
        i = np.flatnonzero(sel >= 0)
        assert ((sel[i] - i - a >= 0) & (sel[i] - i - a <= g)).all()
        assert (sel[i] + a <= cls.size).all()
        # identical window content gives the identical choice
        sel2 = lp.randstrobe_partner(cls.copy(), a, g)
        assert (sel == sel2).all()


def test_identical_sequences_share_every_randstrobe_key():
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAACRESHLLL"
    rows = lp.score_pair(seq, seq, np.random.default_rng(0), n_shuffle=0)
    r = rows[0]
    tab = TABLES[lp.ALPHABET]
    cls = tab[np.frombuffer(seq.encode(), np.uint8)]
    for a in lp.ANCHOR_LENGTHS:
        for g in lp.GAP_MAXES:
            n_first = int((lp.randstrobe_partner(cls, a, g) >= 0).sum())
            assert r[f"{lp.setting_name('rand', a, g, 'exact')}_count"] == n_first
            # whole sequence matches: best key score is the full length
            assert r[f"{lp.setting_name('rand', a, g, 'exact')}_ext"] == len(seq)


def _brute_key2(sc, c1, a1, c2, a2):
    """_brute_key with first anchor length a1 and second anchor length a2."""
    n = sc.size
    best = -np.inf
    for i in range(0, c1 + 1):
        for j in range(c2 + a2, n + 1):
            best = max(best, _seg(sc, i, j))
    for i1 in range(0, c1 + 1):
        for j1 in range(c1 + a1, c2 + 1):
            s1 = _seg(sc, i1, j1)
            for i2 in range(j1, c2 + 1):
                for j2 in range(c2 + a2, n + 1):
                    best = max(best, s1 + _seg(sc, i2, j2))
    return best


def _arrays(qa, ta):
    sc, same = column_scores(qa, ta, 2.0)
    both = (qa != GAP) & (ta != GAP)
    P = np.concatenate(([0.0], np.cumsum(sc)))
    left = P - np.minimum.accumulate(P)
    right = np.maximum.accumulate(P[::-1])[::-1] - P
    return sc, same, P, left, right, np.cumsum(~both)


def test_mixed_length_key_scores_and_b3_match_brute_force():
    rng = np.random.default_rng(3)
    n_checked = 0
    for trial in range(80):
        n = int(rng.integers(12, 20))
        qa = rng.integers(0, 2, n).astype(np.uint8)
        ta = qa.copy()
        flip = rng.random(n) < 0.25
        ta[flip] = 1 - ta[flip]
        # gaps as alignment columns: a residue in one sequence only
        gq = rng.random(n) < 0.06
        gt = (rng.random(n) < 0.06) & ~gq
        qa[gq] = GAP
        ta[gt] = GAP
        sc, same, P, left, right, gapcount = _arrays(qa, ta)
        qpos = np.where(qa != GAP, np.cumsum(qa != GAP) - 1, -1)
        tpos = np.where(ta != GAP, np.cumsum(ta != GAP) - 1, -1)
        seed, gmax = 3, 4
        S = [c for c in range(n - seed + 1) if same[c : c + seed].all()]
        M1 = [c for c in range(n) if same[c]]
        want = 0.0
        for c1, a1, c2s, a2 in [(x, seed, M1, 1) for x in S] + [
            (x, 1, S, seed) for x in M1
        ]:
            for c2 in c2s:
                dq = qpos[c2] - qpos[c1] - a1
                dt = tpos[c2] - tpos[c1] - a1
                if 0 <= dq <= gmax and 0 <= dt <= gmax:
                    want = max(want, _brute_key2(sc, c1, a1, c2, a2))
                    got1 = lp._key_scores(
                        np.array([c1]),
                        np.array([c2]),
                        a1,
                        P,
                        left,
                        right,
                        gapcount,
                        gmax,
                        a2=a2,
                    )[0]
                    assert got1 == _brute_key2(sc, c1, a1, c2, a2)
                    n_checked += 1
        got = lp.seed_plus_second_segment(
            qpos, tpos, same, seed, gmax, P, left, right, gapcount
        )
        assert got == want
    assert n_checked > 100


def test_recall_at_exact_fpr_calls_exactly_fpr_of_the_null():
    rng = np.random.default_rng(4)
    null = np.zeros((4000, 2, 1))
    null[:6, 0, 0] = 5  # 6 of 8,000 null scores at 5
    null[6:20, 1, 0] = 3  # 14 at 3
    real = rng.integers(0, 7, (4000, 1)).astype(float)
    bins = rng.integers(0, len(lp.BINS), 4000)
    thr, w, rec = lp.recall_at_exact_fpr(real, null, bins, 1e-3)
    assert thr[0] == 3
    # randomised tie rule: everything above 3, plus fraction w of the ties at 3
    achieved = (null.ravel() > 3).mean() + w[0] * (null.ravel() == 3).mean()
    assert abs(achieved - 1e-3) < 1e-12
    for b in range(len(lp.BINS)):
        r = real[bins == b, 0]
        assert np.isclose(rec[b, 0], (r > 3).mean() + w[0] * (r == 3).mean())
