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


def _to_alignment(cls: np.ndarray) -> str:
    """HP classes back to residues: H -> A, P -> D, gap -> '-'."""
    return "".join("-" if c == GAP else ("A" if c == 0 else "D") for c in cls)


def _random_gapped_pair(rng, n, flip=0.12, gap=0.05):
    qa = rng.integers(0, 2, n).astype(np.uint8)
    ta = qa.copy()
    f = rng.random(n) < flip
    ta[f] = 1 - ta[f]
    gq = rng.random(n) < gap
    gt = (rng.random(n) < gap) & ~gq
    qa[gq] = GAP
    ta[gt] = GAP
    return qa, ta


def test_gap_stored_and_gap_binned_keys_match_brute_force():
    """score_pair_designs' every-pair keys with the gap stored exactly or in 7-residue
    bins, against enumerating every pair of anchors by hand."""
    rng = np.random.default_rng(5)
    n_keys_seen = {"exact": 0, "binned7": 0}
    for trial in range(30):
        n = int(rng.integers(22, 34))
        qa, ta = _random_gapped_pair(rng, n)
        q, t = _to_alignment(qa), _to_alignment(ta)
        got = lp.score_pair_designs(q, t, np.random.default_rng(0), n_shuffle=0)[0]
        sc, same, *_ = _arrays(qa, ta)
        qpos = np.where(qa != GAP, np.cumsum(qa != GAP) - 1, -1)
        tpos = np.where(ta != GAP, np.cumsum(ta != GAP) - 1, -1)
        for a in lp.DESIGN_ANCHORS:
            anchors = [c for c in range(n - a + 1) if same[c : c + a].all()]
            for tr in ["exact", "binned7"]:
                best, count = 0.0, 0
                for c1 in anchors:
                    for c2 in anchors:
                        dq = qpos[c2] - qpos[c1] - a
                        dt = tpos[c2] - tpos[c1] - a
                        if not (0 <= dq <= 32 and 0 <= dt <= 32):
                            continue
                        if tr == "exact" and dq != dt:
                            continue
                        if tr == "binned7" and dq // 7 != dt // 7:
                            continue
                        count += 1
                        best = max(best, _brute_key(sc, c1, c2, a))
                name = lp.setting_name("all", a, 32, tr)
                assert got[f"{name}_count"] == count
                assert got[f"{name}_ext"] == best
                n_keys_seen[tr] += count
    # the test reaches keys, and some of them span an indel (binned keeps more than exact)
    assert n_keys_seen["exact"] > 50
    assert n_keys_seen["binned7"] > n_keys_seen["exact"]


def test_gap_stored_and_binned_key_counts_reach_the_32_residue_cap():
    """Key counts on alignments long enough for gaps up to and past 32, so the last bins
    (21-27, 28-32) and the cap itself are exercised; counts only, against enumeration.
    """
    rng = np.random.default_rng(8)
    seen_gaps = set()
    for trial in range(25):
        n = int(rng.integers(60, 80))
        qa, ta = _random_gapped_pair(rng, n, flip=0.04, gap=0.03)
        got = lp.score_pair_designs(
            _to_alignment(qa), _to_alignment(ta), np.random.default_rng(0), 0
        )[0]
        same = (qa == ta) & (qa != GAP) & (ta != GAP)
        qpos = np.where(qa != GAP, np.cumsum(qa != GAP) - 1, -1)
        tpos = np.where(ta != GAP, np.cumsum(ta != GAP) - 1, -1)
        for a in lp.DESIGN_ANCHORS:
            anchors = [c for c in range(n - a + 1) if same[c : c + a].all()]
            for tr in ["exact", "binned7"]:
                count = 0
                for c1 in anchors:
                    for c2 in anchors:
                        dq = qpos[c2] - qpos[c1] - a
                        dt = tpos[c2] - tpos[c1] - a
                        if not (0 <= dq <= 32 and 0 <= dt <= 32):
                            continue
                        if tr == "exact" and dq != dt:
                            continue
                        if tr == "binned7" and dq // 7 != dt // 7:
                            continue
                        count += 1
                        seen_gaps.add(int(dq))
                assert got[f"{lp.setting_name('all', a, 32, tr)}_count"] == count
    assert {28, 30, 32} <= seen_gaps


def test_gap_binned_rule_bins():
    g = np.arange(33)
    assert [int(x) for x in np.unique(g // lp.GAP_BIN7)] == [0, 1, 2, 3, 4]
    # 6 apart in one bin agree; 7 apart never do
    assert lp._gap_rule(np.array([0, 7, 28]), np.array([6, 13, 32]), "binned7").all()
    assert not lp._gap_rule(np.arange(26), np.arange(26) + 7, "binned7").any()
    # binned3 is unchanged
    assert (lp._gap_rule(g, g + 2, "binned3") == (g // 3 == (g + 2) // 3)).all()


def test_single_words_and_longest_run_match_brute_force():
    rng = np.random.default_rng(6)
    for trial in range(60):
        n = int(rng.integers(20, 40))
        qa, ta = _random_gapped_pair(rng, n, flip=0.08, gap=0.03)
        got = lp.score_pair_designs(
            _to_alignment(qa), _to_alignment(ta), np.random.default_rng(0), 0
        )[0]
        sc, same, *_ = _arrays(qa, ta)
        run, longest = 0, 0
        for x in same:
            run = run + 1 if x else 0
            longest = max(longest, run)
        assert got["longest_run"] == longest
        for k in lp.SINGLE_WORDS:
            want = 0.0
            for i in range(n):
                for j in range(i + k, n + 1):
                    s = _seg(sc, i, j)
                    has_run = any(same[c : c + k].all() for c in range(i, j - k + 1))
                    if np.isfinite(s) and has_run:
                        want = max(want, s)
            assert got[f"word{k}_ext"] == want


def test_designs_use_score_pairs_shuffles():
    """score_pair_designs draws the same shuffled targets as score_pair, so its null rows
    pair with the stored ones."""
    rng = np.random.default_rng(7)
    qa, ta = _random_gapped_pair(rng, 60, flip=0.2)
    q, t = _to_alignment(qa), _to_alignment(ta)
    a = lp.score_pair(q, t, np.random.default_rng(11), n_shuffle=2)
    b = lp.score_pair_designs(q, t, np.random.default_rng(11), n_shuffle=2)
    assert (
        [r["kind"] for r in a] == [r["kind"] for r in b] == ["real", "null0", "null1"]
    )
    for ra, rb in zip(a, b):
        assert ra["B2_ext"] == rb["word8_ext"] and ra["B1_ext"] == rb["word10_ext"]
        for x in lp.DESIGN_ANCHORS:
            nm = lp.setting_name("all", x, 32, "exact")
            assert ra[f"{nm}_ext"] == rb[f"{nm}_ext"]
            assert ra[f"{nm}_count"] == rb[f"{nm}_count"]


def test_joint_codes_use_score_pairs_shuffles():
    """comparison_joint_codes' null rows are score_pair's shuffled targets: B2 rescored from
    the joint codes (match = HH or PP) equals score_pair's B2_ext on every row."""
    from seed_extend_utils import best_segment_containing_runs

    rng = np.random.default_rng(3)
    for trial in range(20):
        qa, ta = _random_gapped_pair(rng, 70, flip=0.2)
        q, t = _to_alignment(qa), _to_alignment(ta)
        a = lp.score_pair(q, t, np.random.default_rng(trial), n_shuffle=2)
        b = lp.comparison_joint_codes(q, t, np.random.default_rng(trial), n_shuffle=2)
        assert [r["kind"] for r in a] == [k for k, _ in b]
        for ra, (_, codes) in zip(a, b):
            ok = codes != lp.NOT_SCORED
            same = ok & ((codes == 0) | (codes == 3))
            sc = np.where(same, 1.0, -2.0)
            sc[~ok] = -10_000.0
            assert best_segment_containing_runs(sc, same, 8) == ra["B2_ext"]


def test_logodds_matrix_by_hand():
    m = lp.logodds_matrix(np.array([40, 10, 20, 30]))
    sym = np.array([0.4, 0.15, 0.15, 0.3])
    h = 0.55
    want = np.log2(sym / np.array([h * h, h * (1 - h), (1 - h) * h, (1 - h) ** 2]))
    assert np.allclose(m["pr"], [0.4, 0.1, 0.2, 0.3])
    assert np.allclose(m["score"], want) and m["score"][1] == m["score"][2]
    assert np.isclose(m["relative_entropy"], (sym * want).sum())
    # relative entropy is >= 0 and the unrelated-pair expectation is <= 0
    assert m["relative_entropy"] > 0 > m["expected_unrelated"]


def test_best_stretch_and_full_sum_match_brute_force():
    rng = np.random.default_rng(5)
    score = np.array([0.7, -0.9, -0.9, 0.5])
    codes_all, comp_all, want_best, want_full = [], [], [], []
    for c in range(200):
        n = int(rng.integers(0, 25))
        codes = rng.integers(0, 4, n).astype(np.int8)
        codes[rng.random(n) < 0.15] = lp.NOT_SCORED
        best = 0.0
        for i in range(n):
            for j in range(i + 1, n + 1):
                if (codes[i:j] == lp.NOT_SCORED).any():
                    break
                best = max(best, score[codes[i:j]].sum())
        want_best.append(best)
        want_full.append(score[codes[codes != lp.NOT_SCORED]].sum())
        codes_all.append(codes)
        comp_all.append(np.full(n, c))
    codes = np.concatenate(codes_all)
    comp = np.concatenate(comp_all)
    got = lp.best_stretch_scores(codes, comp, 200, score)
    assert np.allclose(got, want_best)
    full = lp.joint_counts(codes, comp, 200) @ score
    assert np.allclose(full, want_full)
