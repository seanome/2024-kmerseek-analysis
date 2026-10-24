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


# ---------------------------------------------------------------------------
# Section 4f
# ---------------------------------------------------------------------------


def test_place_in_gaps_keeps_gaps_and_x_columns():
    taln = "AD-X-DDA--"
    assert lp.place_in_gaps(taln, "ADDDA") == taln  # its own scored residues back
    assert lp.place_in_gaps(taln, "WKKLL") == "WK-X-KLL--"
    try:
        lp.place_in_gaps(taln, "ADD")
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong residue count must be refused")


def test_draw_unrelated_other_clan_long_enough():
    rng = np.random.default_rng(0)
    clans = np.array(["CL1", "CL2", "CL1", "CL2", "CL3"])
    lens = np.array([5, 10, 20, 30, 40])
    for _ in range(500):
        L = int(rng.integers(1, 41))
        j, off = lp.draw_unrelated(rng, L, "CL2", clans, lens)
        assert clans[j] != "CL2" and lens[j] >= L and 0 <= off <= lens[j] - L
    assert lp.draw_unrelated(rng, 41, "CL2", clans, lens) == (
        None,
        "longer than any candidate",
    )
    clans2 = np.array(["CL1", "CL1", "CL1", "CL2", "CL2"])
    assert lp.draw_unrelated(rng, 25, "CL2", clans2, lens, max_tries=5) == (
        None,
        "no other-clan candidate long enough",
    )
    # the fallback after max_tries still finds the one other-clan candidate
    clans3 = np.array(["CL2", "CL2", "CL2", "CL2", "CL1"])
    hits = {
        lp.draw_unrelated(rng, 1, "CL2", clans3, lens, max_tries=0)[0]
        for _ in range(20)
    }
    assert hits == {4}


def test_pfam_clans_reads_ac_and_cl(tmp_path):
    f = tmp_path / "seed"
    f.write_text(
        "# STOCKHOLM 1.0\n#=GF ID   a\n#=GF AC   PF00001.5\n#=GF CL   CL0192\n//\n"
        "# STOCKHOLM 1.0\n#=GF ID   b\n#=GF AC   PF00002.1\n//\n"
    )
    assert lp.pfam_clans(f) == {"PF00001": "CL0192", "PF00002": "PF00002"}


def test_shuffled_target_rows_are_score_pairs_shuffles():
    rng = np.random.default_rng(3)
    qa, ta = _random_gapped_pair(rng, 80, flip=0.2)
    q = _to_alignment(qa)
    t = "".join(
        c if i % 17 else ("X" if c != "-" else c)
        for i, c in enumerate(_to_alignment(ta))
    )
    stored = lp.score_pair(q, t, np.random.default_rng(42), n_shuffle=2)
    rows = lp.shuffled_target_rows(t, 42, 2)
    for k, row in enumerate(rows):
        assert [i for i, c in enumerate(row) if c in "-X"] == [
            i for i, c in enumerate(t) if c in "-X"
        ]
        (r,) = lp.score_pair(q, row, np.random.default_rng(0), n_shuffle=0)
        for col in [
            "B1_ext",
            "B2_ext",
            "B3_ext",
            "all_a6_g32_none_ext",
            "rand_a6_g32_none_ext",
        ]:
            assert r[col] == stored[k + 1][col]


def test_recall_at_exact_fpr_skips_nan_null_scores():
    rng = np.random.default_rng(5)
    real = rng.integers(0, 30, (400, 2)).astype(float)
    null = rng.integers(0, 25, (400, 2, 2)).astype(float)
    bins = rng.integers(0, len(lp.BINS), 400)
    nan_null = null.copy()
    nan_null[:7, 1, :] = np.nan
    thr, w, rec = lp.recall_at_exact_fpr(real, nan_null, bins, 0.01)
    flat = null.reshape(-1, 2)
    keep = ~np.isnan(nan_null.reshape(-1, 2)[:, 0])
    # the same rule on the remaining null scores, by hand
    for c in range(2):
        v = np.sort(flat[keep, c])
        t = np.quantile(v, 0.99, method="higher")
        assert thr[c] == t
        f_hi, f_lo = (v > t).mean(), (v >= t).mean()
        assert np.isclose(w[c], (0.01 - f_hi) / (f_lo - f_hi) if f_lo > f_hi else 0.0)
    # and no NaN gives what it gave before
    a = lp.recall_at_exact_fpr(real, null, bins, 0.01)
    flat2 = null.reshape(-1, 2)
    assert np.array_equal(a[0], np.quantile(flat2, 0.99, axis=0, method="higher"))


def _fake_kmerseek_csv(path, rows):
    """rows: (query, target, evalue, n_chained); every other column filled with a
    placeholder, in kmerseek v0.4.0's order."""
    with open(path, "w") as fh:
        fh.write(",".join(lp.KMERSEEK_040_CSV_COLUMNS) + "\n")
        for q, t, e, ch in rows:
            vals = {c: "1" for c in lp.KMERSEEK_040_CSV_COLUMNS}
            vals.update(
                query_name=q,
                target_name=t,
                region_evalue=e,
                region_n_chained=str(ch),
                moltype="hp_thomas_dill2",
                region_subseq="ACDA",
                target_subseq="ACDA",
                moltype_seq="hphh",
                remove_low_complexity="false",
            )
            fh.write(",".join(vals[c] for c in lp.KMERSEEK_040_CSV_COLUMNS) + "\n")


def test_read_kmerseek_best_evalue_and_cutoff(tmp_path):
    import polars as pl

    real = tmp_path / "real.csv"
    _fake_kmerseek_csv(
        real,
        [
            ("q1", "t1", "0.5", 1),
            ("q1", "t1", "1.5e-3", 2),
            ("q2", "t2", "3.0", 1),
            ("q1", "t2", "0.0001", 1),
            ("q3", "t9", "0.1", 1),
        ],
    )
    best = lp.read_kmerseek_best_evalue(real)
    got = {
        (r["query_name"], r["target_name"]): (r["region_evalue"], r["region_n_chained"])
        for r in best.iter_rows(named=True)
    }
    assert got == {
        ("q1", "t1"): (1.5e-3, 2),
        ("q2", "t2"): (3.0, 1),
        ("q1", "t2"): (1e-4, 1),
        ("q3", "t9"): (0.1, 1),
    }
    bad = tmp_path / "bad.csv"
    bad.write_text("query_name,target_name,region_evalue\nq,t,1\n")
    try:
        lp.read_kmerseek_best_evalue(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a header in another layout must be refused")


def test_evalue_cutoff_recall_matches_brute_force():
    rng = np.random.default_rng(9)
    n_pairs, n_targets = 60, 50
    query_of = rng.integers(0, 45, n_pairs)  # some pairs share a query
    share = np.bincount(query_of)[query_of]
    own = np.where(rng.random(n_pairs) < 0.1, np.inf, rng.exponential(1.0, n_pairs))
    # every query's null row against every target; report only E <= 3
    full = {q: rng.exponential(30.0, n_targets) for q in np.unique(query_of)}
    for fpr in (0.01, 0.02, 0.05):
        for mult in (None, rng.integers(0, 3, n_pairs)):
            m = np.ones(n_pairs) if mult is None else mult
            # brute force over distinct queries, each weighted by its draws / share
            wq = {q: (m[query_of == q] / share[query_of == q]).sum() for q in full}
            vals = np.concatenate([full[q] for q in full])
            wts = np.concatenate([np.full(n_targets, wq[q]) for q in full])
            n_null = sum(wq.values()) * n_targets
            cands = [v for v in np.unique(vals) if wts[vals <= v].sum() <= fpr * n_null]
            want_cut = max(cands) if cands else None
            np_, ne_ = [], []
            for p in range(n_pairs):  # per pair, its query's reported row
                r = full[query_of[p]]
                r = r[r <= 3]
                np_ += [p] * r.size
                ne_ += list(r)
            o = np.argsort(ne_, kind="stable")
            cut, ach, rec = lp.evalue_cutoff_recall(
                own, np.array(np_)[o], np.array(ne_)[o], share, n_targets, fpr, mult
            )
            if want_cut is None:
                assert ach == 0.0 and cut < vals.min()
            else:
                assert cut == want_cut and np.isclose(
                    ach, wts[vals <= cut].sum() / n_null
                )
            assert np.isclose(rec, (m * (own <= cut)).sum() / m.sum())


def test_evalue_cutoff_refuses_when_reported_rows_run_out():
    try:
        lp.evalue_cutoff_recall(
            np.array([0.1]), np.array([0]), np.array([0.5]), np.array([1]), 10_000, 0.01
        )
    except ValueError:
        pass
    else:
        raise AssertionError("must refuse: the cutoff could lie among unreported rows")


def test_group_multiplicities_draw_whole_groups():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    for m in lp.group_multiplicities(groups, 50, 1):
        assert m[0] == m[1] and m[3] == m[4] == m[5]
        assert (
            sum(m[[0, 2, 3]]) == 3
        )  # three groups drawn, each item counted by its group


def test_evalue_cutoff_refuses_unsorted_or_empty_null():
    for pair_, e_ in [
        (np.array([0, 0, 0]), np.array([3.0, 1.0, 2.0])),
        (np.array([], int), np.array([])),
    ]:
        try:
            lp.evalue_cutoff_recall(
                np.array([0.1]), pair_, e_, np.array([1]), 10_000, 0.01
            )
        except ValueError:
            pass
        else:
            raise AssertionError("must refuse an unsorted or empty null")


# ---------------------------------------------------------------------------
# Notebook 346: diagonal_scan, the four scores on every diagonal of two whole proteins.
# ---------------------------------------------------------------------------


def _cell(q, t, i, j):
    """Score of cell (i, j), or None for a wall (outside the proteins or not H/P)."""
    if not (0 <= i < q.size and 0 <= j < t.size) or q[i] == GAP or t[j] == GAP:
        return None
    return 1 if q[i] == t[j] else -2


def _diag_seg(q, t, d, s, e):
    """Total of cells (s, s+d) .. (e-1, e-1+d), or None if a wall is inside."""
    tot = 0
    for i in range(s, e):
        c = _cell(q, t, i, i + d)
        if c is None:
            return None
        tot += c
    return tot


def _brute_two_anchors(q, t, i1, j1, a1, i2, j2, a2):
    """Best of one segment holding both anchors (same diagonal) or two segments, one per
    anchor, that do not overlap in either protein; plain enumeration."""
    d1, d2 = j1 - i1, j2 - i2
    best = None
    n = q.size
    if d1 == d2:
        for s in range(0, i1 + 1):
            for e in range(i2 + a2, n + 1):
                v = _diag_seg(q, t, d1, s, e)
                if v is not None and (best is None or v > best):
                    best = v
    for s1 in range(0, i1 + 1):
        for e1 in range(i1 + a1, n + 1):
            v1 = _diag_seg(q, t, d1, s1, e1)
            if v1 is None:
                continue
            for s2 in range(0, i2 + 1):
                if s2 < e1 or s2 + d2 < e1 + d1:
                    continue  # overlap in the query or in the target
                for e2 in range(i2 + a2, n + 1):
                    v2 = _diag_seg(q, t, d2, s2, e2)
                    if v2 is not None and (best is None or v1 + v2 > best):
                        best = v1 + v2
    return best


def _brute_words(q, t, a):
    return [
        (i, j)
        for i in range(q.size - a + 1)
        for j in range(t.size - a + 1)
        if all(_cell(q, t, i + y, j + y) == 1 for y in range(a))
    ]


def _brute_scan(q, t, a, seed, window, q_partner, t_partner):
    A = _brute_words(q, t, a)
    S = _brute_words(q, t, seed)
    M1 = _brute_words(q, t, 1)
    Aset = set(A)
    res = {"words6": len(A), "words8": len(S)}
    b2 = [
        max(
            v
            for s in range(0, i + 1)
            for e in range(i + seed, q.size + 1)
            if (v := _diag_seg(q, t, j - i, s, e)) is not None
        )
        for i, j in S
    ]
    res["B2"] = max(b2, default=0)
    keys = [
        (k1, k2)
        for k1 in A
        for k2 in A
        if 0 <= k2[0] - k1[0] - a <= window and 0 <= k2[1] - k1[1] - a <= window
    ]
    res["linked_count"] = len(keys)
    res["linked"] = max(
        (_brute_two_anchors(q, t, *k1, a, *k2, a) for k1, k2 in keys), default=0
    )
    rkeys = [
        ((i, j), (q_partner[i], t_partner[j]))
        for i, j in A
        if q_partner[i] >= 0
        and t_partner[j] >= 0
        and (q_partner[i], t_partner[j]) in Aset
    ]
    res["randstrobe_count"] = len(rkeys)
    res["randstrobe"] = max(
        (_brute_two_anchors(q, t, *k1, a, *k2, a) for k1, k2 in rkeys), default=0
    )
    b3 = []
    for si, sj in S:
        for mi, mj in M1:
            if 0 <= mi - si - seed <= window and 0 <= mj - sj - seed <= window:
                b3.append(_brute_two_anchors(q, t, si, sj, seed, mi, mj, 1))
            if 0 <= si - mi - 1 <= window and 0 <= sj - mj - 1 <= window:
                b3.append(_brute_two_anchors(q, t, mi, mj, 1, si, sj, seed))
    res["B3_count"] = len(b3)
    res["B3"] = max(b3, default=0)
    return res


def _related_pair(rng, nq, nt, flip=0.15, x_rate=0.02):
    """A query and a target that share a stretch on a shifted diagonal, with a few
    non-H/P letters (GAP) sprinkled in."""
    q = rng.integers(0, 2, nq).astype(np.uint8)
    t = rng.integers(0, 2, nt).astype(np.uint8)
    off = int(rng.integers(0, max(1, nt - nq + 1)))
    m = min(nq, nt - off)
    t[off : off + m] = q[:m]
    f = rng.random(nt) < flip
    t[f] = 1 - t[f]
    for arr in (q, t):
        arr[rng.random(arr.size) < x_rate] = GAP
    return q, t


def test_diagonal_scan_matches_brute_force_small():
    """Scores and counts on short proteins: every diagonal, keys across diagonals."""
    rng = np.random.default_rng(343)
    for trial in range(25):
        q, t = _related_pair(rng, int(rng.integers(14, 19)), int(rng.integers(14, 21)))
        qp = lp.randstrobe_partner(q, lp.DB_ANCHOR, lp.DB_WINDOW)
        tp = lp.randstrobe_partner(t, lp.DB_ANCHOR, lp.DB_WINDOW)
        got = lp.diagonal_scan(q, t, qp, tp)
        want = _brute_scan(q, t, lp.DB_ANCHOR, lp.DB_SEED, lp.DB_WINDOW, qp, tp)
        for k, v in want.items():
            assert got[k] == v, (trial, k, got[k], v)


def test_diagonal_scan_short_anchor_keys_across_diagonals():
    """The same scorer logic with 3-letter anchors (many keys on different diagonals), by
    patching the module constants, so cross-diagonal keys are tested on short proteins.
    """
    rng = np.random.default_rng(7)
    old = lp.DB_ANCHOR, lp.DB_SEED
    try:
        lp.DB_ANCHOR, lp.DB_SEED = 3, 4
        for trial in range(20):
            q, t = _related_pair(
                rng, int(rng.integers(10, 14)), int(rng.integers(10, 15)), flip=0.3
            )
            qp = lp.randstrobe_partner(q, 3, lp.DB_WINDOW)
            tp = lp.randstrobe_partner(t, 3, lp.DB_WINDOW)
            got = lp.diagonal_scan(q, t, qp, tp)
            want = _brute_scan(q, t, 3, 4, lp.DB_WINDOW, qp, tp)
            for k, v in want.items():
                assert got[k] == v, (trial, k, got[k], v)
    finally:
        lp.DB_ANCHOR, lp.DB_SEED = old


def test_diagonal_scan_counts_respect_the_32_residue_window():
    """Counts on proteins longer than the window, by brute enumeration of keys."""
    rng = np.random.default_rng(11)
    for trial in range(4):
        q, t = _related_pair(
            rng, int(rng.integers(70, 90)), int(rng.integers(70, 95)), flip=0.08
        )
        qp = lp.randstrobe_partner(q, lp.DB_ANCHOR, lp.DB_WINDOW)
        tp = lp.randstrobe_partner(t, lp.DB_ANCHOR, lp.DB_WINDOW)
        got = lp.diagonal_scan(q, t, qp, tp)
        A = _brute_words(q, t, lp.DB_ANCHOR)
        S = _brute_words(q, t, lp.DB_SEED)
        M1 = _brute_words(q, t, 1)
        w, a, s8 = lp.DB_WINDOW, lp.DB_ANCHOR, lp.DB_SEED
        n_linked = sum(
            1
            for k1 in A
            for k2 in A
            if 0 <= k2[0] - k1[0] - a <= w and 0 <= k2[1] - k1[1] - a <= w
        )
        n_b3 = sum(
            (0 <= mi - si - s8 <= w and 0 <= mj - sj - s8 <= w)
            + (0 <= si - mi - 1 <= w and 0 <= sj - mj - 1 <= w)
            for si, sj in S
            for mi, mj in M1
        )
        assert got["linked_count"] == n_linked
        assert got["B3_count"] == n_b3
        assert got["words6"] == len(A) and got["words8"] == len(S)


def _random_alignment(rng, n, identity=0.75, gap_rate=0.06):
    """Two aligned rows of residues (A = hydrophobic, D = polar in hp_thomas_dill2) with
    gaps, X letters and runs long enough to hold 6- and 8-letter anchors."""
    hp = np.array(list("AD"))
    q = rng.integers(0, 2, n)
    t = q.copy()
    f = rng.random(n) < (1 - identity)
    t[f] = 1 - t[f]
    qs, ts = hp[q].copy(), hp[t].copy()
    qs[rng.random(n) < gap_rate] = "-"
    ts[rng.random(n) < gap_rate] = "-"
    ts[rng.random(n) < 0.01] = "X"
    keep = ~((qs == "-") & (ts == "-"))
    return "".join(qs[keep]), "".join(ts[keep])


def test_diagonal_scan_restricted_to_an_alignment_equals_score_pair():
    """With the alignment's cells as the mask, diagonal_scan gives score_pair's B2, B3,
    chosen linked pair and randstrobe, scores and key counts (gates 1 and 2 rely on it).
    """
    rng = np.random.default_rng(2026)
    name = lp.setting_name("all", 6, 32, "none")
    rname = lp.setting_name("rand", 6, 32, "none")
    for trial in range(30):
        qaln, taln = _random_alignment(rng, int(rng.integers(60, 140)))
        ref = lp.score_pair(qaln, taln, np.random.default_rng(0), n_shuffle=0)[0]
        q_seq, t_seq, allowed = lp.alignment_cells(qaln, taln)
        got = lp.scan_sequences(q_seq, t_seq, allowed)
        assert got["B2"] == ref["B2_ext"], trial
        assert got["B3"] == ref["B3_ext"], trial
        assert got["linked"] == ref[f"{name}_ext"], trial
        assert got["linked_count"] == ref[f"{name}_count"], trial
        assert got["randstrobe"] == ref[f"{rname}_ext"], trial
        assert got["randstrobe_count"] == ref[f"{rname}_count"], trial
        assert (
            got["words6"] == ref["seed6_count"] and got["words8"] == ref["seed8_count"]
        )


def test_alignment_rows_from_pairs_round_trip():
    """Rows built from aligned pairs pair exactly those residues, and alignment_cells reads
    the same pairs back."""
    q, t = "ACDEFGHIKLMN", "PQRSTVWYACDE"
    pq, pt = [2, 3, 6, 7, 11], [1, 4, 5, 6, 9]
    qrow, trow = lp.alignment_rows_from_pairs(q, t, pq, pt)
    assert len(qrow) == len(trow)
    assert qrow.replace("-", "") == q[1:11] and trow.replace("-", "") == t[0:9]
    pairs = []
    iq, it = pq[0] - 1, pt[0] - 1
    for a, b in zip(qrow, trow):
        if a != "-" and b != "-":
            pairs.append((iq + 1, it + 1))
        iq += a != "-"
        it += b != "-"
    assert pairs == list(zip(pq, pt))
    _, _, allowed = lp.alignment_cells(qrow, trow)
    assert sorted(zip(*np.nonzero(allowed))) == [
        (a - pq[0], b - pt[0]) for a, b in zip(pq, pt)
    ]


def test_forward_max_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, w = int(rng.integers(1, 120)), int(rng.integers(1, 40))
        x = rng.integers(-50, 50, (n, 7))
        want = np.array([[x[i : i + w, j].max() for j in range(7)] for i in range(n)])
        assert (lp._forward_max(x, w, 0) == want).all()
        assert (lp._forward_max(x.T, w, 1) == want.T).all()


def test_best_keys_with_segments_add_up():
    """The segments returned for the best linked and B3 keys hold both anchors, do not
    overlap in either protein, contain no wall, and their cells add up to the score, which
    equals diagonal_scan's."""
    rng = np.random.default_rng(12)
    hp = np.array(list("AD"))
    for trial in range(30):
        q, t = _related_pair(
            rng, int(rng.integers(50, 90)), int(rng.integers(50, 90)), flip=0.15
        )
        qs = "".join(np.where(q == GAP, "X", hp[np.minimum(q, 1)]))
        ts = "".join(np.where(t == GAP, "X", hp[np.minimum(t, 1)]))
        got = lp.scan_sequences(qs, ts)
        best = lp.best_keys_with_segments(qs, ts)
        for name in ["linked", "B3"]:
            b = best[name]
            if b is None:
                assert got[name] == 0
                continue
            assert b["score"] == got[name]
            total = 0
            for qs0, qe, ts0 in b["segments"]:
                cells = [_cell(q, t, i, ts0 + i - qs0) for i in range(qs0, qe)]
                assert None not in cells
                total += sum(cells)
            assert total == b["score"], (trial, name)
            segs = b["segments"]
            if len(segs) == 2:
                (a0, a1_, at), (b0, b1, bt) = segs
                assert a1_ <= b0 and at + (a1_ - a0) <= bt
            for qa, ta, a in [(b["q1"], b["t1"], b["a1"]), (b["q2"], b["t2"], b["a2"])]:
                assert any(
                    s0 <= qa and qa + a <= e and ta - qa == t0 - s0
                    for s0, e, t0 in segs
                )


def test_best_keys_with_segments_small_chunk():
    """A chunk below 100 still searches every first anchor (it used to search none)."""
    q = "A" * 10 + "D" * 10 + "A" * 10
    want = lp.scan_sequences(q, q)["linked"]
    for chunk in [1, 50, 100, 20_000]:
        assert lp.best_keys_with_segments(q, q, chunk=chunk)["linked"]["score"] == want
