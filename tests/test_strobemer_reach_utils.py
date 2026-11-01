"""Brute-force checks of the key builders and counter in notebooks/strobemer_reach_utils.py."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import linked_pairs_utils as lp  # noqa: E402
import strobemer_reach_utils as sr  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand_seq(rng, n, unknown=0.0):
    s = rng.choice(list(AA), n)
    if unknown:
        s[rng.random(n) < unknown] = "X"
    return "".join(s)


def _classes_str(seq, alphabet):
    """Each residue's class as a letter, '*' for unknown: the string the tests compare."""
    m = {r: chr(65 + i) for i, cl in enumerate(sr.CLUSTERS[alphabet]) for r in cl}
    return "".join(m.get(c, "*") for c in seq)


def _brute_pair_keys(seq, alphabet, a, family):
    """Keys as tuples of strings, from the class string alone, by the definition."""
    s = _classes_str(seq, alphabet)
    keys = set()
    gaps = range(0, 33) if family == "pair_nogap" else range(1, 41)
    for i in range(len(s) - a + 1):
        w1 = s[i : i + a]
        if "*" in w1:
            continue
        for g in gaps:
            j = i + a + g
            w2 = s[j : j + a]
            if len(w2) < a or "*" in w2:
                continue
            gf = {"pair_gap": g, "pair_gapbin3": g // 3, "pair_nogap": None}[family]
            keys.add((w1, gf, w2))
    return keys


@pytest.mark.parametrize("family", ["pair_gap", "pair_gapbin3", "pair_nogap"])
@pytest.mark.parametrize(
    "alphabet,a",
    [("hp_thomas_dill2", 3), ("thr2", 4), ("dayhoff6", 3), ("protein20", 2)],
)
def test_pair_keys_match_brute_force(family, alphabet, a):
    rng = np.random.default_rng(1)
    for n in [5, 20, 60, 130]:
        seq = _rand_seq(rng, n, unknown=0.03)
        brute = _brute_pair_keys(seq, alphabet, a, family)
        got = sr.sequence_keys(seq, alphabet, a, family)
        assert len(got) == len(brute)
        # every brute key maps to exactly one integer key, and back
        s = _classes_str(seq, alphabet)
        decoded = set()
        for occ in got.values():
            i, g, j = occ[0]
            gf = {"pair_gap": g, "pair_gapbin3": g // 3, "pair_nogap": None}[family]
            decoded.add((s[i : i + a], gf, s[j : j + a]))
            # all occurrences of one integer key spell the same words
            for i2, g2, j2 in occ:
                assert s[i2 : i2 + a] == s[i : i + a] and s[j2 : j2 + a] == s[j : j + a]
        assert decoded == brute


@pytest.mark.parametrize("family", sr.FAMILIES + ["kmer"])
@pytest.mark.parametrize("alphabet,a", [("hp_pbotc_1st_ed2", 6), ("dayhoff6", 4)])
def test_concatenated_counter_equals_one_sequence_at_a_time(family, alphabet, a):
    rng = np.random.default_rng(2)
    seqs = [_rand_seq(rng, n, unknown=0.02) for n in [3, 40, 90, 250, 31, 500]]
    s_ids, keys = sr.all_keys(seqs, alphabet, a, family)
    for i, seq in enumerate(seqs):
        want = sorted(sr.sequence_keys(seq, alphabet, a, family))
        assert keys[s_ids == i].tolist() == want


def test_shared_counts_brute_force():
    rng = np.random.default_rng(3)
    # A low-entropy alphabet so that entries really share keys with the queries.
    entries = [_rand_seq(rng, n) for n in [50, 80, 120, 10, 200]]
    queries = [_rand_seq(rng, n) for n in [60, 150]] + [entries[2]]
    for family in sr.FAMILIES:
        s_ids, keys = sr.all_keys(entries + queries, "hp_thomas_dill2", 4, family)
        S, n_keys = sr.shared_counts(len(entries), len(queries), s_ids, keys)
        ks = [set(sr.sequence_keys(x, "hp_thomas_dill2", 4, family)) for x in entries]
        kq = [set(sr.sequence_keys(x, "hp_thomas_dill2", 4, family)) for x in queries]
        for e in range(len(entries)):
            for q in range(len(queries)):
                assert S[e, q] == len(ks[e] & kq[q])
            assert n_keys[e] == len(ks[e])
        # a query against itself shares all its keys
        assert S[2, 2] == len(kq[2])


def test_general_word_codes_equal_notebook_294_on_two_letters():
    rng = np.random.default_rng(4)
    cls = rng.integers(0, 2, 300).astype(np.uint8)
    cls[rng.random(300) < 0.05] = sr.GAP
    for a in [3, 6, 10]:
        c1, ok1 = lp.word_codes(cls, a)
        c2, ok2 = sr._word_codes_general(cls, a, 2)
        assert (ok1 == ok2).all()
        assert (c1[ok1] == c2[ok2]).all()
        for g in [8, 32]:
            assert (
                lp.randstrobe_partner(cls, a, g) == sr._randstrobe_general(cls, a, g, 2)
            ).all()


def test_randstrobe_partner_is_the_argmin():
    rng = np.random.default_rng(5)
    cls = rng.integers(0, 6, 200).astype(np.uint8)
    a, w = 3, 32
    sel = sr.randstrobe_partner_any(cls, a, w, 6)
    codes, ok = sr.word_codes_any(cls, a, 6)
    h = (lp._mix64(codes) % np.uint64(997)).astype(int)
    for i in range(codes.size):
        cands = [j for j in range(i + a, min(i + a + w, codes.size - 1) + 1) if ok[j]]
        if not cands:
            assert sel[i] == -1
            continue
        best = min(cands, key=lambda j: ((h[i] + h[j]) % 997, j))
        assert sel[i] == best


def test_long_kmer_hash_matches_strings():
    """Exact k-mers longer than 2**62 allows are hashed; distinct words stay distinct."""
    rng = np.random.default_rng(6)
    seq = _rand_seq(rng, 400)
    for k in [20, 40]:
        got = sr.sequence_keys(seq, "protein20", k, "kmer")
        words = {seq[i : i + k] for i in range(len(seq) - k + 1)}
        assert len(got) == len(words)


def test_null_c_keeps_each_phase():
    rng = np.random.default_rng(7)
    seq = _rand_seq(rng, 103)
    for p in [1, 3, 11, 15]:
        out = sr.null_c(seq, p, np.random.default_rng(p))
        assert len(out) == len(seq)
        for r in range(p):
            assert Counter(out[r::p]) == Counter(seq[r::p])
    # a perfect repeat of period p is unchanged by construction
    rep = "AATAATAATAAT"
    assert sr.null_c(rep, 3, np.random.default_rng(0)) == rep
    # period 1 is an ordinary shuffle and does change a varied sequence
    assert sr.null_c(seq, 1, np.random.default_rng(0)) != seq


def test_selection_requires_all_four_distinct_queries():
    import polars as pl

    rows = [
        dict(
            alphabet="protein",
            word_length=4,
            family="pairs",
            kind="real",
            min_shared=2,
            lineage=lineage,
            reach=0.5,
            random_share=0.005,
        )
        for lineage in sr.SELECT_LINEAGES
    ]
    assert sr.selection_table(pl.DataFrame(rows))["eligible"][0]
    assert not sr.selection_table(pl.DataFrame(rows[:3]))["eligible"][0]
    assert not sr.selection_table(pl.DataFrame(rows[:3] + [rows[0]]))["eligible"][0]
    rows[-1]["random_share"] = None
    assert not sr.selection_table(pl.DataFrame(rows))["eligible"][0]
