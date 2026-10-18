"""Checks of bin/novelty.py against brute force on small inputs. Run: pytest tests/"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import novelty as nv  # noqa: E402

RNG = np.random.default_rng(343)


def random_seqs(n, lo=5, hi=60, alphabet=nv.AMINO_ACIDS + "X"):
    return ["".join(RNG.choice(list(alphabet), RNG.integers(lo, hi))) for _ in range(n)]


def brute_words(seqs, alphabet, k):
    groups = nv.ALPHABETS[alphabet]
    code = {aa: i for i, aas in enumerate(groups.values()) for aa in aas}
    base = len(groups)
    out = []
    for p, s in enumerate(seqs):
        for st in range(len(s) - k + 1):
            win = s[st : st + k]
            if all(c in code for c in win):
                out.append((sum(code[c] * base ** (k - 1 - j) for j, c in enumerate(win)), p, st))
    return out


@pytest.mark.parametrize("alphabet,k", [("protein", 3), ("protein", 10), ("hp_pbotc_1st_ed", 7), ("hp_pbotc_1st_ed", 42)])
def test_encode_matches_brute_force(alphabet, k):
    seqs = random_seqs(30, hi=80)
    w, p, s = nv.encode_words(seqs, alphabet, k)
    assert list(zip(w.tolist(), p.tolist(), s.tolist())) == brute_words(seqs, alphabet, k)


def test_hp_groups_match_shared_utils():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
    import ortholog_analysis_utils as ou

    assert nv.ALPHABETS["hp_pbotc_1st_ed"] == ou.ALPHABET_AA_GROUPS["hp_pbotc_1st_ed"]


def write(tmp, name, seqs):
    f = tmp / name
    nv.write_fasta(f, [f"p{i}" for i in range(len(seqs))], seqs)
    return f


def test_masks_match_sets(tmp_path):
    # Short words in a small alphabet so proteomes share many of them.
    proteomes = [random_seqs(20, alphabet="ACDEG") for _ in range(5)]
    fastas = [write(tmp_path, f"x{i}.fa.gz", s) for i, s in enumerate(proteomes)]
    words, masks, _ = nv.clade_masks(fastas, "protein", 4, 1)
    sets = [{w for w, _, _ in brute_words(s, "protein", 4)} for s in proteomes]
    expect = {}
    for i, st in enumerate(sets):
        for w in st:
            expect[w] = expect.get(w, 0) | (1 << i)
    assert dict(zip(words.tolist(), masks.tolist())) == expect


def test_expected_curve_is_average_over_all_orderings():
    size = 5
    masks = RNG.integers(1, 2**size, 40).astype(np.uint64)
    n = RNG.integers(1, 9, 40).astype(float)
    c = nv.hist_in_subset(masks, n, 2**size - 1, size)
    got = nv.expected_recurring(c, size)
    perms = list(itertools.permutations(range(size)))
    want = np.zeros(size)
    for perm in perms:
        for t in range(1, size + 1):
            sub = sum(1 << i for i in perm[:t])
            want[t - 1] += n[nv.popcount(masks & np.uint64(sub)) >= 2].sum()
    np.testing.assert_allclose(got, want / len(perms))


def test_heaps_fit_recovers_parameters():
    n = np.arange(1, 21)
    u = 1234.0 * n**0.37
    k, g = nv.heaps_fit(u, 20)
    assert k == pytest.approx(1234.0) and g == pytest.approx(0.37)


def test_splits_are_fixed_and_hide_20_percent():
    a = nv.splits_for("Archaea", 7, 200)
    assert a == nv.splits_for("Archaea", 7, 200)
    assert len(a) == math.comb(7, 5) * 2  # every distinct split, fewer than 200
    assert all(len(obs) == 5 and nxt not in obs for obs, nxt in a)
    b = nv.splits_for("Bacteria", 23, 200)
    assert len(b) == 200 and len(set(b)) == 200 and all(len(o) == 18 for o, _ in b)


def test_true_added_counts_words_held_once_before():
    # Words: A in {0,1}, B in {0,2}, C in {2}, D in {1,2,3}. Observed {0,1}, next 2.
    masks = np.array([0b0011, 0b0101, 0b0100, 0b1110], dtype=np.uint64)
    n = np.ones(4)
    assert nv.true_added(masks, n, 0b0011, 2) == 2  # B (held by 0) and D (held by 1) become recurring; C is still single


def test_null_keeps_lengths_and_composition(tmp_path):
    seqs = random_seqs(200, lo=50, hi=400, alphabet="AAAALLLKKEW")
    f = write(tmp_path, "real.fa.gz", seqs)
    out = tmp_path / "null.fa.gz"
    nv.cmd_null(type("A", (), {"fasta": f, "seed": 1, "out": out})())
    names, null = nv.read_fasta(out)
    assert [len(s) for s in null] == [len(s) for s in seqs]
    real_c = {c: "".join(seqs).count(c) / sum(map(len, seqs)) for c in "ALKEW"}
    null_c = {c: "".join(null).count(c) / sum(map(len, null)) for c in "ALKEW"}
    for c in "ALKEW":
        assert null_c[c] == pytest.approx(real_c[c], abs=0.02)
    assert set("".join(null)) <= set("ALKEW")


def test_examples_returns_the_shared_stretch(tmp_path):
    shared = "MKVLAAGIWQERTYPLHDNSCFEKLM" * 2
    a = random_seqs(3, alphabet="DEKRS") + ["DDDD" + shared + "EEEE"]
    b = random_seqs(3, alphabet="GHNQT") + ["KK" + shared]
    c = random_seqs(4, alphabet="DEKRSGH")
    fastas = [write(tmp_path, f"{x}.fa.gz", s) for x, s in zip("abc", [a, b, c])]
    pairs = tmp_path / "pairs.parquet"
    words, masks, _ = nv.clade_masks(fastas, "protein", 8, 1)
    two = nv.popcount(masks) == 2
    pl.DataFrame({"word": words[two], "mask": masks[two]}).write_parquet(pairs)
    out = tmp_path / "ex.parquet"
    args = type("A", (), dict(fastas=[str(f) for f in fastas], proteomes=["a", "b", "c"], clade="t",
                              specimen="a", pairs=pairs, alphabet="protein", ksize=8, top=1, out=out))()
    nv.cmd_examples(args)
    ex = pl.read_parquet(out).row(0, named=True)
    assert ex["partner"] == "b" and ex["specimen_protein"] == "p3" and ex["partner_protein"] == "p3"
    assert ex["specimen_start"] == 5 and ex["partner_start"] == 3
    assert ex["specimen_seq"] == ex["partner_seq"] == shared
    assert (ex["specimen_length"], ex["partner_length"]) == (len(a[3]), len(b[3]))
