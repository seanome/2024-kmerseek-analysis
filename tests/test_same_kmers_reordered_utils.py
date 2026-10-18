"""Tests for notebooks/same_kmers_reordered_utils.py against brute-force enumeration."""

import random
import sys
from collections import Counter
from itertools import permutations
from math import exp
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
import same_kmers_reordered_utils as u  # noqa: E402


def brute_force_strings(s: str, k: int) -> set[str]:
    """Every string with the same length-k word counts and the same first k-1 letters."""
    target = u.kmer_counts(s, k)
    return {
        "".join(p)
        for p in set(permutations(s))
        if "".join(p[: k - 1]) == s[: k - 1] and u.kmer_counts("".join(p), k) == target
    }


CASES = [
    ("hhpphphhpp", 3),
    ("hphphphhpp", 2),
    ("hhhhpppphh", 3),
    ("hphhphpphh", 4),
    ("abcabdcab", 2),
    ("hppphhphph", 3),
    ("hhhhhhhh", 3),
]


@pytest.mark.parametrize("s,k", CASES)
def test_count_matches_brute_force(s, k):
    g = u.WordGraph(s, k)
    assert round(exp(g.log_n_strings())) == len(brute_force_strings(s, k))


@pytest.mark.parametrize("s,k", CASES)
def test_draws_are_uniform_and_exact(s, k):
    truth = brute_force_strings(s, k)
    g = u.WordGraph(s, k)
    rng = random.Random(1)
    n = 3000 * len(truth)
    draws = Counter(g.sample(rng) for _ in range(n))
    assert set(draws) == truth
    expected = n / len(truth)
    for c in draws.values():
        assert abs(c - expected) < 5 * expected**0.5


def test_realize_protein_keeps_composition_and_encodes():
    protein = "MAKLVDEFGHIKLMNPQRSTVWYAAKLLE"
    red = u.encode(protein, "hp_pbotc_1st_ed2")
    g = u.WordGraph(red, 4)
    rng = random.Random(3)
    for _ in range(50):
        alt = g.sample(rng)
        aa = u.realize_protein(protein, red, alt)
        assert Counter(aa) == Counter(protein)
        assert u.encode(aa, "hp_pbotc_1st_ed2") == alt
        assert aa[:3] == protein[:3] and aa[-3:] == protein[-3:]
