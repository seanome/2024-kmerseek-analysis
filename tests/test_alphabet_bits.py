"""Bits per letter and per k-mer for every kmerseek alphabet."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import ortholog_analysis_utils as ou  # noqa: E402


@pytest.mark.parametrize("alphabet", ou.KMERSEEK_ALPHABET_CLUSTERS)
def test_every_residue_is_in_exactly_one_class(alphabet):
    residues = "".join(ou.KMERSEEK_ALPHABET_CLUSTERS[alphabet])
    assert sorted(residues) == sorted(ou.AA_FREQUENCIES)


def test_class_counts_match_the_number_in_each_name():
    for alphabet, clusters in ou.KMERSEEK_ALPHABET_CLUSTERS.items():
        assert len(clusters) == int(alphabet[-2:].lstrip("abcdefghijklmnopqrstuvwxyz_"))


def test_merging_residues_can_only_lose_bits():
    protein = ou.entropy_per_symbol("protein20")
    for alphabet in ou.KMERSEEK_ALPHABET_CLUSTERS:
        n_classes = len(ou.KMERSEEK_ALPHABET_CLUSTERS[alphabet])
        assert ou.entropy_per_symbol(alphabet) <= protein + 1e-9
        assert ou.entropy_per_symbol(alphabet) <= math.log2(n_classes) + 1e-9


def test_cys2_by_hand():
    cysteine = ou.AA_FREQUENCIES["C"] / sum(ou.AA_FREQUENCIES.values())
    other = 1 - cysteine
    expected = -cysteine * math.log2(cysteine) - other * math.log2(other)
    assert ou.entropy_per_symbol("cys2") == pytest.approx(expected)


def test_ksize_for_bits_picks_the_nearest_k():
    assert ou.ksize_for_bits("protein20", 7 * ou.entropy_per_symbol("protein20")) == 7
    assert ou.ksize_for_bits("hp_lehninger2", 29.2) == 29
    assert ou.ksize_for_bits("protein20", 0.1) == 1
