"""Bits per letter and per k-mer for every kmerseek alphabet."""

import math
import re
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
        assert len(clusters) == int(re.search(r"(\d+)$", alphabet).group(1))


@pytest.mark.parametrize("alphabet", ou.KMERSEEK_ALPHABET_CLUSTERS)
def test_registry_symbols_are_single_characters(alphabet):
    groups = ou.ALPHABET_AA_GROUPS[alphabet]
    assert all(len(symbol) == 1 for symbol in groups)
    assert len(groups) == len(ou.KMERSEEK_ALPHABET_CLUSTERS[alphabet])


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


def test_swissprot_frequencies_match_the_counted_table():
    table = Path(__file__).resolve().parents[1] / (
        "notebooks/363_data/swissprot_2026_03_amino_acid_counts.tsv"
    )
    rows = [line.split("\t") for line in table.read_text().splitlines()[1:]]
    assert {
        aa: float(frequency) for aa, _, frequency in rows
    } == ou.SWISSPROT_AA_FREQUENCIES
    assert set(ou.SWISSPROT_AA_FREQUENCIES) == set(ou.AA_FREQUENCIES)
    assert sum(ou.SWISSPROT_AA_FREQUENCIES.values()) == pytest.approx(1, abs=1e-4)


def test_frequencies_argument_changes_the_answer():
    frequencies = ou.SWISSPROT_AA_FREQUENCIES
    cysteine = frequencies["C"] / sum(frequencies.values())
    other = 1 - cysteine
    expected = -cysteine * math.log2(cysteine) - other * math.log2(other)
    assert ou.entropy_per_symbol("cys2", frequencies) == pytest.approx(expected)
    assert ou.entropy_per_symbol("cys2", frequencies) != ou.entropy_per_symbol("cys2")
    assert ou.bits_per_kmer("cys2", 10, frequencies) == pytest.approx(10 * expected)
    assert ou.ksize_for_bits("cys2", 30, frequencies) == round(30 / expected)


def test_equal_frequencies_give_log2_of_the_class_count_for_protein():
    equal = {aa: 0.05 for aa in ou.AA_FREQUENCIES}
    assert ou.entropy_per_symbol("protein20", equal) == pytest.approx(math.log2(20))


def test_ksize_for_bits_per_letter_matches_ksize_for_bits():
    for alphabet in ou.KMERSEEK_ALPHABET_CLUSTERS:
        per_letter = ou.entropy_per_symbol(alphabet)
        assert ou.ksize_for_bits_per_letter(29.2, per_letter) == ou.ksize_for_bits(alphabet, 29.2)
    assert ou.ksize_for_bits_per_letter(0.1, 4.0) == 1
