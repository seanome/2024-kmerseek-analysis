"""Tests for notebooks/kmer_offset_utils.py (notebook 293, the k-mer offset test)."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import kmer_offset_utils as ku  # noqa: E402


def _random_protein(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(ku.CANONICAL_AA) for _ in range(n))


def test_circular_permutation_is_called_with_its_blocks():
    a = _random_protein(240, 1)
    b = a[100:] + a[:100]
    row = ku.offset_test(a, b, "protein20", 5, [3])[0]
    assert row["called"]
    # Block 1 comes first along A: A 0-100 sits at B 140-240.
    assert (row["block1_a_start"], row["block1_a_end"]) == (0, 100)
    assert (row["block1_b_start"], row["block1_b_end"]) == (140, 240)
    assert (row["block2_a_start"], row["block2_a_end"]) == (100, 240)
    assert (row["block2_b_start"], row["block2_b_end"]) == (0, 140)
    assert row["block1_identical"] == row["block1_compared"] == 100


def test_domain_swap_with_linker_is_called():
    x, y, linker = _random_protein(120, 2), _random_protein(90, 3), "GSGSGS"
    row = ku.offset_test(x + linker + y, y + x, "protein20", 6, [5])[0]
    assert row["called"]


def test_colinear_pair_with_an_insertion_is_not_called():
    a = _random_protein(240, 4)
    b = a[:120] + "W" * 15 + a[120:]
    for alphabet, k in ku.all_settings():
        assert not ku.offset_test(a, b, alphabet, k, [3])[0]["called"], (alphabet, k)


def test_shuffled_copy_keeps_composition_and_is_not_called():
    a = _random_protein(240, 6)
    b = ku.shuffled(a, 0)
    assert sorted(b) == sorted(a) and b != a
    assert ku.shuffled(a, 0) == b  # fixed seed
    assert not ku.offset_test(a, b, "protein20", 5, [3])[0]["called"]


def test_low_complexity_kmers_are_dropped():
    a = "Q" * 40 + _random_protein(60, 5)
    hits, dropped = ku.shared_kmer_hits(a, a, "protein20", 5)
    assert dropped == 1  # QQQQQ occurs 36 times
    assert all(a[i : i + 5] != "QQQQQ" for i, _ in hits)


def test_encoding_matches_sourmash_tables():
    import sourmash

    lib = sourmash._lowlevel.lib
    for alphabet, fn in [
        ("dayhoff6", lib.sourmash_aa_to_dayhoff),
        ("hp_lehninger2", lib.sourmash_aa_to_hp),
    ]:
        expected = "".join(fn(a.encode()).decode() for a in ku.CANONICAL_AA)
        assert ku.encode(ku.CANONICAL_AA, alphabet) == expected
    assert ku.encode("AXA", "protein20") == "A*A"


def test_gapless_identity_reads_b_at_the_block_offset():
    a = "MKTAYIAKQR"
    b = "GG" + "MKTAYIAKQR"
    blk = ku.Block(2, 2, 3, 0, 10, 2, 12, 2)
    assert ku.gapless_identity(a, b, blk) == (10, 10, 2, 12)
