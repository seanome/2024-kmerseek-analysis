"""Tests for nextflow-runs/shared/decoys.py on human BCL2 and C. elegans CED-9."""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "nextflow-runs" / "shared"))
from decoys import composition_shuffle, dipeptide_shuffle, window_shuffle  # noqa: E402

FASTA = REPO / "data" / "fig1_ced9_bcl2" / "ced9_P41958_bcl2_P10415.fasta"


def read_fasta(path: Path) -> dict[str, str]:
    """Sequences keyed by UniProt accession (the second field of a sp|...| header)."""
    seqs, acc = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            acc = line.split("|")[1]
            seqs[acc] = ""
        else:
            seqs[acc] += line.strip()
    return seqs


SEQS = read_fasta(FASTA)
BCL2 = SEQS["P10415"]
CED9 = SEQS["P41958"]


def pair_counts(seq: str) -> Counter:
    return Counter(zip(seq, seq[1:]))


@pytest.mark.parametrize(
    "seq", [BCL2, CED9, "ACACACAC", "AAAAB", "MKKLLPTAAAGLLLLAAQPAMA"]
)
@pytest.mark.parametrize("seed", range(20))
def test_keeps_ends_and_every_adjacent_pair(seq, seed):
    shuffled = dipeptide_shuffle(seq, random.Random(seed))
    assert shuffled[0] == seq[0]
    assert shuffled[-1] == seq[-1]
    assert pair_counts(shuffled) == pair_counts(seq)
    assert Counter(shuffled) == Counter(seq)


def test_reorders_a_real_protein():
    """BCL2 has many orders with its pair counts; 20 seeds give 20 different ones."""
    shuffles = {dipeptide_shuffle(BCL2, random.Random(seed)) for seed in range(20)}
    assert BCL2 not in shuffles
    assert len(shuffles) == 20


@pytest.mark.parametrize("seq", ["", "M", "MK"])
def test_too_short_to_shuffle_is_returned_as_is(seq):
    assert dipeptide_shuffle(seq, random.Random(0)) == seq


# Written by the copy in notebooks/256_bhf_dipeptide_shuffle_null.py on
# olgabot/bhf-dipeptide-shuffle-null, so a branch that switches to the shared module gets
# the same decoys it made before.
BCL2_SHUFFLED = {
    0: "MSSQDNGAGHLFSPSVGAAASLQAFDNIVNRDAAPHTPFTEMAFRQTWDFESWLWDDFVDAHYDFSRREFSQLYLSMSPARPGGD"
    "NRDPGIAGREITPPAAGPVVHLTLGYIHLVNWMCVEYGHKYRIVGWGYLHAGAPGALSRGRGGPGVMTPLVVALGVMRTSQPLKTA"
    "TVPPAPVEEMKLRTLSLLAYEWIFAARRFGDVELFFAELSSPACIQRDGRHTLTGALHPVAAPLNRYK",
    1: "MAACIAALAAYEMCVGAAAGDNIVDFAGGRTPVMKYLSMTGDVEITLQLHTPGRIFDNRYLYDALNWDNRGAFFAPAPARQPVNR"
    "DAPAAHTLGASQREWIHLGHAPFVVEFGYIVAGAGPPLSQTVMSLTPLKTSLFSRTLLSQAFTEYRRRFSRPLFEESSPGPSSPVAL"
    "VGWGRDDPGVNGHLVVELTAEMRHYKLRGVPGYGGARDGIQDFSPHPPATWLWMSWDFRELHLSVHK",
    2024: "MTLSPAYRDFGGIQTGALSREFDGAEMKTLSRTLHAGDNWGPGVELHKYDFRPPHLQPAAGRRQLTPVAALRDDNRTVPLALWI"
    "FSLLFSMSSQAPVEIHLGYLTSLSSVMSPAGRIAAREMAAARYIVMRGGVGHTAGAPASWLYGHTPLNIVVAFAAPGWDAHYESQRR"
    "GAPLGPPGAFVVDNGDFSQDPSPFFEYLKLVGRDVNRFTEEWDATWMCITPVELVNRHLFACVHPGYK",
}


@pytest.mark.parametrize("seed", sorted(BCL2_SHUFFLED))
def test_fixed_seed_gives_fixed_output(seed):
    assert dipeptide_shuffle(BCL2, random.Random(seed)) == BCL2_SHUFFLED[seed]


@pytest.mark.parametrize("seq", [BCL2, CED9, "AAAAB"])
@pytest.mark.parametrize("seed", range(5))
def test_composition_shuffle_keeps_only_the_residue_counts(seq, seed):
    assert Counter(composition_shuffle(seq, random.Random(seed))) == Counter(seq)


def test_composition_shuffle_breaks_pairs_dipeptide_shuffle_keeps():
    """On BCL2, 20 plain shuffles never keep every adjacent pair count."""
    for seed in range(20):
        assert pair_counts(
            composition_shuffle(BCL2, random.Random(seed))
        ) != pair_counts(BCL2)


@pytest.mark.parametrize("seq", [BCL2, CED9, "MKKLLPTAAAGLLLLAAQPAMA"])
@pytest.mark.parametrize("window", [1, 10, 37])
def test_window_shuffle_keeps_each_window(seq, window):
    shuffled = window_shuffle(seq, random.Random(0), window)
    assert len(shuffled) == len(seq)
    for i in range(0, len(seq), window):
        assert Counter(shuffled[i : i + window]) == Counter(seq[i : i + window])


def test_window_of_one_changes_nothing():
    assert window_shuffle(BCL2, random.Random(0), 1) == BCL2


def test_window_shuffle_reorders_bcl2():
    assert window_shuffle(BCL2, random.Random(0), 10) != BCL2


def test_window_shuffle_matches_pr114_copy():
    """Same draws as the PR 114 / PR 134 copy, which shuffles a list per window."""
    rng = random.Random(7)
    expected = []
    for i in range(0, len(BCL2), 10):
        w = list(BCL2[i : i + 10])
        rng.shuffle(w)
        expected.extend(w)
    assert window_shuffle(BCL2, random.Random(7), 10) == "".join(expected)


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        window_shuffle(BCL2, random.Random(0), 0)
