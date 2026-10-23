"""Tests for nextflow-runs/shared/decoys.py on human BCL2 and C. elegans CED-9."""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "nextflow-runs" / "shared"))
from decoys import (  # noqa: E402
    dipeptide_shuffle,
    full_shuffle,
    shuffle_around_fixed,
)

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


def test_full_shuffle_keeps_only_the_residue_counts():
    rng = random.Random(0)
    seq = "CKGKGAKCSRLMYDCCTGSCRSGKC"
    out = full_shuffle(seq, rng)
    assert sorted(out) == sorted(seq)
    assert len(out) == len(seq)
    # Over many draws the cysteines land somewhere other than their real positions, which
    # is the point of this control: it breaks the framework.
    real = [i for i, r in enumerate(seq) if r == "C"]
    moved = 0
    for _ in range(50):
        shuffled = full_shuffle(seq, rng)
        if [i for i, r in enumerate(shuffled) if r == "C"] != real:
            moved += 1
    assert moved >= 45, moved


def test_shuffle_around_fixed_keeps_every_cysteine_in_place():
    rng = random.Random(0)
    seq = "CKGKGAKCSRLMYDCCTGSCRSGKC"
    for _ in range(50):
        out = shuffle_around_fixed(seq, rng)
        assert len(out) == len(seq)
        assert sorted(out) == sorted(seq)
        assert [i for i, r in enumerate(out) if r == "C"] == [
            i for i, r in enumerate(seq) if r == "C"
        ]


def test_shuffle_around_fixed_does_move_the_other_residues():
    rng = random.Random(1)
    seq = "CKGKGAKCSRLMYDCCTGSCRSGKC"
    assert any(shuffle_around_fixed(seq, rng) != seq for _ in range(50))


def test_shuffle_around_fixed_on_an_all_cysteine_sequence_is_unchanged():
    assert shuffle_around_fixed("CCCC", random.Random(0)) == "CCCC"
