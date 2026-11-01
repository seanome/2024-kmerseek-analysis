"""Tests for bin/lookup.py: shuffles, normalisation, the exact index, and the check."""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import lookup  # noqa: E402

# A test sequence typed to resemble serum albumin; not checked against UniProt, only used as a target.
ALBUMIN = "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"


def test_normalise_merges_il_and_qk():
    assert lookup.normalise("IQLK") == "LKLK"


def test_composition_shuffle_keeps_residues():
    rng = random.Random(1)
    s = "PEPTIDEKQLLR"
    out = lookup.composition_shuffle(s, rng)
    assert Counter(out) == Counter(s)


def test_dipeptide_shuffle_keeps_pairs_and_ends():
    rng = random.Random(2)
    s = "AAKLEGLLKDVAEK"
    for _ in range(20):
        out = lookup.dipeptide_shuffle(s, rng)
        assert Counter(zip(out, out[1:])) == Counter(zip(s, s[1:]))
        assert out[0] == s[0] and out[-1] == s[-1]


def test_exact_lookup_finds_il_and_qk_readings():
    db = {"ALB": ALBUMIN, "OTHER": "MSTNPKPQRKTKRNTNRRPQDVKFPGG"}
    index = lookup.build_exact_index(db)
    # ALBUMIN holds LVLIAFAQYLQQ; read with I as L and Q as K it must still be found.
    hits = lookup.exact_lookup({"q1": "LVLLAFAKYLKK", "q2": "WWWWWWW"}, db, index)
    assert hits["q1"] == {"ALB"}
    assert "q2" not in hits


def test_align_coordinates_are_one_based_and_match_the_residues():
    pep = "LVLLAFAQYL"  # I read as L at the 4th residue
    a = lookup.align(pep, ALBUMIN)
    start = ALBUMIN.index("LVLIAFAQYL")
    assert a["target_start"] == start + 1
    assert a["target_end"] == start + len(pep)
    assert ALBUMIN[a["target_start"] - 1:a["target_end"]] == a["target_line"]
    assert a["n_columns"] == 10 and a["n_identical"] == 9
    assert a["match_line"] == "|||:||||||"


def test_align_with_a_gap():
    # Drop one residue from the target stretch: the alignment needs a gap in the peptide's
    # partner, and the counts include that column.
    target = "MMMMM" + "AFAQYLQQCPFEDHVK" + "MMMMM"
    pep = "AFAQYLQCPFEDHVK"  # one Q missing
    a = lookup.align(pep, target)
    assert "-" in a["query_line"]
    assert a["query_line"].replace("-", "") == pep[a["query_start"] - 1:a["query_end"]]
    assert a["target_line"].replace("-", "") == target[a["target_start"] - 1:a["target_end"]]


def test_verify_scores_the_true_source_highest():
    db = {"ALB": ALBUMIN, "OTHER": "MSTNPKPQRKTKRNTNRRPQDVKFPGG" * 3}
    rows = lookup.verify({"q": "KSLHTLFGDK"}, {"q": {"ALB", "OTHER"}}, db)
    best = max(rows, key=lambda r: r[2])
    assert best[1] == "ALB"
