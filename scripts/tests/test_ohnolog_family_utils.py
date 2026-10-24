import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))

import ohnolog_family_utils as ofu  # noqa: E402

ON = [True] * 9


def off(*domains):
    """All nine domains on except the given (0-based) ones."""
    return [i not in domains for i in range(9)]


def only(*domains):
    return [i in domains for i in range(9)]


# Each case follows one branch of Analyze_Red_Subf_Spec.pl.
@pytest.mark.parametrize(
    "ancestor, copies, fate",
    [
        (ON, [ON, ON], "redundant"),
        (ON, [ON, ON, ON], "redundant"),
        # one copy keeps all nine, the other keeps 3 -> mild (more than 2 kept)
        (ON, [ON, only(0, 1, 2)], "specialized_mild"),
        # one copy keeps all nine, the other keeps 2 -> strong (2 or fewer kept)
        (ON, [ON, only(4, 8)], "specialized_strong"),
        # strong wins if any losing copy keeps 2 or fewer
        (ON, [ON, off(0), only(8)], "specialized_strong"),
        # nobody keeps all nine, together they cover all nine
        (ON, [only(0, 1, 2, 3, 4), only(5, 6, 7, 8)], "subfunctionalized"),
        (ON, [off(0), off(1)], "subfunctionalized"),
        # ancestor not on everywhere: not in Fig. 6d
        (off(3), [ON, ON], "not_tested"),
        # copies together miss a domain: not in Fig. 6d
        (ON, [off(3), off(3)], "not_tested"),
    ],
)
def test_classify_family_marletaz(ancestor, copies, fate):
    assert ofu.classify_family_marletaz(ancestor, copies) == fate


def test_classify_needs_two_copies():
    with pytest.raises(ValueError):
        ofu.classify_family_marletaz(ON, [ON])


@pytest.mark.parametrize(
    "ancestor, copies, fate, new",
    [
        (ON, [ON, ON], "both_keep_all", False),
        (ON, [ON, off(2)], "one_narrows", False),
        (ON, [only(0, 1, 2, 3), only(4, 5, 6, 7, 8)], "split", False),
        (ON, [off(3), off(3)], "cell_type_lost", False),
        (off(3), [ON, off(3)], "ancestor_not_broad", True),
        (off(3), [off(3), off(3)], "ancestor_not_broad", False),
    ],
)
def test_classify_copies_against_ancestor(ancestor, copies, fate, new):
    got = ofu.classify_copies_against_ancestor(ancestor, copies)
    assert got == {"fate": fate, "new_cell_type": new}


def test_published_counts_add_up():
    """Fates in F6ae.md sum to the valid families, and give the paper's 80-88%."""
    t = ofu.MARLETAZ_PUBLISHED_FATES
    parts = (
        t["subfunctionalized"]
        + t["specialized_strong"]
        + t["specialized_mild"]
        + t["redundant"]
    )
    assert (parts == t["valid_all_nine_domains"]).all()
    lost = (
        t["subfunctionalized"] + t["specialized_strong"] + t["specialized_mild"]
    ) / t["valid_all_nine_domains"]
    assert lost.min() == pytest.approx(289 / 360)  # frog, 80.3%
    assert lost.max() == pytest.approx(321 / 366)  # zebrafish, 87.7%


ORTHOGROUPS = (
    "Orthogroup\tamphioxus\tbotryllus\tciona\thuman\n"
    "OG0\tBL1\tFUN1_X\tC1\tH1, H2, H3\n"  # 1 Botryllus, 3 human, H1-H2 and H2-H3 pairs
    "OG1\tBL2\t\t\tH4, H5\n"  # no tunicate copy; H4-H5 pair
    "OG2\tBL3, BL4\tFUN2_Y, FUN3_Y\tC2\tH6\n"  # H6's partner H7 is in OG3
    "OG3\t\t\t\tH7\n"
)
PAIRS = pl.DataFrame(
    {
        "ohno1": ["H1", "H2", "H4", "H6"],
        "ohno2": ["H2", "H3", "H5", "H7"],
        "symbol1": ["A", "B", "D", "F"],
        "symbol2": ["B", "C", "E", "G"],
    }
)


def test_family_table(tmp_path):
    path = tmp_path / "Orthogroups.tsv"
    path.write_text(ORTHOGROUPS)
    genes = ofu.read_orthogroups(path)
    assert genes.height == 16  # 6 + 3 + 6 + 1
    table = ofu.family_table(genes, PAIRS).to_dicts()
    by_id = {r["family_id"]: r for r in table}

    og0 = by_id["OG0"]
    assert (og0["n_botryllus"], og0["n_ciona"], og0["n_amphioxus"], og0["n_human"]) == (
        1,
        1,
        1,
        3,
    )
    assert og0["human_ohnologs"] == ["H1", "H2", "H3"]

    og1 = by_id["OG1"]
    assert (og1["n_botryllus"], og1["n_amphioxus"], og1["n_human_ohnologs"]) == (
        0,
        1,
        2,
    )

    # H6 is in OHNOLOGS, but its partner is in another family: listed, not counted.
    og2 = by_id["OG2"]
    assert og2["n_human_ohnologs"] == 0
    assert og2["n_human_listed_in_ohnologs"] == 1
    assert og2["botryllus_genes"] == ["FUN2_Y", "FUN3_Y"]

    grid = {
        (r["botryllus_copies"], r["human_ohnolog_copies"]): r["n_families"]
        for r in ofu.copy_number_grid(ofu.family_table(genes, PAIRS)).to_dicts()
    }
    assert grid == {(1, 3): 1, (0, 2): 1, (2, 0): 1, (0, 0): 1}
