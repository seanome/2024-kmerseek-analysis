"""Tests for notebooks/region_evalue_calibration_utils.py (notebook 344)."""

import gzip
import math
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
import region_evalue_calibration_utils as u  # noqa: E402


@pytest.mark.parametrize(
    "length, name",
    [
        (1, "<100 aa"),
        (99, "<100 aa"),
        (100, "100-299 aa"),
        (299, "100-299 aa"),
        (300, "300-1000 aa"),
        (1000, "300-1000 aa"),
        (1001, ">1000 aa"),
        (35_000, ">1000 aa"),
    ],
)
def test_length_edges_follow_rules_md(length, name):
    assert u.length_bin(length) == name


@pytest.mark.parametrize(
    "fraction, name",
    [
        (0.0, "SEG <5%"),
        (0.0499, "SEG <5%"),
        (0.05, "SEG 5-15%"),
        (0.1499, "SEG 5-15%"),
        (0.15, "SEG >=15%"),
        (1.0, "SEG >=15%"),
    ],
)
def test_seg_edges(fraction, name):
    assert u.seg_bin(fraction) == name


def test_twelve_strata():
    assert len(u.STRATA) == 12 == len(set(u.STRATA))


def test_poisson_cdf_matches_closed_form():
    # Pr(X <= 2 | 3) = e^-3 (1 + 3 + 4.5)
    assert u.poisson_cdf(2, 3.0) == pytest.approx(math.exp(-3) * 8.5)


def test_chance_inside_factor_two_at_10():
    # Pr(5 <= X <= 20 | 10)
    expected = sum(math.exp(-10) * 10**k / math.factorial(k) for k in range(5, 21))
    assert u.chance_inside_factor(10) == pytest.approx(expected)


@pytest.mark.parametrize(
    "observed, ok", [(4, False), (5, True), (20, True), (21, False), (0, False)]
)
def test_factor_two_band_is_inclusive(observed, ok):
    assert u.passes(observed, 10) is ok


def write_results(tmp_path, regions_rows):
    """Two strata of 1000 dipeptide decoys each, and a region file."""
    (tmp_path / "decoys").mkdir()
    (tmp_path / "seg").mkdir()
    (tmp_path / "regions").mkdir()
    dec = ["decoy_id\tsource_id\taccession\tkind\treplicate\tlength\tunchanged"]
    seg = ["id\tlength\tseg_residues\tseg_fraction"]
    for i in range(1000):
        dec.append(f"S{i}__dipeptide_r1\tsp|S{i}|X\tS{i}\tdipeptide\t1\t50\t0")
        seg.append(f"sp|S{i}|X\t50\t0\t0.0")
        dec.append(f"L{i}__dipeptide_r1\tsp|L{i}|X\tL{i}\tdipeptide\t1\t2000\t0")
        seg.append(f"sp|L{i}|X\t2000\t400\t0.2")
    (tmp_path / "decoys" / "decoys.tsv").write_text("\n".join(dec) + "\n")
    (tmp_path / "seg" / "human.seg.tsv").write_text("\n".join(seg) + "\n")
    cols = [
        "query_name",
        "target_name",
        "region_start",
        "region_end",
        "target_start",
        "target_end",
        "region_subseq",
        "target_subseq",
        "moltype_seq",
        "region_length",
        "region_n_mismatches",
        "region_n_shared_kmers",
        "region_n_chained",
        "region_run_length",
        "region_pr_same",
        "region_run_evalue",
        "region_ka_bits",
        "region_ka_evalue",
        "region_evalue",
        "region_evalue_source",
        "db_n_targets",
        "region_search_space",
    ]
    with gzip.open(tmp_path / "regions" / "dipeptide.regions.tsv.gz", "wt") as fh:
        fh.write("\t".join(cols) + "\n")
        for q, run, ka in regions_rows:
            fh.write(
                "\t".join(
                    [
                        q,
                        "t",
                        "0",
                        "20",
                        "0",
                        "20",
                        "A" * 20,
                        "A" * 20,
                        "h" * 20,
                        "20",
                        "0",
                        "2",
                        "1",
                        "20",
                        "0.5",
                        run,
                        "1",
                        ka,
                        ka or run,
                        "ka",
                        "1",
                        "1",
                    ]
                )
                + "\n"
            )


def test_calibration_table_counts_per_stratum(tmp_path):
    # Stratum "<100 aa, SEG <5%": Q = 1000, so the cutoffs are 0.01 and 0.1.
    rows = [
        (f"S{i}__dipeptide_r1", "0.005", "") for i in range(10)
    ]  # 10 at run <= 0.01
    rows += [
        (f"S{i}__dipeptide_r1", "0.05", "0.05") for i in range(40)
    ]  # 40 more at <= 0.1
    rows += [
        (f"L{i}__dipeptide_r1", "0.5", "0.0001") for i in range(300)
    ]  # long: run too big
    write_results(tmp_path, rows)
    decoys = u.load_decoys(tmp_path)
    t = u.calibration_table(decoys, u.load_regions(tmp_path, "dipeptide"), "dipeptide")

    def get(s, c, target):
        return t.filter(
            (pl.col("stratum") == s)
            & (pl.col("column") == c)
            & (pl.col("expected") == target)
        ).row(0, named=True)

    short, long_ = "<100 aa, SEG <5%", ">1000 aa, SEG >=15%"
    assert get(short, "region_run_evalue", 10)["observed"] == 10
    assert get(short, "region_run_evalue", 100)["observed"] == 50
    assert (
        get(short, "region_ka_evalue", 10)["observed"] == 0
    )  # empty KA is not counted
    assert get(short, "region_ka_evalue", 100)["observed"] == 40
    assert get(long_, "region_ka_evalue", 10)["observed"] == 300
    assert get(long_, "region_run_evalue", 100)["observed"] == 0
    assert get(short, "region_run_evalue", 10)["inside_factor_two"]
    assert not get(long_, "region_ka_evalue", 10)["inside_factor_two"]
    # Strata with no decoys fail: every stratum must be tested.
    assert not get("300-1000 aa, SEG 5-15%", "region_run_evalue", 10)[
        "inside_factor_two"
    ]
    assert u.verdict(t) == {"region_run_evalue": False, "region_ka_evalue": False}


def test_own_source_and_unchanged_decoys_are_left_out(tmp_path):
    rows = [(f"S{i}__dipeptide_r1", "0.005", "") for i in range(10)]
    write_results(tmp_path, rows)
    # Region on the decoy's own source protein, and a decoy the shuffle left unchanged.
    with gzip.open(tmp_path / "regions" / "dipeptide.regions.tsv.gz", "rt") as fh:
        text = fh.read()
    own = text.splitlines()[1].replace("\tt\t", "\tsp|S0|X\t", 1)
    with gzip.open(tmp_path / "regions" / "dipeptide.regions.tsv.gz", "wt") as fh:
        fh.write(
            text + own + "\n" + text.splitlines()[1].replace("S0__", "S1__") + "\n"
        )
    dec = (
        (tmp_path / "decoys" / "decoys.tsv")
        .read_text()
        .replace(
            "S1__dipeptide_r1\tsp|S1|X\tS1\tdipeptide\t1\t50\t0",
            "S1__dipeptide_r1\tsp|S1|X\tS1\tdipeptide\t1\t50\t1",
        )
    )
    (tmp_path / "decoys" / "decoys.tsv").write_text(dec)
    t = u.calibration_table(
        u.load_decoys(tmp_path), u.load_regions(tmp_path, "dipeptide"), "dipeptide"
    )
    row = t.filter(
        (pl.col("stratum") == "<100 aa, SEG <5%")
        & (pl.col("column") == "region_run_evalue")
        & (pl.col("expected") == 10)
    ).row(0, named=True)
    assert row["n_decoys"] == 999  # S1 is not a decoy
    assert row["observed"] == 9  # S1's region is gone; the own-source row is gone


def test_region_from_another_kind_is_refused(tmp_path):
    write_results(tmp_path, [("S1__window10_r1", "0.1", "0.1")])
    with pytest.raises(ValueError, match="not a dipeptide decoy"):
        u.calibration_table(
            u.load_decoys(tmp_path), u.load_regions(tmp_path, "dipeptide"), "dipeptide"
        )


def test_aligned_region_counts_and_lines():
    a = u.AlignedRegion(query="AKLLE", target="AKIDE", query_start=11, target_start=101)
    c = a.counts()
    # A=A K=K L/I both h, L/D h vs p, E=E
    assert c == {
        "length": 5,
        "identical": 3,
        "same_class": 4,
        "longest_same_class_run": 3,
    }
    lines = a.lines()
    assert lines[:6] == [
        "decoy      11 AKLLE 15",
        "              ||  |",
        "target    101 AKIDE 105",
        "hp         11 hphhp",
        "              ||| |",
        "hp        101 hphpp",
    ]
