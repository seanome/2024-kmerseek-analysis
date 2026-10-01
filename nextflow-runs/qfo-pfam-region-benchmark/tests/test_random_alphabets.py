"""Tests for the random 2-letter partitions (scripts/random_alphabets.py) and the FASTA
encoder the pipeline runs them through (bin/encode_partition.py). Notebook 246."""

import importlib.util
import sys
from pathlib import Path

import pytest

from conftest import BIN

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BIN))
import encode_partition as ep  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "random_alphabets", REPO / "scripts" / "random_alphabets.py"
)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

#: hp_pbotc_1st_ed2 from kmerseek's README alphabet table (h first, then p). kmerseek has
#: no `encode` subcommand to ask directly.
PBOTC_H, PBOTC_P = "ACFILMPVWY", "DEGHKNQRST"


# --------------------------------------------------------------------------
# random_alphabets.py
# --------------------------------------------------------------------------


def test_every_partition_is_ten_and_ten():
    parts, _, _ = ra.draw_partitions()
    assert len(parts) == 10
    for class1 in parts:
        assert len(class1) == 10
        assert class1 <= set(ra.AMINO_ACIDS)


def test_no_partition_within_three_residues_of_an_hp_alphabet():
    parts, _, _ = ra.draw_partitions()
    for class1 in parts:
        class2 = frozenset(ra.AMINO_ACIDS) - class1
        for h in ra.HP_HYDROPHOBIC.values():
            assert len(class1 ^ frozenset(h)) >= 4
            assert len(class2 ^ frozenset(h)) >= 4


def test_rejection_distance_counts_either_class():
    # pbotc's own partition is at distance 0 whichever class is called 1.
    assert ra.distance_to_hp(frozenset(PBOTC_H)) == (0, "hp_pbotc_1st_ed2")
    assert ra.distance_to_hp(frozenset(PBOTC_P))[0] == 0
    # Swap one residue across pbotc's split: two residues differ.
    moved = frozenset(PBOTC_H.replace("A", "D"))
    assert ra.distance_to_hp(moved)[0] == 2


def test_rejection_happens_when_the_distance_is_raised():
    # At a 9-residue minimum about 92% of draws are rejected, so the counter has to move.
    # Proves the rejection branch runs, not only that the seeded draw never needed it.
    _, n_near, _ = ra.draw_partitions(n=2, min_distance=9)
    assert n_near > 0


def test_an_unreachable_distance_stops_instead_of_looping():
    with pytest.raises(RuntimeError):
        ra.draw_partitions(n=1, min_distance=10, max_draws=500)


def test_same_seed_same_partitions_and_different_seed_differs():
    a = ra.draw_partitions(seed=ra.SEED)
    b = ra.draw_partitions(seed=ra.SEED)
    c = ra.draw_partitions(seed=ra.SEED + 1)
    assert a == b
    assert a[0] != c[0]


def test_no_partition_repeats():
    parts, _, _ = ra.draw_partitions()
    keys = {min(p, frozenset(ra.AMINO_ACIDS) - p, key=sorted) for p in parts}
    assert len(keys) == len(parts)


def test_committed_manifest_matches_the_seed(tmp_path):
    ra.write(tmp_path)
    committed = REPO / "data" / "random_alphabets"
    assert (tmp_path / "random2.manifest.tsv").read_text() == (
        committed / "random2.manifest.tsv"
    ).read_text()
    for i in range(1, 11):
        name = f"random2_{i:02d}.tsv"
        assert (tmp_path / name).read_text() == (committed / name).read_text()


def test_no_random_arm_is_named_hp():
    text = (REPO / "data" / "random_alphabets" / "random2.manifest.tsv").read_text()
    names = [
        line.split("\t")[0]
        for line in text.splitlines()
        if line and not line.startswith(("#", "name"))
    ]
    assert names == [f"random2_{i:02d}" for i in range(1, 11)]


# --------------------------------------------------------------------------
# encode_partition.py
# --------------------------------------------------------------------------


def pbotc_table():
    return {r: 1 if r in PBOTC_H else 2 for r in ep.CANONICAL}


def test_pbotc_partition_round_trips_to_kmerseek_hp_string():
    mapping, split = ep.translation(pbotc_table())
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    # kmerseek's hp_pbotc_1st_ed2 writes h/p; the encoder writes A/D.
    hp = "".join("h" if c in PBOTC_H else "p" for c in seq)
    assert ep.encode_sequence(seq, mapping) == hp.replace("h", "A").replace("p", "D")
    assert split == set()


def test_committed_control_partition_is_pbotc():
    table = ep.read_partition(
        REPO / "data" / "random_alphabets" / "encoded_hp_pbotc_1st_ed2.tsv"
    )
    assert table == pbotc_table()


def test_noncanonical_and_ambiguity_codes_follow_kmerseek():
    mapping, split = ep.translation(pbotc_table())
    # U takes C's class (h), O takes K's (p); B, J, Z each sit inside one pbotc class.
    assert ep.encode_sequence("UOBJZ", mapping) == "ADDAD"
    # X, the stop and lower case.
    assert ep.encode_sequence("aX*", mapping) == "AX*"


def test_ambiguity_code_split_by_partition_becomes_x(tmp_path):
    table = pbotc_table()
    table["N"] = 1  # D is class 2, N class 1: B's two readings now disagree
    table["A"] = 2  # keep the classes 10 and 10 is not required by the encoder
    mapping, split = ep.translation(table)
    assert split == {"B"}
    assert ep.encode_sequence("B", mapping) == "X"


def test_encode_fasta_keeps_headers_and_counts(tmp_path):
    part = tmp_path / "p.tsv"
    part.write_text(
        "residue\tclass\n"
        + "".join(f"{r}\t{1 if r in PBOTC_H else 2}\n" for r in ep.CANONICAL)
    )
    fin = tmp_path / "in.fasta"
    fin.write_text(">sp|P1|X one\nMKTA\nYIAK\n>sp|P2|Y two\nQRS\n")
    fout = tmp_path / "out.fasta"
    counts = ep.encode_fasta(part, fin, fout)
    assert fout.read_text() == ">sp|P1|X one\nADDA\nAAAD\n>sp|P2|Y two\nDDD\n"
    assert counts["n_records"] == 2 and counts["n_residues"] == 11


def test_bad_partition_is_refused(tmp_path):
    part = tmp_path / "p.tsv"
    part.write_text("residue\tclass\nA\t1\n")
    with pytest.raises(ValueError):
        ep.read_partition(part)


# --------------------------------------------------------------------------
# compare_encoded_control.py and the reduction script's file names
# --------------------------------------------------------------------------


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _regions(**overrides):
    import polars as pl

    base = {
        "query_name": ["q1", "q1", "q2"],
        "target_name": ["t1", "t1", "t2"],
        "region_start": [0, 0, 5],
        "region_end": [30, 30, 40],
        "target_start": [2, 50, 7],
        "target_end": [32, 80, 42],
        "region_n_shared_kmers": [12, 12, 17],
        "region_length": [30, 30, 35],
        "region_mean_idf": [3.1, 3.1, 2.2],
        "region_enrichment": [5.0, 5.0, 7.5],
        "region_tail_probability": [1e-6, 1e-6, 1e-9],
        "region_evalue": [1e-3, 1e-3, 1e-5],
        "region_ka_bits": [30.0, 30.0, 40.0],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_control_diff_passes_on_identical_tables(tmp_path):
    cc = _load("compare_encoded_control", "scripts/compare_encoded_control.py")
    _regions().write_parquet(tmp_path / "a.parquet")
    _regions().write_parquet(tmp_path / "b.parquet")
    res = cc.compare(tmp_path / "a.parquet", tmp_path / "b.parquet")
    assert res["n_rows_differing_gated"] == 0
    # q1 has two rows over one query region: 2 distinct regions, 3 rows.
    assert res["n_regions_reference"] == 2 and res["n_rows_reference"] == 3


def test_control_diff_gates_counts_and_only_reports_evalues(tmp_path):
    cc = _load("compare_encoded_control", "scripts/compare_encoded_control.py")
    _regions().write_parquet(tmp_path / "a.parquet")
    _regions(
        region_n_shared_kmers=[12, 11, 17], region_evalue=[1e-3, 1e-3, 1e-1]
    ).write_parquet(tmp_path / "b.parquet")
    res = cc.compare(tmp_path / "a.parquet", tmp_path / "b.parquet")
    assert res["n_gated_diff"] == 1
    assert res["n_region_evalue_diff"] == 1
    assert res["n_pass_evalue_0.01_in_one_only"] == 1
    assert res["max_evalue_ratio"] == pytest.approx(1e4)


def test_reduction_reads_extension_arm_names():
    rd = _load("reduce_landing", "scripts/reduce_swissprot_instance_landing.py")
    m = rd.KM_RE.match(
        "human_vs_zebrafish.random2_01.k19.lcfalse.extend-c1.63.regions.parquet"
    )
    assert (m["alpha"], m["k"], m["ext"]) == ("random2_01", "19", "extend-c1.63")
    m = rd.KM_RE.match(
        "human_vs_zebrafish.hp_pbotc_1st_ed2.k19.lcfalse.regions.parquet"
    )
    assert (m["alpha"], m["ext"]) == ("hp_pbotc_1st_ed2", None)


def test_encoded_hp_partitions_match_the_readme(tmp_path):
    ra.write_hp(tmp_path)
    for alphabet, h in ra.HP_HYDROPHOBIC.items():
        table = ep.read_partition(tmp_path / f"encoded_{alphabet}.tsv")
        assert {r for r, c in table.items() if c == 1} == set(h)
        committed = REPO / "data" / "random_alphabets" / f"encoded_{alphabet}.tsv"
        assert (
            committed.read_text() == (tmp_path / f"encoded_{alphabet}.tsv").read_text()
        )
    assert (tmp_path / "hp2.manifest.tsv").read_text() == (
        REPO / "data" / "random_alphabets" / "hp2.manifest.tsv"
    ).read_text()
