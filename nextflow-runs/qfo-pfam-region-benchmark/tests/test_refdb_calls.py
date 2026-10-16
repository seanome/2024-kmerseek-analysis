"""Reference-database tools (bin/refdb_calls.py): InterProScan and eggNOG-mapper calls.

The fixture under tests/fixtures/refdb is hand-built in each tool's output layout from
hmmscan on the 60 human CI fixture proteins (tests/make_refdb_fixture.sh); it tests the
parsers and the wiring, not the tools.
"""

import gzip
import json
import math
import shutil
import sys

import polars as pl

from conftest import BIN, FIXTURES, run

sys.path.insert(0, str(BIN))
import annotator_tool_rows as atr  # noqa: E402
import refdb_calls as rc  # noqa: E402

REF = FIXTURES / "refdb"
HUMAN = FIXTURES / "qfo" / "human.fasta"


def regions(path) -> list[list[str]]:
    with gzip.open(path, "rt") as fh:
        return [line.rstrip("\n").split("\t") for line in fh]


def test_pfam_names_reads_name_and_bare_accession(tmp_path):
    hmm = tmp_path / "Pfam-A.hmm"
    hmm.write_text(
        "HMMER3/f\nNAME  Pkinase\nACC   PF00069.31\nDESC  x\n//\n"
        "HMMER3/f\nNAME  zf-C2H2\nACC   PF00096.32\n//\n"
    )
    run("refdb_calls.py", "pfam-names", "--pfam-hmm", hmm, "--out", tmp_path / "n.tsv")
    assert (tmp_path / "n.tsv").read_text() == "Pkinase\tPF00069\nzf-C2H2\tPF00096\n"


def test_interproscan_keeps_only_pfam_matches_with_their_coordinates(tmp_path):
    tsv = tmp_path / "ips.tsv"
    tsv.write_text(
        "sp|P1|A_HUMAN\tmd5\t300\tPfam\tPF00069\tPkinase\t10\t250\t1.5E-40\tT\td\tIPR000719\tx\n"
        "sp|P1|A_HUMAN\tmd5\t300\tGene3D\tG3DSA:1.10.510.10\t-\t5\t260\t2E-50\tT\td\t-\t-\n"
        "sp|P1|A_HUMAN\tmd5\t300\tMobiDBLite\tmobidb-lite\t-\t280\t300\t-\tT\td\t-\t-\n"
    )
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">sp|P1|A_HUMAN\n" + "M" * 300 + "\n")
    run(
        "refdb_calls.py",
        "interproscan",
        "--tsv",
        tsv,
        "--queries",
        fasta,
        "--setting",
        "all_members",
        "--out",
        tmp_path / "r.tsv.gz",
        "--qc-out",
        tmp_path / "qc.json",
    )
    r = regions(tmp_path / "r.tsv.gz")
    assert len(r) == 1
    q, fam, s, e, score, ev = r[0]
    assert (q, fam, s, e) == ("sp|P1|A_HUMAN", "PF00069", "10", "250")
    assert math.isclose(float(score), -math.log10(1.5e-40))
    qc = json.loads((tmp_path / "qc.json").read_text())
    assert qc["matches_per_member_database"] == {
        "Gene3D": 1,
        "MobiDBLite": 1,
        "Pfam": 1,
    }
    assert qc["n_queries_with_pfam_match"] == 1


def test_eggnog_calls_span_the_whole_query_and_count_unknown_names(tmp_path):
    r = regions(REF / "eggnog_mapper.diamond.regions.tsv.gz")
    lengths = rc.read_fasta_lengths(HUMAN)
    assert r and all(row[2] == "1" and int(row[3]) == lengths[row[0]] for row in r)
    qc = json.loads((REF / "eggnog_mapper.diamond.calls_qc.json").read_text())
    assert qc["family_names_not_in_pfam_hmm"] == {"NotAPfamName": 1}
    # One call per (query, family), however often the family is named.
    assert len({(row[0], row[1]) for row in r}) == len(r)


def test_both_parsers_reproduce_the_committed_fixture(tmp_path):
    run(
        "refdb_calls.py",
        "interproscan",
        "--tsv",
        REF / "interproscan.all_members.tsv.gz",
        "--queries",
        HUMAN,
        "--setting",
        "all_members",
        "--out",
        tmp_path / "i.tsv.gz",
        "--qc-out",
        tmp_path / "i.json",
    )
    run(
        "refdb_calls.py",
        "eggnog",
        "--annotations",
        REF / "eggnog_mapper.diamond.emapper.annotations",
        "--pfam-names",
        REF / "pfam_names.tsv",
        "--queries",
        HUMAN,
        "--setting",
        "diamond",
        "--out",
        tmp_path / "e.tsv.gz",
        "--qc-out",
        tmp_path / "e.json",
    )
    assert regions(tmp_path / "i.tsv.gz") == regions(
        REF / "interproscan.all_members.regions.tsv.gz"
    )
    assert regions(tmp_path / "e.tsv.gz") == regions(
        REF / "eggnog_mapper.diamond.regions.tsv.gz"
    )


def test_rows_for_a_tool_without_coordinates_hold_only_family_scores(tmp_path):
    res = tmp_path / "results"
    for sub in ("metrics", "calls", "regions"):
        (res / sub).mkdir(parents=True)
    for tool, setting in (
        ("eggnog_mapper", "diamond"),
        ("interproscan", "all_members"),
    ):
        stem = f"pfam.{tool}.{setting}.own_models"
        shutil.copy(REF / f"{stem}.metrics.parquet", res / "metrics")
        shutil.copy(REF / f"{stem}.calls.parquet", res / "calls")
        shutil.copy(REF / f"{tool}.{setting}.calls_qc.json", res / "regions")
    out = tmp_path / "lb"
    run("annotator_tool_rows.py", "--results", res, "--out", out)
    rows = pl.read_parquet(out / "rows.parquet")
    egg = rows.filter(pl.col("tool") == "eggnog_mapper")
    assert set(egg["metric"]) == {"family_fmax"}
    assert set(egg["target_db"]) == {"eggNOG 5.0 (emapperdb 5.0.2)"}
    ips = rows.filter(pl.col("tool") == "interproscan")
    assert "fmax" in set(ips["metric"]) and "median_iou_tp" in set(ips["metric"])
    pile = pl.read_parquet(out / "pileup_calls.parquet")
    assert set(pile["tool"]) == {"interproscan"}
    card = json.loads((out / "cards" / "eggnog_mapper.diamond.json").read_text())
    assert card["group"] == "reference-database tool, graded on its own models"
    assert card["scores_split_all"]["median_iou_tp"] == "not applicable"
    assert card["queries"]["n_family_names_not_in_pfam_hmm"] == 1
    assert atr.TOOL_INFO["interproscan"]["group"] == card["group"]
