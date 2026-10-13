"""The staging step must not touch a file whose content it did not change.

make_mini_testset.py is a prerequisite of `make run-midi`, so it runs before every run.
Nextflow's default cache hashes an input file by path, size and last-modified TIME, so a
rewrite with byte-identical content invalidates every task that reads that file. That is
what made phmmerSearch, jackhmmerSearch and hmmscanAnnotate re-run on every resume while
everything else cached -- the database builders are on storeDir and their searches take the
stable database directory, so only the arms reading a regenerated file directly paid.

mtime is asserted, not "was it written", because mtime is the thing Nextflow reads.
"""
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
from make_mini_testset import (  # noqa: E402
    write_fasta, write_if_changed, write_parquet_if_changed,
)

OLD = 10_000_000  # a timestamp no test run can produce by accident


def aged(path: Path) -> float:
    os.utime(path, (OLD, OLD))
    return path.stat().st_mtime


def test_identical_bytes_leave_the_file_alone(tmp_path):
    out = tmp_path / "f.txt"
    assert write_if_changed(out, b"hello") is True
    stamp = aged(out)
    assert write_if_changed(out, b"hello") is False
    assert out.stat().st_mtime == stamp


def test_different_bytes_rewrite_the_file(tmp_path):
    out = tmp_path / "f.txt"
    write_if_changed(out, b"hello")
    aged(out)
    assert write_if_changed(out, b"goodbye") is True
    assert out.stat().st_mtime != OLD
    assert out.read_bytes() == b"goodbye"


def test_same_length_different_content_is_still_rewritten(tmp_path):
    # The size check is a shortcut, not the comparison. Two subsets of a proteome can
    # differ while having the same byte count.
    out = tmp_path / "f.txt"
    write_if_changed(out, b"AAAAA")
    aged(out)
    assert write_if_changed(out, b"BBBBB") is True
    assert out.read_bytes() == b"BBBBB"


def test_rewriting_the_same_fasta_subset_does_not_touch_it(tmp_path):
    records = {">sp|P1|A_HUMAN d": "MKV\n", ">sp|P2|B_HUMAN d": "MTT\n",
               ">sp|P3|C_HUMAN d": "MGG\n"}
    out = tmp_path / "q.fasta"
    assert write_fasta(records, {"P1", "P3"}, out) == 2
    assert out.read_text() == ">sp|P1|A_HUMAN d\nMKV\n>sp|P3|C_HUMAN d\nMGG\n"
    stamp = aged(out)
    assert write_fasta(records, {"P1", "P3"}, out) == 2
    assert out.stat().st_mtime == stamp, "an unchanged query FASTA must keep its timestamp"


def test_a_changed_fasta_subset_is_rewritten(tmp_path):
    records = {">sp|P1|A_HUMAN d": "MKV\n", ">sp|P2|B_HUMAN d": "MTT\n"}
    out = tmp_path / "q.fasta"
    write_fasta(records, {"P1"}, out)
    aged(out)
    write_fasta(records, {"P1", "P2"}, out)
    assert out.stat().st_mtime != OLD


def test_rewriting_the_same_annotation_parquet_does_not_touch_it(tmp_path):
    df = pl.DataFrame({"accession": ["P1", "P2"], "pfam_id": ["PF00001", "PF00002"],
                       "domain_start": [1, 5], "domain_end": [90, 200]})
    out = tmp_path / "ann.parquet"
    assert write_parquet_if_changed(df, out) is True
    stamp = aged(out)
    assert write_parquet_if_changed(df, out) is False
    assert out.stat().st_mtime == stamp
    assert pl.read_parquet(out).equals(df)


def test_a_changed_annotation_parquet_is_rewritten(tmp_path):
    df = pl.DataFrame({"accession": ["P1"], "pfam_id": ["PF00001"],
                       "domain_start": [1], "domain_end": [90]})
    out = tmp_path / "ann.parquet"
    write_parquet_if_changed(df, out)
    aged(out)
    assert write_parquet_if_changed(df.with_columns(domain_end=pl.lit(91)), out) is True
    assert out.stat().st_mtime != OLD
