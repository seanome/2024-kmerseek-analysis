"""Incomplete or invalid baseline evidence must not become a dark denominator."""

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bin/compute_dark_set.py"
spec = importlib.util.spec_from_file_location("dark_set_review", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def files(tmp_path, rows=None):
    rows = rows or {
        "phmmer": "Q1\tT1\t1\t10\t50\t0.0001\n",
        "jackhmmer": "Q2\tT1\t1\t10\t50\t1\n",
        "mmseqs2": "",
    }
    hits = []
    for arm, text in rows.items():
        p = tmp_path / f"species.chunk_0001.{arm}.tsv.gz"
        with gzip.open(p, "wt") as f:
            f.write(text)
        hits.append(p)
    return hits


def run(tmp_path, hits, *extra):
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">Q1\nAAAA\n>Q2\nGGGG\n")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--query",
            str(fasta),
            "--species",
            "test",
            "--hits",
            *map(str, hits),
            "--out",
            str(tmp_path / "dark.parquet"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_complete_evidence_including_empty_gzip(tmp_path):
    result = run(tmp_path, files(tmp_path), "--expected-chunks", "1")
    assert result.returncode == 0, result.stderr
    assert pl.read_parquet(tmp_path / "dark.parquet")["accession"].to_list() == ["Q2"]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["input_validation"]["complete_chunk_count_checked"]
    assert s["proteins_placed_per_arm"] == {"phmmer": 1, "jackhmmer": 0, "mmseqs2": 0}
    assert s["input_validation"]["files_per_arm"] == dict.fromkeys(mod.ARMS, 1)


def test_missing_arm_or_shared_missing_chunk_rejected(tmp_path):
    hits = files(tmp_path)
    result = run(tmp_path, hits[:2])
    assert result.returncode != 0 and not (tmp_path / "dark.parquet").exists()
    result = run(tmp_path, hits, "--expected-chunks", "2")
    assert result.returncode != 0 and "expected 2 chunks" in result.stderr


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "oops"])
def test_invalid_evalue_rejected_before_output(tmp_path, value):
    hits = files(
        tmp_path,
        {"phmmer": f"Q1\tT1\t1\t10\t50\t{value}\n", "jackhmmer": "", "mmseqs2": ""},
    )
    result = run(tmp_path, hits)
    assert result.returncode != 0 and "invalid E-value" in result.stderr
    assert not (tmp_path / "dark.parquet").exists()


def test_unknown_query_or_bad_cutoff_rejected(tmp_path):
    hits = files(
        tmp_path,
        {"phmmer": "OTHER\tT1\t1\t10\t50\t0.0001\n", "jackhmmer": "", "mmseqs2": ""},
    )
    result = run(tmp_path, hits)
    assert result.returncode != 0 and "absent from FASTA" in result.stderr
    result = run(tmp_path, hits, "--evalue-call", "nan")
    assert result.returncode != 0 and "finite and nonnegative" in result.stderr


def test_duplicate_or_mismatched_chunk_sets_rejected(tmp_path):
    hits = files(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        mod.validate_hit_files(hits + [hits[0]])
    other = tmp_path / "species.chunk_0002.mmseqs2.tsv.gz"
    other.write_bytes(hits[2].read_bytes())
    with pytest.raises(ValueError, match="different chunk coverage"):
        mod.validate_hit_files(hits[:2] + [other])
