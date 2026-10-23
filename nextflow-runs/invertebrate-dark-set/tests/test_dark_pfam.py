"""Pfam summaries distinguish a real zero from missing or malformed inputs."""

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parents[1]


def run_summary(tmp, content, *, threshold=None, missing=False):
    dark = tmp / "dark.parquet"
    pl.DataFrame({"accession": ["A", "B"]}).write_parquet(dark)
    hits = tmp / "hits.tsv.gz"
    if not missing:
        with gzip.open(hits, "wt") as handle:
            handle.write(content)
    summary = tmp / "summary.json"
    command = [
        sys.executable,
        str(ROOT / "bin/summarize_dark_pfam.py"),
        "--hits",
        str(hits),
        "--dark",
        str(dark),
        "--species",
        "toy",
        "--out",
        str(tmp / "out.parquet"),
        "--summary-out",
        str(summary),
    ]
    if threshold is not None:
        command += ["--i-evalue", str(threshold)]
    result = subprocess.run(command, capture_output=True, text=True)
    return result, summary


@pytest.mark.parametrize(
    "content,missing",
    [
        ("", True),
        ("not a domain table\n", False),
        ("A\tPF\tPF00001.1\t1\t10\tnot-a-score\t0.01\n", False),
    ],
)
def test_bad_inputs_are_not_reported_as_zero_hits(tmp_path, content, missing):
    result, summary = run_summary(tmp_path, content, missing=missing)
    assert result.returncode != 0
    assert not summary.exists()


def test_empty_gzip_is_a_real_zero(tmp_path):
    result, summary = run_summary(tmp_path, "")
    assert result.returncode == 0, result.stderr
    data = json.loads(summary.read_text())
    assert data["proteins_dark"] == 2
    assert data["dark_proteins_with_pfam"] == 0


def test_default_preserves_calls_passing_gathering_threshold(tmp_path):
    result, summary = run_summary(tmp_path, "A\tFAMILY\tPF00001.1\t1\t10\t20\t2.0\n")
    assert result.returncode == 0, result.stderr
    data = json.loads(summary.read_text())
    assert data["dark_proteins_with_pfam"] == 1
    assert data["i_evalue_threshold"] is None


def test_optional_evalue_ceiling_is_explicit(tmp_path):
    result, summary = run_summary(
        tmp_path, "A\tFAMILY\tPF00001.1\t1\t10\t20\t2.0\n", threshold=1
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(summary.read_text())["dark_proteins_with_pfam"] == 0


def test_hits_from_another_query_set_are_rejected(tmp_path):
    result, summary = run_summary(
        tmp_path, "UNKNOWN\tFAMILY\tPF00001.1\t1\t10\t20\t0.01\n"
    )
    assert result.returncode != 0
    assert not summary.exists()


def test_no_hit_hmmer_output_does_not_fail_the_pipeline():
    source = (ROOT / "modules/pfam.nf").read_text()
    pipeline = re.search(r"(\| (?:grep|awk).*?)\| gzip", source, re.S).group(1)
    pipeline = (
        re.sub(r"\\+\s*\n", " ", pipeline).replace(r"\$", "$").replace(r"\\t", r"\t")
    )
    result = subprocess.run(
        [
            "bash",
            "-ue",
            "-o",
            "pipefail",
            "-c",
            "printf '# no hits\\n' " + pipeline + "| cat",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_duplicate_dark_fasta_records_are_rejected(tmp_path):
    fasta = tmp_path / "query.fa"
    fasta.write_text(">A first\nAAA\n>A repeated\nAAA\n")
    dark = tmp_path / "dark.parquet"
    pl.DataFrame({"accession": ["A"]}).write_parquet(dark)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin/extract_dark_fasta.py"),
            "--query",
            str(fasta),
            "--dark",
            str(dark),
            "--out",
            str(tmp_path / "out.fa"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()


def test_empty_summary_keeps_numeric_coordinate_types(tmp_path):
    result, _ = run_summary(tmp_path, "")
    assert result.returncode == 0
    data = pl.read_parquet(tmp_path / "out.parquet")
    assert data.schema["env_start"] == pl.Int64
    assert data.schema["i_evalue"] == pl.Float64


def test_hmmer_pipeline_extracts_the_documented_columns():
    source = (ROOT / "modules/pfam.nf").read_text()
    pipeline = re.search(r"(\| (?:grep|awk).*?)\| gzip", source, re.S).group(1)
    pipeline = (
        re.sub(r"\\+\s*\n", " ", pipeline).replace(r"\$", "$").replace(r"\\t", r"\t")
    )
    fields = ["1"] * 22
    for position, value in {
        0: "A",
        3: "FAMILY",
        4: "PF00001.1",
        12: "0.01",
        13: "20",
        19: "1",
        20: "10",
    }.items():
        fields[position] = value
    result = subprocess.run(
        ["bash", "-ue", "-o", "pipefail", "-c", "cat " + pipeline + "| cat"],
        input=" ".join(fields) + "\n",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split("\t") == [
        "A",
        "FAMILY",
        "PF00001.1",
        "1",
        "10",
        "20",
        "0.01",
    ]


@pytest.mark.parametrize(
    "raw", ["A - 100 PFAM PF00001.1 100 0.01\n", "malformed domain row\n"]
)
def test_hmmer_pipeline_rejects_truncated_noncomment_rows(raw):
    source = (ROOT / "modules/pfam.nf").read_text()
    pipeline = re.search(r"(\| (?:grep|awk).*?)\| gzip", source, re.S).group(1)
    pipeline = (
        re.sub(r"\\+\s*\n", " ", pipeline).replace(r"\$", "$").replace(r"\\t", r"\t")
    )
    result = subprocess.run(
        ["bash", "-ue", "-o", "pipefail", "-c", "cat " + pipeline + "| cat"],
        input=raw,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_invalid_pfam_accession_is_rejected(tmp_path):
    result, summary = run_summary(tmp_path, "A\tFAMILY\tNOT_PFAM\t1\t10\t20\t0.01\n")
    assert result.returncode != 0
    assert not summary.exists()
