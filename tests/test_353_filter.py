"""Notebook 353: the awk region filter keeps the same rows as the Python one."""

import gzip
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COLS = "query_name,target_name,region_start,region_end,target_start,target_end,region_subseq,target_subseq,moltype_seq,region_length,region_n_mismatches,region_n_shared_kmers,region_pr_same,region_run_evalue,region_ka_bits,region_ka_evalue,region_evalue,region_evalue_source,extra".split(
    ","
)


def fake_csv(n=2000):
    rng = random.Random(1)
    lines = [",".join(COLS)]
    for i in range(n):
        e = rng.choice(
            [
                "",
                f"{rng.uniform(0, 30):.4g}",
                f"{rng.uniform(0, 1):.3e}",
                "10",
                "10.0001",
            ]
        )
        lines.append(
            ",".join(
                [
                    f"q{i % 97}",
                    "T:P1:0",
                    "0",
                    "15",
                    "3",
                    "18",
                    "MSEYQ",
                    "MSEYQ",
                    "bbb",
                    "15",
                    "0",
                    "4",
                    "0.2",
                    e,
                    "3",
                    e,
                    e,
                    "ka",
                    "x",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def test_awk_matches_python(tmp_path):
    text = fake_csv()
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/353/filter_regions.py"),
            "10",
            tmp_path / "py.tsv.gz",
            tmp_path / "py.summary.tsv",
        ],
        input=text,
        text=True,
        check=True,
    )
    awk = subprocess.run(
        [
            "awk",
            "-v",
            "max_e=10",
            "-v",
            f"summary={tmp_path / 'awk.summary.tsv'}",
            "-f",
            str(REPO / "scripts/353/filter_regions.awk"),
        ],
        input=text,
        text=True,
        check=True,
        capture_output=True,
    ).stdout
    py = gzip.open(tmp_path / "py.tsv.gz", "rt").read()
    assert awk == py
    assert (tmp_path / "awk.summary.tsv").read_text() == (
        tmp_path / "py.summary.tsv"
    ).read_text()


def test_awk_refuses_quoted_fields(tmp_path):
    text = (
        ",".join(COLS)
        + "\n"
        + ",".join(["q1", '"T,1"'] + ["0"] * (len(COLS) - 2))
        + "\n"
    )
    res = subprocess.run(
        [
            "awk",
            "-v",
            "max_e=10",
            "-v",
            f"summary={tmp_path / 's.tsv'}",
            "-f",
            str(REPO / "scripts/353/filter_regions.awk"),
        ],
        input=text,
        text=True,
        capture_output=True,
    )
    assert res.returncode == 3
