"""The hits format a submitted tool module must write (CONTRIBUTING.md), checked by
bin/check_tool_hits.py before scoring."""

import gzip
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "check_tool_hits.py"
Q = "sp|P10415|BCL2_HUMAN"
T = "tr|Q7ZTE4|Q7ZTE4_DANRE"


def run(
    tmp_path: Path, rows: list[str], fastas: bool = False
) -> subprocess.CompletedProcess:
    hits = tmp_path / "human_vs_zebrafish.tool.tsv.gz"
    with gzip.open(hits, "wt") as f:
        f.write("".join(r + "\n" for r in rows))
    cmd = [sys.executable, str(SCRIPT), str(hits)]
    if fastas:
        (tmp_path / "q.fasta").write_text(
            f">{Q} Apoptosis regulator Bcl-2\n" + "M" * 239 + "\n"
        )
        (tmp_path / "t.fasta").write_text(f">{T}\n" + "A" * 200 + "\n")
        cmd += [
            "--query-fasta",
            str(tmp_path / "q.fasta"),
            "--target-fasta",
            str(tmp_path / "t.fasta"),
        ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_good_rows_pass(tmp_path):
    r = run(
        tmp_path,
        [
            f"{Q}\t{T}\t1\t200\t5\t190\t250.5\t1e-60",
            f"{Q}\t{T}\t10\t30\t12\t32\t20\tNA",
        ],
        fastas=True,
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "row, words",
    [
        (f"{Q}\t{T}\t1\t200\t5\t190\t250.5", "7 columns, expected 8"),
        (f"{Q}\t{T}\t0\t200\t5\t190\t250.5\t1e-60", "1-based"),
        (f"{Q}\t{T}\t200\t1\t5\t190\t250.5\t1e-60", "end comes before its start"),
        (f"{Q}\t{T}\t1.5\t200\t5\t190\t250.5\t1e-60", "not a whole number"),
        (f"{Q}\t{T}\t1\t200\t5\t190\tstrong\t1e-60", "is not a number"),
        (f"{Q}\t{T}\t1\t200\t5\t190\t250.5\t-1", "negative"),
    ],
)
def test_bad_rows_fail(tmp_path, row, words):
    r = run(tmp_path, [row])
    assert r.returncode == 1 and words in r.stderr, r.stderr


def test_end_past_sequence_and_unknown_name_fail(tmp_path):
    r = run(tmp_path, [f"{Q}\t{T}\t1\t240\t5\t190\t1\t1"], fastas=True)
    assert r.returncode == 1 and "past the 239-residue" in r.stderr
    r = run(tmp_path, [f"sp|X|Y\t{T}\t1\t20\t5\t19\t1\t1"], fastas=True)
    assert r.returncode == 1 and "not in the query FASTA" in r.stderr


def test_empty_file_fails(tmp_path):
    r = run(tmp_path, [])
    assert r.returncode == 1 and "no rows" in r.stderr


def test_ninth_column_for_motif_tools(tmp_path):
    assert run(tmp_path, [f"{Q}\t{T}\t1\t20\t5\t24\t3.5\tNA\t12"]).returncode == 0
    r = run(tmp_path, [f"{Q}\t{T}\t1\t20\t5\t24\t3.5\tNA\tmany"])
    assert r.returncode == 1 and "n_matched_residues" in r.stderr
