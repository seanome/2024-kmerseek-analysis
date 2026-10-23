"""Reject failed pilot records and unsafe full-data requests."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
from measure_indexes import read_time


def test_gnu_time_units_and_failure(tmp_path):
    f = tmp_path / "time.txt"
    f.write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.50\n"
        "Maximum resident set size (kbytes): 1048576\nExit status: 0\n"
    )
    assert read_time(f) == (1, 3723.5)
    f.write_text(f.read_text().replace("status: 0", "status: 137"))
    with pytest.raises(ValueError):
        read_time(f)


def test_unsafe_plan_never_submits(tmp_path):
    f = tmp_path / "resources.tsv"
    with f.open("w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "alphabet",
                "ksize",
                "full_calibration_linear_gb",
                "full_time_limit_hours",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for i in range(10):
            w.writerow(
                dict(
                    alphabet=f"a{i}",
                    ksize=7,
                    full_calibration_linear_gb=2000,
                    full_time_limit_hours=1,
                )
            )
    result = subprocess.run(
        [
            sys.executable,
            str(BIN / "plan_full_indexes.py"),
            str(f),
            str(tmp_path / "out"),
            "--script",
            "/does/not/exist",
            "--submit",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "No jobs submitted" in result.stderr
    assert not (tmp_path / "out/full_jobs.tsv").exists()


def test_submission_manifest_prevents_duplicates(tmp_path):
    import os

    f = tmp_path / "resources.tsv"
    with f.open("w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "alphabet",
                "ksize",
                "full_calibration_linear_gb",
                "full_time_limit_hours",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for i in range(10):
            w.writerow(
                dict(
                    alphabet=f"a{i}",
                    ksize=7,
                    full_calibration_linear_gb=20,
                    full_time_limit_hours=1,
                )
            )
    fake = tmp_path / "sbatch"
    fake.write_text('#!/bin/sh\nprintf "12345\\n"\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    command = [
        sys.executable,
        str(BIN / "plan_full_indexes.py"),
        str(f),
        str(tmp_path / "out"),
        "--script",
        str(tmp_path / "job.sh"),
        "--submit",
    ]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    manifest = tmp_path / "out/full_jobs.tsv"
    assert len(manifest.read_text().splitlines()) == 11
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode != 0 and "manifest already exists" in second.stderr
    assert len(manifest.read_text().splitlines()) == 11
