"""Checks on main.nf. Those that run it are skipped unless GEO2KMERS_TEST_NEXTFLOW is set."""

import csv
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from geo2kmers import sketch

HERE = Path(__file__).resolve().parent.parent
MAIN = (HERE / "main.nf").read_text()
PROCESSES = re.findall(r"^process (\w+) \{(.*?)^\}", MAIN, re.DOTALL | re.MULTILINE)


def default(name):
    return re.search(rf"^params\.{name}\s*=\s*'(.*?)'", MAIN, re.MULTILINE).group(1)


def test_there_are_three_processes():
    assert [name for name, _ in PROCESSES] == ["FETCH", "SEQUENCES", "PROTEOME_INDEX", "SKETCH", "SUMMARISE"]


@pytest.mark.parametrize("name, body", PROCESSES)
def test_each_process_retries_only_when_killed_and_uses_its_own_python(name, body):
    assert "killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry'" in body
    assert "maxRetries 3" in body
    python = "sketch" if name in ("PROTEOME_INDEX", "SUMMARISE") else name.lower()
    assert f"${{params.python_{python}}}" in body
    assert "conda" not in body


def test_the_retry_rule_covers_each_way_a_task_is_killed():
    rule = re.search(r"def killedFromOutside.*?\n\}", MAIN, re.DOTALL).group()
    assert "status == null" in rule and "Integer.MAX_VALUE" in rule and "128..143" in rule


def test_default_sketches_are_valid_settings():
    for setting in default("sketches").split(","):
        alphabet, k = setting.split(":")
        assert alphabet == "dna" or int(k) <= sketch.LARGEST_K[alphabet]
    assert set(default("modes").split(",")) == {"weighted", "present"}


def test_no_email_address_is_in_the_workflow():
    assert "@" not in MAIN and "NCBI_EMAIL" in MAIN


@pytest.fixture
def nextflow(tmp_path):
    if not os.environ.get("GEO2KMERS_TEST_NEXTFLOW"):
        pytest.skip("set GEO2KMERS_TEST_NEXTFLOW to run the workflow; never on a laptop")
    if not shutil.which("nextflow") or not os.environ.get("NCBI_EMAIL"):
        pytest.fail("the workflow tests need nextflow on PATH and NCBI_EMAIL set")

    def run(*args, sketches="dna:21"):
        out = tmp_path / "out"
        command = ["nextflow", "run", str(HERE / "main.nf"), "-profile", "small", "--gse", "GSE2183"]
        command += ["--outdir", str(out), "--sketches", sketches, "--modes", "present"]
        command += ["--python_sequences", sys.executable, "--python_sketch", sys.executable, *args]
        done = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
        with open(tmp_path / "geo2kmers.trace.txt") as f:
            return done, list(csv.DictReader(f, delimiter="\t")), out

    return run


def wrapper(tmp_path, body):
    path = tmp_path / "wrapper"
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_the_workflow_makes_one_sketch_per_channel(nextflow):
    done, trace, out = nextflow("--python_fetch", sys.executable)
    assert done.returncode == 0, done.stdout
    assert sum("SKETCH" in t["name"] for t in trace) == 8  # 4 samples of 2 channels
    assert len(list((out / "sketches").glob("*.dna.k21.present.tsv"))) == 8
    assert (out / "sketch_summary.tsv").exists() and (out / "probe_summary.tsv").exists()
    assert (out / "wall_time.txt").read_text().endswith("exit 0\n")


def test_a_task_killed_twice_is_retried(nextflow, tmp_path):
    count = tmp_path / "count"
    body = f'n=$(cat {count} 2>/dev/null || echo 0); echo $((n+1)) > {count}; [ "$n" -lt 2 ] && kill -9 $$; exec {sys.executable} "$@"'
    done, trace, _ = nextflow("--python_fetch", wrapper(tmp_path, body))
    assert done.returncode == 0, done.stdout
    fetches = [(t["status"], t["exit"], t["attempt"]) for t in trace if "FETCH" in t["name"]]
    assert fetches == [("FAILED", "137", "1"), ("FAILED", "137", "2"), ("COMPLETED", "0", "3")]


def test_a_task_killed_every_time_stops_after_4_tries(nextflow, tmp_path):
    done, trace, _ = nextflow("--python_fetch", wrapper(tmp_path, "kill -9 $$"))
    assert done.returncode != 0
    assert [t["attempt"] for t in trace if "FETCH" in t["name"]] == ["1", "2", "3", "4"]


def test_a_task_that_exits_1_stops_the_run_at_once(nextflow, tmp_path):
    done, trace, _ = nextflow("--python_fetch", wrapper(tmp_path, "exit 1"))
    assert done.returncode != 0
    assert [(t["exit"], t["attempt"]) for t in trace if "FETCH" in t["name"]] == [("1", "1")]


def test_a_proteome_is_read_once_per_setting(nextflow, tmp_path):
    probes = sketch.read_fasta(HERE / "tests/data/GSE2183/probes.fasta.gz")
    proteins = sorted({sketch.longest_stretch(sequence)[2] for sequence in probes.values()})
    proteome = tmp_path / "proteins.fasta"
    proteome.write_text("".join(f">{i}\n{protein}\n" for i, protein in enumerate(proteins)))
    done, trace, out = nextflow(
        "--python_fetch", sys.executable, "--proteome", str(proteome), sketches="dna:21,protein:5"
    )
    assert done.returncode == 0, done.stdout
    assert sum("PROTEOME_INDEX" in t["name"] for t in trace) == 1
    assert len(list((out / "sketches").glob("*.protein.k5.present.tsv"))) == 8
    assert len(list((out / "sketches").glob("*.dna.k21.present.tsv"))) == 8
