"""Check selective restarts against the captured ten-job accounting."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[3]
PIPE = ROOT / "nextflow-runs/peptide-lookup"
spec = importlib.util.spec_from_file_location(
    "resume_failed", PIPE / "bin/resume_failed_indexes.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_real_accounting_selects_exactly_seven():
    data = ROOT / "notebooks/300_data/step4"
    jobs = list(csv.DictReader((data / "full_jobs.tsv").open(), delimiter="\t"))
    accounting = list(
        csv.DictReader((data / "retry_2026_10_23/sacct.psv").open(), delimiter="|")
    )
    rows = mod.plan(jobs, accounting)
    assert len(rows) == 7
    assert {r["alphabet"] for r in rows} == mod.FAILED
    assert (
        next(r for r in rows if r["alphabet"] == "sdm12")["requested_memory_gib"]
        == 1322
    )
    assert all(r["requested_hours"] == 18 for r in rows if r["alphabet"] != "sdm12")
    accounting[0]["State"] = "RUNNING"
    import pytest

    with pytest.raises(ValueError):
        mod.plan(jobs, accounting)


def test_archive_and_protected_guard(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "full_index.sbatch").write_text(
        'test ! -e "$BASE.rocksdb"\ntouch "$OUT/called"\n'
    )
    env = dict(
        os.environ,
        OUT=str(tmp_path),
        KMERSEEK_SHA="pinned",
        ALPHABET="protein20",
        KSIZE="7",
        PREVIOUS_JOB_ID="old",
        INDEX_SCRIPT_DIR=str(scripts),
    )
    base = tmp_path / "step4/full/protein20.k7.s1.C2.X8.masknone"
    base.parent.mkdir(parents=True)
    Path(str(base) + ".rocksdb").mkdir()
    Path(str(base) + ".log").write_text("old log")
    script = str(PIPE / "sbatch/retry_index.sbatch")
    subprocess.run(["bash", script], env=env, check=True)
    assert (tmp_path / "step4/failed_attempts/old" / (base.name + ".rocksdb")).is_dir()
    assert (tmp_path / "called").exists()
    env["ALPHABET"] = "massspec18"
    assert subprocess.run(["bash", script], env=env).returncode != 0
    env["ALPHABET"] = "protein20"
    Path(str(base) + ".rocksdb").mkdir()
    Path(str(base) + ".complete").touch()
    assert subprocess.run(["bash", script], env=env).returncode != 0
    assert Path(str(base) + ".rocksdb").exists()
