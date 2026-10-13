"""A dead arm stops the report, unless the report is a snapshot of a run still going.

An arm that scored zero calls across every species and truth set is a broken arm rather
than a result -- hhblits and folddisco were in that state unnoticed for the whole life of
this pipeline, reading as a bar of length zero. So aggregate_domain_metrics exits non-zero
on one, on purpose.

`make multiqc-partial` is the exception: it reports on the arms scored SO FAR, where an arm
reads as dead simply because its search has not finished yet. The banner still prints.
"""
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

BIN = Path(__file__).resolve().parent.parent / "bin" / "aggregate_domain_metrics.py"


def metrics_row(tool, n_calls, fmax):
    return {"tool": tool, "variant": "v1", "species": "ecoli", "split": "heldout",
            "stratum_axis": "all", "stratum": "all", "truth_set": "pfam",
            "dedup_transfers": False, "n_calls": n_calls, "fmax": fmax, "auprc": fmax}


def build(tmp_path, rows):
    metrics, curves = tmp_path / "metrics", tmp_path / "curves"
    metrics.mkdir(exist_ok=True), curves.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(metrics / "a.metrics.parquet")
    pl.DataFrame({"split": ["heldout"], "recall": [0.5]}).write_parquet(
        curves / "a.curve.parquet")
    return metrics, curves


def run(tmp_path, rows, *flags):
    metrics, curves = build(tmp_path, rows)
    return subprocess.run(
        [sys.executable, str(BIN), *flags, str(metrics), str(curves),
         str(tmp_path / "m.parquet"), str(tmp_path / "m.csv"), str(tmp_path / "c.parquet")],
        capture_output=True, text=True)


ALIVE_AND_DEAD = [metrics_row("kmerseek", 120, 0.4), metrics_row("hhblits", 0, 0.0)]


def test_a_dead_arm_fails_the_report(tmp_path):
    proc = run(tmp_path, ALIVE_AND_DEAD)
    assert proc.returncode != 0
    assert "DEAD ARM" in (proc.stdout + proc.stderr)


def test_the_tables_are_still_written_before_it_fails(tmp_path):
    # The parquet and CSV are what you debug the dead arm FROM, so the failure must not
    # take them with it.
    run(tmp_path, ALIVE_AND_DEAD)
    assert (tmp_path / "m.parquet").exists() and (tmp_path / "m.csv").exists()


def test_allow_dead_arms_reports_anyway_and_still_says_so(tmp_path):
    proc = run(tmp_path, ALIVE_AND_DEAD, "--allow-dead-arms")
    assert proc.returncode == 0, proc.stderr
    assert "DEAD ARM" in proc.stderr
    assert "Reported anyway" in proc.stderr


def test_every_arm_alive_passes_either_way(tmp_path):
    rows = [metrics_row("kmerseek", 120, 0.4), metrics_row("hhblits", 7, 0.1)]
    for flags in ([], ["--allow-dead-arms"]):
        proc = run(tmp_path, rows, *flags)
        assert proc.returncode == 0, proc.stderr
        assert "DEAD ARM" not in proc.stderr


def test_every_arm_dead_is_not_flagged(tmp_path):
    # Nothing to compare against: a run where no tool called anything is a different
    # failure, and check_for_dead_arms deliberately needs a live arm to contrast with.
    proc = run(tmp_path, [metrics_row("kmerseek", 0, 0.0), metrics_row("hhblits", 0, 0.0)])
    assert proc.returncode == 0, proc.stderr


def test_an_unknown_flag_is_refused(tmp_path):
    proc = run(tmp_path, ALIVE_AND_DEAD, "--allow-everything")
    assert proc.returncode != 0
    assert "unknown flag" in proc.stdout + proc.stderr


def test_missing_positionals_are_refused():
    proc = subprocess.run([sys.executable, str(BIN), "--allow-dead-arms"],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "usage:" in proc.stdout + proc.stderr
