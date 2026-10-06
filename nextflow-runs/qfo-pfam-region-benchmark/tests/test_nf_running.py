"""scripts/nf-running against a stand-in squeue, one job of every kind it has to tell apart.

The queue below is modelled on Sherlock on 2026-09-28: a run with its head and tasks, a run
whose head is gone (its tasks are left over), a relaunch waiting on a head, and an sbatch
array that is not a Nextflow run at all.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "nf-running"


def fake_queue(tmp_path, rows, collapsed=None):
    """squeue that prints `rows` when called with -r (one row per array element) and
    `collapsed` without it, the way the real one folds a pending array range into one row."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (tmp_path / "queue.txt").write_text("\n".join(rows) + "\n")
    (tmp_path / "collapsed.txt").write_text("\n".join(collapsed if collapsed is not None else rows) + "\n")
    sq = bin_dir / "squeue"
    sq.write_text("#!/bin/sh\ncase \" $* \" in *\" -r \"*) cat {q};; *) cat {c};; esac\n".format(
        q=tmp_path / "queue.txt", c=tmp_path / "collapsed.txt"))
    sq.chmod(sq.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def task_dir(root, run, h, name):
    d = root / run / "work" / h[:2] / h
    d.mkdir(parents=True)
    # Nextflow writes the #SBATCH block first; the name line comes after it.
    sbatch = "".join(f"#SBATCH --opt{i}\n" for i in range(12))
    (d / ".command.run").write_text(f"#!/bin/bash\n{sbatch}### ---\n### name: '{name}'\n### outputs:\n")
    return d


def run(tmp_path, rows, *args, collapsed=None):
    env = dict(os.environ, PATH=f"{fake_queue(tmp_path, rows, collapsed)}:{os.environ['PATH']}", USER="me")
    out = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_runs_are_named_and_counted(tmp_path):
    midi = tmp_path / "cloneA" / "nextflow-runs" / "invertebrate-dark-set"
    old = tmp_path / "cloneB" / "nextflow-runs" / "qfo-pfam-region-benchmark"
    t1 = task_dir(midi, "runs/ladder-0.4-midi", "aa11", "darkSet:kmerseekSearch (worm.chunk_0001)")
    t2 = task_dir(midi, "runs/ladder-0.4-midi", "bb22", "darkSet:kmerseekIndex (minus_Chromadorea)")
    t3 = task_dir(old, "run-elm-search", "cc33", "kmerseekSearch (human_vs_chicken)")
    rows = [
        f"1|R|36:50|None|{midi}/runs/ladder-0.4-midi|nfdrv-ladder-0.4-midi",
        f"2|R|4:00|None|{t1}|nf-darkSet_kmerseekSearch_(worm.chunk_0001)",
        f"3|PD|0:00|Priority|{t2}|nf-darkSet_kmerseekIndex_(minus_Chromadorea)",
        f"4|PD|0:00|Priority|{t3}|nf-kmers",
        f"5|PD|0:00|BeginTime|{midi}|nfrelaunch-ladder-0.4-midi",
        f"6_2|PD|0:00|Resources|{tmp_path}/250-elm-cover|250-elm-cover",
        f"6_0|R|1:10|None|{tmp_path}/250-elm-cover|250-elm-cover",
    ]
    elm_pending = [f"6_{i}|PD|0:00|Resources|{tmp_path}/250-elm-cover|250-elm-cover" for i in range(3, 20)]
    folded = [r for r in rows] + [f"6_[3-19]|PD|0:00|Resources|{tmp_path}/250-elm-cover|250-elm-cover"]
    out = run(tmp_path, rows + elm_pending, collapsed=folded)
    assert "24 job(s) in the queue: 3 running, 21 pending" in out
    assert "\n2. " in out and "\n3. " not in out          # two runs, not one per task
    midi_block = out.split("1. invertebrate-dark-set, run runs/ladder-0.4-midi\n")[1].split("\n\n")[0]
    assert "tasks   1 running, 1 pending" in midi_block
    assert "invertebrate-dark-set, run runs/ladder-0.4-midi" in out
    assert "head    job 1, running for 36:50" in out
    assert "relaunch queued as job 5 (pending)" in out
    assert "darkSet:kmerseekSearch" in out and "darkSet:kmerseekIndex" in out
    # A run with tasks and no head is flagged, and its process comes from .command.run,
    # not from the truncated job name.
    assert "qfo-pfam-region-benchmark, run run-elm-search" in out
    assert "none in the queue: these tasks are left over" in out
    assert "kmerseekSearch" in out.split("run-elm-search")[1] and "nf-kmers" not in out
    assert "250-elm-cover (job 6) in" in out and "1 running, 18 pending" in out


def test_jobs_lists_only_runs_under_the_folder(tmp_path):
    midi = tmp_path / "c" / "nextflow-runs" / "invertebrate-dark-set"
    t1 = task_dir(midi, "runs/midi", "aa11", "darkSet:kmerseekSearch (worm.chunk_0001)")
    rows = [f"1|R|10:00|None|{midi}/runs/midi|nfdrv-midi",
            f"2|R|4:00|None|{t1}|nf-darkSet_kmerseekSearch_(worm.chunk_0001)"]
    out = run(tmp_path, rows, "--jobs", str(midi))
    table = out.split("Jobs under")[1]
    assert "Nextflow head" in table and "nfdrv-midi" in table
    assert "darkSet:kmerseekSearch" in table and "worm.chunk_0001" in table
    out = run(tmp_path, rows, "--jobs", str(tmp_path / "elsewhere"))
    assert out.rstrip().endswith("none")


def test_empty_queue(tmp_path):
    assert run(tmp_path, []).strip() == "no jobs in the queue"
