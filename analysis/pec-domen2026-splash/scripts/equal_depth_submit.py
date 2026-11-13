#!/usr/bin/env python3
"""Submit the equal-depth test (notebook 361) on Sherlock: subsample, SPLASH, compactors, search.

Usage: equal_depth_submit.py <pec_root> <code_dir> [--dry-run]

For young_24h and young_2h, a control-PEC pair is kept when both samples have at least MAIN read
pairs after trimming; a pair with one sample below that is dropped whole, so conditions stay
balanced. Every arm of a contrast uses the same kept pairs:
  full      the kept samples at their full depth (separates "fewer samples" from "equal depth")
  2000000   every kept sample cut to 2 M read pairs (seqtk sample -s 361)
  1000000   every kept sample cut to 1 M read pairs, drawn from the 2 M reads
The simulated control (12 samples, about 1 M pairs each) is cut to 500,000, 250,000 and 125,000
pairs, and scored against its planted changes. Jobs are chained with afterok; the summary job
runs after all of them (afterany) and reports what is missing.

Writes <pec_root>/equal_depth/<contrast>/kept_samples.tsv and job_ids.tsv, and prints the job IDs.
"""
import subprocess
import sys
from pathlib import Path

pec, code = Path(sys.argv[1]), Path(sys.argv[2])
DRY = "--dry-run" in sys.argv
MAIN = 2_000_000
DEPTHS = [2_000_000, 1_000_000]
CONTROL_DEPTHS = [500_000, 250_000, 125_000]
ED = pec / "equal_depth"
SB = code / "slurm/20_equal_depth.sbatch"
LOGS = ED / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

# first asks, from the measured runs (see exports/equal_depth/README.md, "Resources")
RES = {
    "subsample": (4, "8G", "2:00:00"),
    "subsample_control": (2, "4G", "0:30:00"),
    "splash_full": (32, "128G", "3:00:00"),
    "splash": (32, "64G", "2:00:00"),
    "splash_control": (16, "16G", "0:30:00"),
    "compactors_full": (32, "64G", "3:00:00"),
    "compactors": (32, "32G", "2:00:00"),
    "search": (16, "128G", "6:00:00"),
    "control": (2, "32G", "1:00:00"),
    "summary": (4, "32G", "1:00:00"),
}
jobs = []


def submit(name, res, env, after=None, kind="afterok"):
    cpus, mem, time = RES[res]
    exports = ",".join(["ALL", f"PEC_ROOT={pec}", f"CODE_DIR={code}"] + [f"{k}={v}" for k, v in env.items()])
    cmd = ["sbatch", "--parsable", "-p", "normal,hns", "--account=ayeletv", f"--job-name=nb361-ed-{name}",
           f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={time}", f"--output={LOGS}/%x.%j.out",
           f"--export={exports}"]
    if after:
        cmd.append(f"--dependency={kind}:" + ":".join(after))
    cmd.append(str(SB) if res != "summary" else str(code / "slurm/21_equal_depth_summary.sbatch"))
    if DRY:
        jid = f"DRY{len(jobs)}"
        print(" ".join(cmd))
    else:
        jid = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    jobs.append((name, jid, res))
    return jid


def reads_pairs(done):
    return int(done.read_text().split()[0]) // 2  # fastp's total_reads counts R1 and R2


def write_arm(d, kept_rows, cjs, il_of):
    d.mkdir(parents=True, exist_ok=True)
    (d / "input.txt").write_text("".join(f"{s} {il_of(s)}\n" for s in kept_rows))
    (d / "cjs.csv").write_text("".join(f"{s},{cjs[s]}\n" for s in kept_rows))


all_last = []
for c in ["young_24h", "young_2h"]:
    design = [ln.split("\t") for ln in (code / f"design/contrast_{c}.design.tsv").read_text().splitlines()[1:]]
    cjs = dict(ln.split(",") for ln in (code / f"design/contrast_{c}.cjs").read_text().split())
    avail = {s: reads_pairs(pec / f"trimmed/{s}.done") for s, *_ in design}
    by_pair = {}
    for s, cond, pair, *_ in design:
        by_pair.setdefault(pair, []).append(s)
    kept = [s for s, cond, pair, *_ in design if all(avail[t] >= MAIN for t in by_pair[pair])]
    (ED / c).mkdir(parents=True, exist_ok=True)
    with open(ED / c / "kept_samples.tsv", "w") as f:
        f.write("sample\tcondition\tpair\tread_pairs_after_trimming\tkept\n")
        for s, cond, pair, *_ in design:
            f.write(f"{s}\t{cond}\t{pair}\t{avail[s]}\t{s in kept}\n")
    print(c, "kept", len(kept), "of", len(design), "samples:", " ".join(kept))
    sub = [submit(f"sub-{s}", "subsample", dict(STAGE="subsample", SAMPLE=s, CONTRAST=c, SRC=pec / "trimmed",
                                                 DEPTHS=":".join(map(str, DEPTHS)))) for s in kept]
    write_arm(ED / c / "full", kept, cjs, lambda s: pec / f"trimmed/{s}.il.fastq.gz")
    for d in DEPTHS:
        write_arm(ED / c / str(d), kept, cjs, lambda s, d=d: ED / c / str(d) / f"reads/{s}.il.fastq.gz")
    for d in ["full", *map(str, DEPTHS)]:
        env = dict(CONTRAST=c, DEPTH=d)
        full = d == "full"
        sp = submit(f"splash-{c}-{d}", "splash_full" if full else "splash", dict(STAGE="splash", **env),
                    after=None if full else sub)
        cp = submit(f"comp-{c}-{d}", "compactors_full" if full else "compactors", dict(STAGE="compactors", **env), after=[sp])
        all_last.append(submit(f"search-{c}-{d}", "search", dict(STAGE="search", **env), after=[cp]))

ctrl = [f"ctrl{i}" for i in range(1, 7)] + [f"pec{i}" for i in range(1, 7)]
ccjs = dict(ln.split(",") for ln in (code / "control/control.cjs").read_text().split())
sub = [submit(f"sub-{s}", "subsample_control", dict(STAGE="subsample", SAMPLE=s, CONTRAST="control",
                                                     SRC=pec / "control/trimmed",
                                                     DEPTHS=":".join(map(str, CONTROL_DEPTHS))))
       for s in ctrl]
for d in CONTROL_DEPTHS:
    write_arm(ED / "control" / str(d), ctrl, ccjs, lambda s, d=d: ED / "control" / str(d) / f"reads/{s}.il.fastq.gz")
    sp = submit(f"splash-control-{d}", "splash_control", dict(STAGE="splash", CONTRAST="control", DEPTH=d), after=sub)
    all_last.append(submit(f"truth-control-{d}", "control", dict(STAGE="control", DEPTH=d), after=[sp]))

submit("summary", "summary", {}, after=all_last, kind="afterany")
with open(ED / "job_ids.tsv", "a") as f:
    for name, jid, res in jobs:
        f.write(f"{name}\t{jid}\t{' '.join(map(str, RES[res]))}\n")
for name, jid, res in jobs:
    print(f"{jid}\t{name}\t{RES[res]}")
