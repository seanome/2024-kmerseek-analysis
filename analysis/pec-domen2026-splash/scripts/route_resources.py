#!/usr/bin/env python3
"""Wall time, CPU hours and peak memory of every step behind each route of the paper-recovery figure.

Usage: route_resources.py <contrasts_dir> <out.tsv> <trace.txt> [<trace.txt> ...]

Added 2026-11-02 at Olga's request ("give times and resource usage for each"). Nothing is
estimated: every number is read from one of three records, named in the `source` column.
- trace: a Nextflow trace of nextflow-runs/pec-domen2026 (realtime, %cpu, peak_rss). CPU hours =
  realtime x %cpu / 100. When a step completed in more than one run, the last run is used (the
  later traces are listed later on the command line).
- time: a /usr/bin/time -v file written by the step (Elapsed, User + System time, Maximum
  resident set size).
- sacct: Slurm's record of the job that ran the step (Elapsed, TotalCPU, MaxRSS over the job's
  steps), used for the anchor counting of the three small contrasts, run by slurm/12_anchor_counts.sbatch.
A step that several routes need (SPLASH for both anchor routes; DE-kupl's k-mer counting for every
DE-kupl and KaMRaT route) is listed under each route that needs it, with `shared` = yes.
The DE-kupl run is counting plus its limma test; the records do not split the two.
"""
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

cdir, out, traces = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
ANCHOR_COUNT_JOBS = {"young_13_49d": "49206522", "young_132d": "49206527", "old_41_93d": "49206542"}


def dur_h(s: str) -> float:
    """Nextflow duration ('2h 2m 5s', '13.1s', '34ms') in hours."""
    tot = 0.0
    for v, u in re.findall(r"([\d.]+)(ms|h|m|s|d)", s):
        tot += float(v) * {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400}[u]
    return tot / 3600


def mem_gb(s: str) -> float:
    v, u = s.split()
    return float(v) * {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1, "TB": 1e3}[u]


def clock_h(s: str) -> float:
    """h:mm:ss, m:ss(.ss) or d-hh:mm:ss in hours."""
    d = 0
    if "-" in s:
        dd, s = s.split("-")
        d = int(dd)
    p = [float(x) for x in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return d * 24 + p[0] + p[1] / 60 + p[2] / 3600


def read_time(path: Path) -> dict:
    t = path.read_text()
    g = lambda k: re.search(rf"{re.escape(k)}: (.+)", t).group(1).strip()
    return dict(wall_h=clock_h(g("Elapsed (wall clock) time (h:mm:ss or m:ss)")),
                cpu_h=(float(g("User time (seconds)")) + float(g("System time (seconds)"))) / 3600,
                peak_gb=int(g("Maximum resident set size (kbytes)")) / 1e6, source=f"time {path}")


def read_times(paths: list[Path], wall_file: Path | None = None) -> dict | None:
    """Several steps run one after another: wall and CPU add up, peak is the largest."""
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    r = [read_time(p) for p in paths]
    wall = sum(x["wall_h"] for x in r)
    if wall_file is not None and wall_file.exists():
        wall = int(wall_file.read_text().split()[-1]) / 3600
    src = r[0]["source"] if len(r) == 1 else "time " + ", ".join(str(p) for p in paths)
    if len(r) > 5:
        src = f"time {paths[0].parent}/*.time ({len(r)} files)"
    return dict(wall_h=wall, cpu_h=sum(x["cpu_h"] for x in r), peak_gb=max(x["peak_gb"] for x in r), source=src)


def read_sacct(job: str) -> dict:
    o = subprocess.run(["sacct", "-j", job, "-n", "-P", "-o", "JobID,Elapsed,TotalCPU,MaxRSS"],
                       capture_output=True, text=True, check=True).stdout.strip().splitlines()
    rows = [x.split("|") for x in o]
    wall = clock_h(rows[0][1])
    cpu = clock_h(rows[0][2])
    rss = 0.0
    for r in rows:
        m = re.match(r"([\d.]+)([KMG]?)", r[3] or "0")
        if m:
            rss = max(rss, float(m.group(1)) * {"": 1e-9, "K": 1e-6, "M": 1e-3, "G": 1}[m.group(2)])
    return dict(wall_h=wall, cpu_h=cpu, peak_gb=rss, source=f"sacct {job}")


# last COMPLETED trace row per (process, tag)
tr = {}
for f in traces:
    t = pl.read_csv(f, separator="\t", infer_schema_length=0)
    for r in t.filter(pl.col("status") == "COMPLETED").iter_rows(named=True):
        cpu = float(r["%cpu"].rstrip("%")) / 100 if r["%cpu"] not in ("-", "") else float("nan")
        tr[(r["process"], r["tag"])] = dict(wall_h=dur_h(r["realtime"]), cpu_h=dur_h(r["realtime"]) * cpu,
                                            peak_gb=mem_gb(r["peak_rss"]), source=f"trace {Path(f).name} task {r['task_id']}")


def step(contrast: str, name: str) -> dict | None:
    D = cdir / contrast
    tc = lambda p, tag=contrast: tr.get((p, tag))
    tm = lambda *ps: read_times([D / p for p in ps])
    if name == "splash":
        return tc("splash") or tm("splash.time")
    if name == "compactors":
        return tc("compactors") or tm("compactors.time")
    if name == "anchor counts + PyDESeq2 (3 designs)":
        if tc("anchor_counts"):
            return tc("anchor_counts")
        return read_sacct(ANCHOR_COUNT_JOBS[contrast]) if contrast in ANCHOR_COUNT_JOBS else None
    if name == "compactors of kept anchors":
        return tc("compactors_nominal")
    if name == "DE-kupl k-mer counting + limma test":
        return tc("dekupl") or tm("dekupl_limma.time")
    if name == "KaMRaT index, t-test, merge":
        return tc("kamrat")
    if name == "DE-kupl DESeq2 test":
        return tm("dekupl_DESeq2_allkmers/test.time")
    if name.startswith("select + merge "):
        src = name.split()[-1]
        return tm(f"{src}/select.time", f"{src}/merge_up.time", f"{src}/merge_down.time")
    if name == "select kamrat_p05":
        return tm("kamrat_p05/select.time")
    if name == "contig counts + PyDESeq2 + select":
        return tm("kamrat_deseq2/counts.time", "kamrat_deseq2/deseq.time", "kamrat_deseq2_p05/select.time")
    if name.startswith("search "):
        src = name.split()[-1]
        r = tc("search", f"{contrast} {src}")
        if r:
            return r
        ts = sorted((D / f"search_{src}").glob("*.time"))
        return read_times(ts, D / f"search_{src}/search.wall") if ts else None
    raise ValueError(name)


ROUTES = {
    "splash": ["splash", "compactors", "search compactors"],
    "anchor_nominal": ["splash", "anchor counts + PyDESeq2 (3 designs)", "compactors of kept anchors",
                       "search compactors_anchor_nominal"],
    "kamrat_ttest_p05": ["DE-kupl k-mer counting + limma test", "KaMRaT index, t-test, merge", "select kamrat_p05",
                         "search kamrat_p05"],
    "kamrat_deseq2_p05": ["DE-kupl k-mer counting + limma test", "KaMRaT index, t-test, merge",
                          "contig counts + PyDESeq2 + select", "search kamrat_deseq2_p05"],
    "dekupl_limma_p05": ["DE-kupl k-mer counting + limma test", "select + merge dekupl_limma_p05",
                         "search dekupl_limma_p05"],
    "dekupl_deseq2_p05": ["DE-kupl k-mer counting + limma test", "DE-kupl DESeq2 test",
                          "select + merge dekupl_deseq2_p05", "search dekupl_deseq2_p05"],
}
SHARED = {"splash", "DE-kupl k-mer counting + limma test", "KaMRaT index, t-test, merge"}
rows = []
for route, steps in ROUTES.items():
    for c in CONTRASTS:
        for s in steps:
            r = step(c, s)
            rows.append(dict(route=route, contrast=c, step=s, shared="yes" if s in SHARED else "no",
                             **(r or dict(wall_h=None, cpu_h=None, peak_gb=None, source="no record"))))
t = pl.DataFrame(rows)
t.write_csv(out, separator="\t", float_precision=3)
with pl.Config(tbl_rows=-1, tbl_width_chars=250, fmt_str_lengths=60):
    print(t.drop("source"))
