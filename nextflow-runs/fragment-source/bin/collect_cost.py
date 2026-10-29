#!/usr/bin/env python3
"""Every notebook 354 SLURM job since 20 Oct 2026: cores, elapsed and used CPU time.

sacct reports TotalCPU on a job's steps (batch, extern, srun steps), not on the allocation
line, so the steps are summed onto their job. Runs with Sherlock's system python3 (3.6).

Usage: collect_cost.py > compute.tsv
"""
import subprocess
from collections import OrderedDict

out = subprocess.run(
    ["sacct", "-S", "2026-10-20", "-P", "--noheader",
     "-o", "JobID,JobName,State,NCPUS,Elapsed,ElapsedRaw,TotalCPU,MaxRSS"],
    stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout


def seconds(t):
    days = 0
    if "-" in t:
        d, t = t.split("-")
        days = int(d)
    parts = [float(p) for p in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


jobs = OrderedDict()
for line in out.splitlines():
    jid, name, state, ncpus, elapsed, elapsed_raw, total_cpu, max_rss = line.split("|")
    base = jid.split(".")[0]
    if "." not in jid:
        if not name.startswith("nb354-"):
            continue
        jobs[base] = {"JobID": base, "JobName": name, "State": state, "NCPUS": ncpus,
                      "Elapsed": elapsed, "ElapsedRaw": elapsed_raw, "used_cpu_s": 0.0}
    elif base in jobs:
        jobs[base]["used_cpu_s"] += seconds(total_cpu)
print("JobID\tJobName\tState\tNCPUS\tElapsed\tElapsedRaw\tused_cpu_s")
for j in jobs.values():
    print("\t".join(str(j[k]) for k in ("JobID", "JobName", "State", "NCPUS", "Elapsed", "ElapsedRaw", "used_cpu_s")))
