#!/usr/bin/env python3
"""Peak memory and wall time of every task in a Nextflow trace, from Slurm's accounting.

Notebook 270 reports memory and time per arm from sacct, not from the trace's own
peak_rss, so the numbers are what the scheduler measured. For each trace row with a
native_id this asks sacct for the job and all its steps and writes one row per task:

  max_rss_gb   the largest MaxRSS over the job's steps (.batch, .extern, srun steps).
               MaxRSS is a per-step peak, so the job's peak is their maximum, not the
               sum. Each Nextflow task here is its own job, not an array task, so no
               sum over array tasks can enter; a job id with an underscore stops the
               script rather than being read wrong.
  elapsed_s    ElapsedRaw of the job line itself.

Runs on a Sherlock login node with the system python3 (3.6), standard library only.

    python3 scripts/sacct_for_trace.py run-dir/qfo_pfam_region.<date>.trace.txt out.tsv
"""

import csv
import subprocess
import sys

UNITS = {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


def rss_gb(text):
    if not text:
        return None
    if text[-1] in UNITS:
        return float(text[:-1]) * UNITS[text[-1]]
    return float(text) / 1024 ** 3  # bytes


def main():
    trace, out = sys.argv[1], sys.argv[2]
    with open(trace) as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["native_id"] not in ("", "-")]
    ids = sorted({r["native_id"] for r in rows})
    if any("_" in i for i in ids):
        sys.exit("array job ids in the trace; this script does not handle them")

    jobs = {}
    for start in range(0, len(ids), 200):
        chunk = ids[start:start + 200]
        res = subprocess.run(
            ["sacct", "-j", ",".join(chunk), "-P", "-n", "--units=K",
             "--format=JobID,State,ElapsedRaw,MaxRSS"],
            stdout=subprocess.PIPE, universal_newlines=True, check=True)
        for line in res.stdout.splitlines():
            jobid, state, elapsed, maxrss = line.split("|")
            base = jobid.split(".")[0]
            j = jobs.setdefault(base, {"state": None, "elapsed_s": None, "max_rss_gb": None,
                                       "n_steps": 0})
            if "." in jobid:
                j["n_steps"] += 1
                g = rss_gb(maxrss)
                if g is not None and (j["max_rss_gb"] is None or g > j["max_rss_gb"]):
                    j["max_rss_gb"] = g
            else:
                j["state"] = state
                j["elapsed_s"] = int(elapsed)

    missing = [i for i in ids if i not in jobs]
    with open(out, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["process", "tag", "native_id", "trace_status", "sacct_state",
                    "elapsed_s", "max_rss_gb", "n_steps"])
        for r in rows:
            j = jobs.get(r["native_id"], {})
            w.writerow([r["process"], r["tag"], r["native_id"], r["status"], j.get("state"),
                        j.get("elapsed_s"), j.get("max_rss_gb"), j.get("n_steps")])
    print("%d tasks, %d jobs from sacct, %d not found" % (len(rows), len(jobs), len(missing)))


if __name__ == "__main__":
    main()
