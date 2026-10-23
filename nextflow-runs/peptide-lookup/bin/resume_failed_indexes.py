"""Remeasure the original ten jobs and resubmit only their seven failed settings."""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess

FAILED = {
    "protein20",
    "dayhoff6",
    "funcgroups8",
    "hsdm17",
    "mmseqs12",
    "sdm12",
    "wass14",
}
KEPT = {"massspec18", "uniprot18", "wwmj5"}
FIELDS = [
    "alphabet",
    "ksize",
    "previous_job_id",
    "previous_state",
    "elapsed_seconds",
    "batch_maxrss_gib",
    "previous_memory_gib",
    "requested_memory_gib",
    "requested_hours",
    "requested_cpus",
    "basis",
]


def plan(jobs, accounting):
    by_id = {r["JobID"]: r for r in accounting}
    if len(jobs) != 10 or {r["alphabet"] for r in jobs} != FAILED | KEPT:
        raise ValueError("Expected exactly the original ten settings")
    result = []
    for r in jobs:
        job = by_id[r["job_id"]]
        if r["alphabet"] in KEPT:
            if job["State"] != "COMPLETED":
                raise ValueError("Protected job is not completed")
            continue
        expected = "OUT_OF_MEMORY" if r["alphabet"] == "sdm12" else "TIMEOUT"
        if job["State"] != expected:
            raise ValueError(f"Unexpected state for {r['alphabet']}: {job['State']}")
        batch = by_id[r["job_id"] + ".batch"]
        if not job["ReqMem"].endswith("K") or not batch["MaxRSS"].endswith("K"):
            raise ValueError("Accounting must use KiB units")
        old_mem = int(job["ReqMem"][:-1]) / 1024**2
        result.append(
            dict(
                alphabet=r["alphabet"],
                ksize=r["ksize"],
                previous_job_id=r["job_id"],
                previous_state=job["State"],
                elapsed_seconds=job["ElapsedRaw"],
                batch_maxrss_gib=round(int(batch["MaxRSS"][:-1]) / 1024**2, 3),
                previous_memory_gib=old_mem,
                requested_memory_gib=math.ceil(
                    old_mem * (2 if expected == "OUT_OF_MEMORY" else 1)
                ),
                requested_hours=6 if expected == "OUT_OF_MEMORY" else 18,
                requested_cpus=8,
                basis=(
                    "2x killed memory; retain 6h allowance"
                    if expected == "OUT_OF_MEMORY"
                    else "3x failed 6h limit; calibration was not reached, retain memory"
                ),
            )
        )
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    jobs = list(csv.DictReader((a.out / "step4/full_jobs.tsv").open(), delimiter="\t"))
    raw = subprocess.check_output(
        [
            "sacct",
            "-j",
            ",".join(r["job_id"] for r in jobs),
            "-P",
            "--units=K",
            "-o",
            "JobID,JobName,State,Elapsed,ElapsedRaw,Timelimit,ReqMem,MaxRSS,AllocCPUS,ExitCode",
        ],
        text=True,
    )
    rows = plan(jobs, list(csv.DictReader(raw.splitlines(), delimiter="|")))
    for r in jobs:
        base = a.out / "step4/full" / f"{r['alphabet']}.k{r['ksize']}.s1.C2.X8.masknone"
        complete = Path(str(base) + ".complete").exists()
        if complete != (r["alphabet"] in KEPT):
            raise SystemExit(f"Unexpected completion marker: {base}")
        if not Path(str(base) + ".rocksdb").is_dir():
            raise SystemExit(f"Missing original index: {base}")
    dest = a.out / "step4/retry_2026_10_23"
    dest.mkdir(exist_ok=True)
    (dest / "sacct.psv").write_text(raw)
    with (dest / "requests.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['alphabet']} k{r['ksize']}: peak {r['batch_maxrss_gib']} GiB; request {r['requested_memory_gib']} GiB, {r['requested_hours']} h, 8 CPUs",
            flush=True,
        )
    if not a.submit:
        return
    # Exclusive manifest prevents duplicate or partially repeated submissions.
    with (dest / "jobs.tsv").open("x") as f:
        f.write("alphabet\tksize\tprevious_job_id\tjob_id\n")
        f.flush()
        for r in rows:
            env = dict(
                os.environ,
                OUT=str(a.out),
                ALPHABET=r["alphabet"],
                KSIZE=r["ksize"],
                PREVIOUS_JOB_ID=r["previous_job_id"],
            )
            job = subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--export=ALL",
                    "--constraint=NO_GPU",
                    f"--job-name=nb300-retry-{r['alphabet']}",
                    f"--mem={r['requested_memory_gib']}G",
                    f"--time={r['requested_hours']}:00:00",
                    str(a.script.resolve()),
                ],
                env=env,
                text=True,
            ).strip()
            f.write(f"{r['alphabet']}\t{r['ksize']}\t{r['previous_job_id']}\t{job}\n")
            f.flush()
            print(f"Submitted {r['alphabet']}: {job}", flush=True)


if __name__ == "__main__":
    main()
