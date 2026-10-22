"""Make full-build requests from completed pilot measurements; optionally submit."""

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument("resources", type=Path)
p.add_argument("out_dir", type=Path)
p.add_argument("--script", type=Path, required=True)
p.add_argument("--submit", action="store_true")
a = p.parse_args()
rows = list(csv.DictReader(a.resources.open(), delimiter="\t"))
if len(rows) != 10 or len({r["alphabet"] for r in rows}) != 10:
    raise SystemExit("All ten paired pilots are required")
planned = []
for r in rows:
    memory = max(32, math.ceil(1.5 * float(r["full_linear_gb"])))
    # Calibration requires a separate allowance: pilot time covers indexing only.
    hours = max(2, math.ceil(float(r["full_time_limit_hours"]) + 4))
    status = "ready" if memory <= 1536 and hours <= 24 else "needs_larger_pilot"
    planned.append(
        dict(
            alphabet=r["alphabet"],
            ksize=r["ksize"],
            requested_memory_gb=memory,
            requested_hours=hours,
            requested_cpus=8,
            status=status,
            basis="1.5x linear pilot memory; 2x linear time plus 4h fit allowance",
        )
    )
a.out_dir.mkdir(parents=True, exist_ok=True)
with (a.out_dir / "full_requests.tsv").open("w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(planned[0]), delimiter="\t")
    w.writeheader()
    w.writerows(planned)
for r in planned:
    print(
        f"{r['alphabet']} k{r['ksize']}: {r['requested_memory_gb']} GB, "
        f"{r['requested_hours']} h, {r['status']}"
    )
if a.submit:
    if any(r["status"] != "ready" for r in planned):
        raise SystemExit("No jobs submitted: requests exceed the node or time ceiling")
    manifest = a.out_dir / "full_jobs.tsv"
    if manifest.exists():
        raise SystemExit(
            "Submission manifest already exists; inspect it before resubmitting"
        )
    # Reserve the manifest before submissions, so partial submissions cannot be duplicated.
    with manifest.open("x") as fh:
        fh.write("alphabet\tksize\tjob_id\n")
        fh.flush()
        for r in planned:
            env = dict(os.environ, ALPHABET=r["alphabet"], KSIZE=r["ksize"])
            job = subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--export=ALL",
                    f"--mem={r['requested_memory_gb']}G",
                    f"--time={r['requested_hours']}:00:00",
                    str(a.script.resolve()),
                ],
                env=env,
                text=True,
            ).strip()
            fh.write(f"{r['alphabet']}\t{r['ksize']}\t{job}\n")
            fh.flush()
