"""Compare the current native count script to a previously verified software fixture."""

import csv
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root, destination = map(Path, sys.argv[1:])
destination.mkdir()
workflow = Path(__file__).resolve().parents[2] / "nextflow-runs/botryllus-shazam/main.nf"
process = (
    workflow.read_text()
    .split("process COUNT_KMERS {")[1]
    .split("process JOIN_KMERS")[0]
)
template = re.search(r'"""(.*?)"""', process, flags=re.S).group(1)
dekupl = Path(os.environ["GROUP_HOME"]) / "opt/botryllus-dekupl-1.3.3-20261024"
old = {}
trace = next((root / "input/results/diagnostics").glob("trace-*.tsv"))
with trace.open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["name"].startswith("COUNT_KMERS ("):
            assert row["status"] == "COMPLETED"
            run = row["name"].split("(")[1].rstrip(")")
            old[run] = Path(row["workdir"]) / f"{run}.counts.txt.gz"


def read_counts(path):
    counts = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            key, value = line.split()
            assert key not in counts
            counts[key] = int(value)
    return counts


records = json.loads((root / "input/inputs.json").read_text())
assert len(records) == 18 and set(old) == {r["run"] for r in records}
checks = []
for record in records:
    run = record["run"]
    work = destination / run
    work.mkdir()
    script = template
    for token, value in {
        "${params.dekupl}": str(dekupl),
        "${task.cpus}": "1",
        "${reads[0]}": record["files"][0]["path"],
        "${reads[1]}": record["files"][1]["path"],
        "${run}": run,
    }.items():
        script = script.replace(token, value)
    subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=work, check=True)
    current = read_counts(work / f"{run}.counts.txt.gz")
    assert current == read_counts(old[run]), run
    checks.append(dict(run=run, canonical_kmers=len(current), counts_identical=True))
report = dict(software_fixture=True, previous_fixture=str(root), libraries=checks)
(destination / "count-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
