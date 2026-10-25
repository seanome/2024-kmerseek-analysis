"""Convert completed GNU time measurements to the resource estimator's trace schema."""

import csv
import json
from pathlib import Path
import re
import sys

preflight, output = map(Path, sys.argv[1:])
json.loads((preflight / "complete.json").read_text())
run = json.loads((preflight / "largest-library.json").read_text())
measurements = {
    "count": "COUNT_KMERS",
    "kallisto-index": "KALLISTO_INDEX",
    "quantify": "QUANTIFY_GENES",
    "gene-map": "GENE_COUNTS",
    "alignment": "ALIGN_CONTEXT",
    "pfam": "QUERY_PFAM",
}
rows = []
for label, process in measurements.items():
    text = (preflight / (label + ".time")).read_text()
    if not re.search(r"Exit status: 0\s*$", text):
        raise ValueError("Incomplete or failed measurement: " + label)
    peak = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)[1])
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)", text)[
        1
    ].strip()
    seconds = 0.0
    for x in elapsed.split(":"):
        seconds = 60 * seconds + float(x)
    rows.append(
        dict(
            name=process
            + " ("
            + (run["run"] if label in ["count", "quantify"] else label)
            + ")",
            status="COMPLETED",
            exit="0",
            peak_rss=f"{peak} KB",
            realtime=f"{seconds} s",
            memory="64 GB",
            time="4h",
        )
    )
with output.open("x") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
print(
    json.dumps(
        dict(
            library=run["run"],
            pairs=run["pairs"],
            trace=str(output),
            caveat="Alignment and Pfam use 500 sequences; other measurements use the largest full library.",
        )
    )
)
