"""Convert completed GNU time measurements to the resource estimator's trace schema."""

import csv
import json
from pathlib import Path
import re
import sys

preflight, output = map(Path, sys.argv[1:3])
completion = json.loads((preflight / "complete.json").read_text())
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
    if label == "count" and "recovered_scheduler_peak_kb" in completion:
        # Conservative allocation scope; the later gene-map failure is retained in provenance.
        peak = max(peak, completion["recovered_scheduler_peak_kb"])
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
if len(sys.argv) > 3:
    accounting = list(csv.reader(Path(sys.argv[3]).open(), delimiter="|"))
    batch = [r for r in accounting if r[0].endswith(".batch")]
    if len(batch) != 1 or batch[0][1] != "COMPLETED" or batch[0][4] != "0:0":
        raise ValueError("Preflight allocation did not complete successfully")
    value = batch[0][3]
    if not re.fullmatch(r"[0-9.]+[KMGT]", value):
        raise ValueError("Missing scheduler peak memory")
    rows.append(
        dict(
            name=f"COUNT_KMERS ({run['run']} scheduler allocation)",
            status="COMPLETED",
            exit="0",
            peak_rss=value + "B",
            realtime="0s",
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
