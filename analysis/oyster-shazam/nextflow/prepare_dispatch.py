"""Write launch resources from committed measurements; never modify tracked files."""

import csv
import json
from pathlib import Path
import sys
from artifacts import sha, write_new


def prepare(history, output):
    output.mkdir()
    traces = sorted(history.glob("job*/summaries/*trace.tsv"))
    if not traces:
        raise ValueError("No measured history")
    evidence = []
    for path in traces:
        with path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        evidence.append(
            dict(
                file=str(path),
                sha256=sha(path),
                tasks=[
                    {
                        k: r[k]
                        for k in ("name", "status", "exit", "realtime", "peak_rss")
                    }
                    for r in rows
                ],
            )
        )
        for row in rows:
            if row["exit"] in ("137", "135", "140"):
                raise ValueError("Review resource failure before launch")
    write_new(
        output / "resources.json",
        dict(
            evidence=evidence,
            cpus=4,
            memory="16 GB",
            time="4h",
            rationale="Previous grouped audit peaked at 1.37 GiB. Each corrected build remains bounded at 2000 records. Full 380867-query grouped runtime has no measurement; four-hour ceiling is a model, not an ETA.",
            expected_repeated_resource_failures=0,
        ),
    )
    (output / "resources.config").write_text(
        "params { task_cpus = 4; task_memory = '16 GB'; task_time = '4h'; partition = 'normal,hns' }\nparams.code = null\n"
    )


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
