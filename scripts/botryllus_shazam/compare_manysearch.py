"""Compare Branchwater manysearch with preserved multisearch peptide retrieval.

No original result is overwritten. Agreement is a search-engine check, not a
validation of protein function or biological occurrence.
"""

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
old = root / "primary-search-pilot-20261023"
out = root / "manysearch-comparison-20261023"
out.mkdir(exist_ok=False)
if (
    importlib.metadata.version("sourmash") != "4.9.4"
    or importlib.metadata.version("sourmash_plugin_branchwater") != "0.9.15"
):
    raise ValueError("Pinned versions required")
execution = json.loads((old / "execution.complete.json").read_text())
report = {
    "source_execution_sha256": hashlib.sha256(
        (old / "execution.complete.json").read_bytes()
    ).hexdigest(),
    "settings": [],
    "scope": "Historical pilot only; engine agreement does not establish biological function",
}


def pairs(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        (r["query_name"], r["match_name"]): (
            int(r["intersect_hashes"]),
            float(r["containment"]),
        )
        for r in rows
        if float(r["containment"]) >= 0.1
    }


for item in execution["results"]:
    db, mode, k, scale = item["database"], item["mode"], item["ksize"], item["scaled"]
    if db == "swissprot" and scale != 5:
        continue
    baseline = Path(item["csv"])
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Baseline CSV changed")
    label = f"{db}.{mode}.scaled{scale}"
    result = out / (label + ".csv")
    args = [
        sys.executable,
        "-m",
        "sourmash",
        "scripts",
        "manysearch",
        str(old / f"queries.{mode}.scaled5.sig.zip"),
        str(old / f"{db}.{mode}.scaled5.sig.zip"),
        "-o",
        str(result),
        "-k",
        str(k),
        "-s",
        str(scale),
        "-m",
        mode,
        "-c",
        os.environ["SLURM_CPUS_PER_TASK"],
        "-t",
        "0.1",
        "-N",
        "--ignore-abundance",
    ]
    with (out / (label + ".log")).open("w") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)
    expected, observed = pairs(baseline), pairs(result)
    shared = expected.keys() & observed.keys()
    differences = sum(
        expected[key][0] != observed[key][0]
        or abs(expected[key][1] - observed[key][1]) > 1e-8
        for key in shared
    )
    report["settings"].append(
        {
            "database": db,
            "mode": mode,
            "scaled": scale,
            "command": args,
            "expected_rows": len(expected),
            "observed_rows": len(observed),
            "missing_pairs": len(expected.keys() - observed.keys()),
            "extra_pairs": len(observed.keys() - expected.keys()),
            "shared_pair_arithmetic_differences": differences,
            "agreement": expected.keys() == observed.keys() and differences == 0,
            "csv_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        }
    )
    (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
report["all_settings_agree"] = all(r["agreement"] for r in report["settings"])
(out / "execution.complete.json").write_text(json.dumps(report, indent=2) + "\n")
