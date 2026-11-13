"""Import only completed external search CSVs, verifying recorded fingerprints."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("report", type=Path)
a = p.parse_args()
script = """import os,json,pathlib
p=pathlib.Path(os.environ['SCRATCH'])/'botryllus-shazam-PEC-20261023/external-tunicate-pilot-20261023/progress.json'
print(p.read_text())
"""
record = json.loads(
    subprocess.check_output(["ssh", "sherlock", "python3 -"], input=script, text=True)
)
dest = a.report / "external-results"
dest.mkdir(exist_ok=True)
for row in record["results"]:
    source = Path(row["retained_csv"])
    if not str(source).startswith(
        "/scratch/users/olgabot/botryllus-shazam-PEC-20261023/external-tunicate-pilot-20261023/"
    ):
        raise ValueError("Unexpected external CSV location")
    path = dest / (row["run"] + "." + source.name)
    if path.exists():
        with path.open("rb") as f:
            if (
                hashlib.file_digest(f, "sha256").hexdigest()
                != row["retained_csv_sha256"]
            ):
                raise ValueError("Existing imported output differs; preserved")
    else:
        partial = path.with_suffix(path.suffix + ".partial")
        subprocess.run(["rsync", "sherlock:" + str(source), str(partial)], check=True)
        with partial.open("rb") as f:
            if (
                hashlib.file_digest(f, "sha256").hexdigest()
                != row["retained_csv_sha256"]
            ):
                raise ValueError("Imported candidate CSV fingerprint mismatch")
        partial.rename(path)
    row["local_retained_csv"] = "external-results/" + path.name
(a.report / "external-search-progress.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "status": record["status"],
            "completed_runs": len(record["runs"]),
            "completed_search_settings": len(record["results"]),
            "query_scope": record["query_scope"],
        }
    )
)
