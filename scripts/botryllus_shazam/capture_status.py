"""Import one bounded Sherlock snapshot; no background polling or inferred success."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
a = parser.parse_args()
remote = """import json,os,pathlib,datetime
r=pathlib.Path(os.environ['SCRATCH'])/'botryllus-shazam-PEC-20261023'
m=[json.loads(p.read_text()) for p in (r/'raw').glob('*/validated.json')]
print(json.dumps(dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),validated_raw_runs=len(m),runs=sorted(x['archive']['run'] for x in m),total_runs=64,trimmed_complete=(r/'trimmed-inputs.validated.json').exists(),index_pilot_complete=(r/'indexing-pilot/SRR32233926/validated.json').exists(),annotation_complete=all((r/'annotation-maskX-20261023'/p).exists() for p in ['diamond.complete','pfam.complete']))))
"""
s = json.loads(
    subprocess.check_output(["ssh", "sherlock", "python3 -"], input=remote, text=True)
)
ids = [
    "49188713",
    "49205641",
    "49205669",
    "49192465",
    "49191037",
    "49192201",
    "49206896",
    "49206899",
    "49210194",
    "49211401",
    "49211406",
    "49211416",
    "49212645",
    "49215765",
    "49241354",
    "49241657",
    "49242047",
    "49242789",
    "49244019",
    "49244021",
    "49244023",
    "49347856",
    "49347857",
    "49347864",
    "49348714",
    "49350166",
    "49350957",
    "49351304",
    "49351544",
    "49351548",
]
pipeline_file = a.report / "current-pipeline.json"
pipeline = json.loads(pipeline_file.read_text()) if pipeline_file.exists() else []
ids = sorted(set(ids) | {j["id"] for j in pipeline})
raw = subprocess.check_output(
    [
        "ssh",
        "sherlock",
        "sacct -j "
        + ",".join(ids)
        + " -n -X -P -o JobID,JobName%60,State,Elapsed,ExitCode",
    ],
    text=True,
)
s["jobs"] = [
    dict(zip(["id", "name", "state", "elapsed", "exit_code"], line.split("|")[:5]))
    for line in raw.splitlines()
    if line.strip()
]
s["scope"] = "Verified timestamped Sherlock snapshot; no automatic polling."
a.report.mkdir(parents=True, exist_ok=True)
(a.report / "status.json").write_text(json.dumps(s, indent=2) + "\n")
byid = {j["id"]: j for j in s["jobs"]}
s["current_pipeline"] = [
    dict(
        **byid.get(
            j["id"],
            dict(id=j["id"], state="NOT RETURNED", elapsed=None, exit_code=None),
        ),
        scope=j["scope"]
    )
    for j in pipeline
]
(a.report / "status.json").write_text(json.dumps(s, indent=2) + "\n")
searches = []
for job, scope in [
    ("49242047", "Swiss-Prot and Botryllus manysearch; attended run"),
    ("49242789", "Four complete external tunicate runs: trim, sketch, manysearch"),
    ("49212645", "RocksDB storage and self-search checks on one Botryllus run"),
    ("49215765", "Earlier comparison: parser failure after first Botryllus search"),
]:
    record = byid.get(job)
    searches.append(
        {
            "id": job,
            "scope": scope,
            "state": record["state"] if record else "NOT RETURNED",
            "elapsed": record["elapsed"] if record else None,
            "exit_code": record["exit_code"] if record else None,
        }
    )
c = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "live_status_available": True,
    "checked_at": s["checked_at"],
    "failure": None,
    "search_jobs": searches,
    "external_tunicate_sra": {
        "job_id": "49242789",
        "scope": "Four external paired runs from four studies; historical pilot queries, not full-depth queries",
        "current_state": byid.get("49242789", {}).get("state"),
    },
    "automatic_polling": False,
}
(a.report / "connection-status.json").write_text(json.dumps(c, indent=2) + "\n")
print(
    json.dumps(
        {
            "checked_at": s["checked_at"],
            "validated_raw_runs": s["validated_raw_runs"],
            "search_jobs": searches,
        },
        indent=2,
    )
)
