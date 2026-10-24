"""Keep canonical Swiss-Prot candidate IDs traceable to original protein records."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
out = root / "manysearch-comparison-20261023-v2/app-export-v2"
summary = json.loads((out / "finished-swissprot-manysearch.json").read_text())
wanted = set()
previews = {}
for result in summary["results"]:
    rows = []
    with gzip.open(out / result["retained_csv"], "rt") as f:
        for row in csv.DictReader(f):
            wanted.add(row["match_name"])
            if len(rows) < 10:
                rows.append(row)
    previews[result["mode"]] = rows
source = root / "primary-search-pilot-20261023/swissprot.record-map.tsv"
dest = out / "matched-swissprot-record-map.tsv.gz"
metadata = {}
kept = 0
with source.open() as f, gzip.open(dest, "wt", newline="") as g:
    reader = csv.DictReader(f, delimiter="\t")
    writer = csv.DictWriter(g, reader.fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in reader:
        if row["record_id"] in wanted:
            kept += 1
            writer.writerow(row)
            if any(
                x["match_name"] == row["record_id"]
                for rows in previews.values()
                for x in rows
            ):
                metadata[row["record_id"]] = row
if kept != len(wanted):
    raise ValueError("Candidate IDs absent from source mapping")
for rows in previews.values():
    for row in rows:
        row.update(metadata[row["match_name"]])
with source.open("rb") as f:
    source_sha = hashlib.file_digest(f, "sha256").hexdigest()
with dest.open("rb") as f:
    dest_sha = hashlib.file_digest(f, "sha256").hexdigest()
(out / "swissprot-candidate-preview.json").write_text(
    json.dumps(
        {
            "preview_rule": "First ten retained rows in each frozen output; not a ranking of biological confidence",
            "matched_reference_records": kept,
            "source_map_sha256": source_sha,
            "matched_map_sha256": dest_sha,
            "preview_rows": previews,
        },
        indent=2,
    )
    + "\n"
)
print("Mapped", kept, "distinct reference records", flush=True)
