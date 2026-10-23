"""Expose only complete, synchronized nf-core trimmed files to discovery tools."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root, manifest, validator = map(Path, sys.argv[1:])
files = list((root / "nfcore/results/trimgalore").rglob("*.fq.gz"))
records = []
for row in csv.DictReader(manifest.open(), delimiter="\t"):
    run = row["run_accession"]
    mates = []
    for mate in (1, 2):
        matches = [
            f
            for f in files
            if f.name.startswith(run + "_") and f.name.endswith(f"_val_{mate}.fq.gz")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{run} mate {mate}: expected one completed trimmed file, found {matches}"
            )
        mates.append(matches[0])
    counts = json.loads(
        subprocess.check_output([str(validator), *map(str, mates)], text=True)
    )
    if counts["pairs"] > int(row["read_count"]):
        raise ValueError(f"{run}: more pairs after trimming")
    records.append(
        dict(
            run=run,
            **counts,
            input_pairs=int(row["read_count"]),
            files=[
                dict(
                    path=str(p.resolve()),
                    sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
                )
                for p in mates
            ],
        )
    )
assert len({r["run"] for r in records}) == len(records)
dest = root / "shared-trimmed"
dest.mkdir(exist_ok=True)
for record in records:
    for mate, source in enumerate(record["files"], 1):
        link = dest / f'{record["run"]}_{mate}.fastq.gz'
        if link.exists() or link.is_symlink():
            if link.resolve() != Path(source["path"]):
                raise ValueError(f"Existing link differs: {link}")
        else:
            link.symlink_to(source["path"])
marker = root / "trimmed-inputs.validated.json"
if marker.exists():
    raise ValueError("Completed validation exists; compare it before revalidating")
partial = marker.with_suffix(".json.partial")
partial.write_text(json.dumps(records, indent=2) + "\n")
partial.rename(marker)
print("Validated complete trimmed runs:", len(records))
