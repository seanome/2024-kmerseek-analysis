"""Expose only complete, synchronized nf-core trimmed files to discovery tools."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("manifest", type=Path)
parser.add_argument("validator", type=Path)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()
root, manifest, validator = args.root, args.manifest, args.validator
if args.workers < 1:
    raise ValueError("workers must be positive")
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
if not rows or len({r["run_accession"] for r in rows}) != len(rows):
    raise ValueError("Manifest is empty or has duplicate runs")
files = list((root / "nfcore/results/trimgalore").rglob("*.fq.gz"))


def validate_run(row):
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
    record = dict(
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
    print(f"Validated {run}: {counts['pairs']} pairs", flush=True)
    return record


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    records = list(pool.map(validate_run, rows))
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
    if json.loads(marker.read_text()) != records:
        raise ValueError("Completed trimmed inputs differ from the saved validation")
    print("Reused validated complete trimmed runs:", len(records))
    sys.exit(0)
partial = marker.with_suffix(".json.partial")
partial.write_text(json.dumps(records, indent=2) + "\n")
partial.rename(marker)
print("Validated complete trimmed runs:", len(records))
