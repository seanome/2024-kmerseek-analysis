"""Recover missing libraries; preserve failed HTTPS data and verify a separate fresh HTTPS download."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from download import recover, digest, transfer

root, manifest, validator = map(Path, sys.argv[1:])
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
missing = [
    r
    for r in rows
    if not (root / "raw" / r["run_accession"] / "validated.json").exists()
]
# Finish unaffected libraries before the repeatedly failing archive object.
missing.sort(key=lambda r: (r["run_accession"] == "SRR32233931", int(r["read_count"])))
for row in missing:
    run = row["run_accession"]
    if run == "SRR32233931":
        folder = root / "raw" / run
        folder.mkdir(parents=True, exist_ok=True)
        for mate, (url, expected, size) in enumerate(
            zip(
                row["fastq_ftp"].split(";"),
                row["fastq_md5"].split(";"),
                map(int, row["fastq_bytes"].split(";")),
            ),
            1,
        ):
            final = folder / f"{run}_{mate}.fastq.gz"
            if final.exists():
                if final.stat().st_size != size or digest(final, "md5") != expected:
                    raise ValueError("Existing final input changed; preserved")
                continue
            old = final.with_suffix(".gz.partial")
            evidence = {
                "expected_md5": expected,
                "expected_bytes": size,
                "transport": "fresh HTTPS after FTP directory rejection",
                "https_partial_preserved": str(old),
            }
            if old.exists():
                evidence.update(
                    https_bytes=old.stat().st_size, https_md5=digest(old, "md5")
                )
            target = final.with_suffix(".gz.fresh-https-v2.partial")
            transfer(
                [
                    "curl",
                    "--silent",
                    "--show-error",
                    "--fail",
                    "--location",
                    "--retry",
                    "5",
                    "--connect-timeout",
                    "30",
                    "--speed-limit",
                    "1024",
                    "--speed-time",
                    "120",
                    "--continue-at",
                    "-",
                    "--output",
                    str(target),
                    "https://" + url,
                ]
            )
            evidence.update(
                fresh_https_bytes=target.stat().st_size,
                fresh_https_md5=digest(target, "md5"),
            )
            (folder / f"fresh-https-v2-mate{mate}.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
            if target.stat().st_size != size or evidence["fresh_https_md5"] != expected:
                raise ValueError(
                    "Fresh HTTPS archive failed expected MD5/size; all files preserved"
                )
            target.rename(final)
    recover(row, root, validator)
