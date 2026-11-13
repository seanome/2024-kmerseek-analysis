"""Measure ZIP overhead, merged reference storage, and Branchwater SRA storage.

Merged Swiss-Prot sketches are storage experiments: they do not retain protein IDs.
The SRA check reuses historical raw-read sketches, not trimmed archive evidence.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path(sys.argv[1])
out = root / "storage-benchmark-20261023"
out.mkdir(exist_ok=False)
cores = os.environ["SLURM_CPUS_PER_TASK"]
base = [sys.executable, "-m", "sourmash"]
commands = []


def run(args, label):
    with (out / (label + ".log")).open("w") as f:
        subprocess.run(base + args, stdout=f, stderr=subprocess.STDOUT, check=True)
    commands.append(base + args)


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


report = {
    "scope": "Storage and index correctness experiment only; historical raw SRA inputs; no external archive search",
    "swissprot": [],
    "sra": [],
    "commands": commands,
}
for mode, k in [("protein", 7), ("dayhoff", 12), ("hp", 30)]:
    source = (
        root
        / "swissprot-size-comparison-20261023"
        / f"swissprot.{mode}.scaled100.sig.zip"
    )
    with zipfile.ZipFile(source) as z:
        entries = z.infolist()
        payload = sum(x.compress_size for x in entries)
    merged = out / f"swissprot.{mode}.merged.scaled100.sig.zip"
    run(
        [
            "sig",
            "merge",
            str(source),
            "--" + mode,
            "-k",
            str(k),
            "--set-name",
            "Swiss-Prot union: protein identity lost",
            "-o",
            str(merged),
        ],
        "merge-" + mode,
    )
    report["swissprot"].append(
        {
            "alphabet": mode,
            "scaled": 100,
            "source_bytes": source.stat().st_size,
            "zip_entries": len(entries),
            "compressed_payload_bytes": payload,
            "zip_container_bytes": source.stat().st_size - payload,
            "merged_bytes": merged.stat().st_size,
            "merged_loses_protein_identity": True,
        }
    )
    (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
raw = root / "raw/SRR32233926"
raw_bytes = sum(
    (raw / f"SRR32233926_{mate}.fastq.gz").stat().st_size for mate in (1, 2)
)
for mode, k, mol in [
    ("dna", 31, "DNA"),
    ("protein", 7, "protein"),
    ("dayhoff", 12, "dayhoff"),
    ("hp", 30, "hp"),
]:
    for scale in (10, 100):
        sig = root / "indexing-pilot/SRR32233926" / mode / f"scaled{scale}.sig.zip"
        label = f"{mode}.scaled{scale}"
        db = out / (label + ".rocksdb")
        run(
            [
                "scripts",
                "index",
                str(sig),
                "-o",
                str(db),
                "-k",
                str(k),
                "-s",
                str(scale),
                "-m",
                mol,
                "-c",
                cores,
                "--internal-storage",
            ],
            label + "-index",
        )
        run(["scripts", "check", str(db)], label + "-check")
        csv_out = out / (label + ".self-search.csv")
        run(
            [
                "scripts",
                "manysearch",
                str(sig),
                str(db),
                "-o",
                str(csv_out),
                "-k",
                str(k),
                "-s",
                str(scale),
                "-m",
                mol,
                "-c",
                cores,
                "-t",
                "1",
                "-N",
                "--ignore-abundance",
            ],
            label + "-search",
        )
        with csv_out.open() as f:
            rows = list(csv.DictReader(f))
        if (
            len(rows) != 1
            or rows[0]["query_name"] != rows[0]["match_name"]
            or float(rows[0]["containment"]) != 1
        ):
            raise ValueError(f"Failed exact self-search: {label}, {rows}")
        report["sra"].append(
            {
                "run": "SRR32233926",
                "mode": mode,
                "scaled": scale,
                "raw_fastq_gz_bytes": raw_bytes,
                "signature_zip_bytes": sig.stat().st_size,
                "rocksdb_directory_bytes": size(db),
                "self_search_rows": len(rows),
                "self_study": True,
            }
        )
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
(out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
