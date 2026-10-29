"""Full-read external tunicate pilot; streaming Branchwater, no SBT or discovery.

Queries are the preserved historical pilot extensions/peptides. This is not the
full-depth discovery result. Every search reports empty-query denominators.
"""

import csv
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import sourmash
from download import recover

MODES = [("dna", 31), ("protein", 7), ("dayhoff", 12), ("hp", 30)]
root, manifest, validator = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
work = root / "external-tunicate-pilot-20261023"
work.mkdir(exist_ok=False)
cores = os.environ["SLURM_CPUS_PER_TASK"]
base = [sys.executable, "-m", "sourmash"]
for package, version in [
    ("sourmash", "4.9.4"),
    ("sourmash_plugin_branchwater", "0.9.15"),
]:
    if importlib.metadata.version(package) != version:
        raise ValueError("Pinned software required")
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
if (
    len(rows) != 4
    or len({x["study_accession"] for x in rows}) != 4
    or any(x["study_accession"] == "PRJNA1219309" for x in rows)
):
    raise ValueError("Expected four independent external study accessions")
report = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "query_scope": "Historical top-300 pilot: 339 extensions and 987 peptides; full-depth queries pending",
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "status": "starting",
    "commands": [],
    "runs": [],
    "queries": [],
    "results": [],
    "versions": {"sourmash": "4.9.4", "branchwater": "0.9.15", "trim_galore": "0.6.10"},
    "search_scope": "Four complete external paired runs; not the full tunicate collection",
}


def save():
    report["updated_at"] = (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat()
    )
    partial = work / "progress.json.partial"
    partial.write_text(json.dumps(report, indent=2) + "\n")
    partial.replace(work / "progress.json")


def call(args, label):
    start = time.monotonic()
    with (work / (label + ".log")).open("w") as log:
        subprocess.run(
            list(map(str, args)), stdout=log, stderr=subprocess.STDOUT, check=True
        )
    report["commands"].append(
        {
            "argv": list(map(str, args)),
            "seconds": time.monotonic() - start,
            "log": label + ".log",
        }
    )
    save()


def checksum(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


qdir = work / "queries"
qdir.mkdir()
query_sources = {
    mode: root / "primary-search-pilot-20261023" / f"queries.{mode}.scaled5.sig.zip"
    for mode, k in MODES
    if mode != "dna"
}
dna = root / "checkpoints/pilot/results/extensions.fasta"
report["dna_query_source_sha256"] = checksum(dna)
call(
    base
    + [
        "sketch",
        "dna",
        dna,
        "--singleton",
        "-p",
        "k=31,scaled=10",
        "-o",
        qdir / "dna.scaled10.sig.zip",
    ],
    "query-dna",
)
for mode, k in MODES:
    if mode != "dna":
        report.setdefault("protein_query_source_sha256", {})[mode] = checksum(
            query_sources[mode]
        )
        call(
            base
            + [
                "sig",
                "downsample",
                query_sources[mode],
                "--scaled",
                "10",
                "-o",
                qdir / f"{mode}.scaled10.sig.zip",
            ],
            f"query-{mode}-10",
        )
    call(
        base
        + [
            "sig",
            "downsample",
            qdir / f"{mode}.scaled10.sig.zip",
            "--scaled",
            "100",
            "-o",
            qdir / f"{mode}.scaled100.sig.zip",
        ],
        f"query-{mode}-100",
    )
    for scale in (10, 100):
        sigs = list(
            sourmash.load_file_as_signatures(
                str(qdir / f"{mode}.scaled{scale}.sig.zip")
            )
        )
        report["queries"].append(
            {
                "mode": mode,
                "scaled": scale,
                "records": len(sigs),
                "empty_queries": sum(not len(s.minhash) for s in sigs),
                "searchable_queries": sum(len(s.minhash) > 0 for s in sigs),
                "eligible_for_two_hash_filter": sum(len(s.minhash) >= 2 for s in sigs),
                "criterion": "At least two retained hashes required for retained candidate rows",
            }
        )
shutil.copyfile(manifest, work / "external-runs.tsv")
for row in rows:
    run = row["run_accession"]
    report["status"] = "download " + run
    save()
    recover(row, work, validator)
    raw = work / "raw" / run
    record = json.loads((raw / "validated.json").read_text())
    folder = work / run
    folder.mkdir()
    trim = folder / "trimmed"
    trim.mkdir()
    report["status"] = "trim " + run
    save()
    call(
        [
            "trim_galore",
            "--paired",
            "--quality",
            "20",
            "--length",
            "62",
            "--stringency",
            "3",
            "--adapter",
            "AGATCGGAAGAGC",
            "--adapter2",
            "AGATCGGAAGAGC",
            "--cores",
            "2",
            "--output_dir",
            trim,
            *[x["path"] for x in record["files"]],
        ],
        run + "-trim",
    )
    r1 = list(trim.glob("*_val_1.fq.gz"))
    r2 = list(trim.glob("*_val_2.fq.gz"))
    if len(r1) != 1 or len(r2) != 1:
        raise ValueError("Expected one trimmed mate pair")
    reads = [r1[0], r2[0]]
    paired = json.loads(
        subprocess.check_output([str(validator), *map(str, reads)], text=True)
    )
    report["runs"].append(
        {
            "run": run,
            "study": row["study_accession"],
            "raw_pairs": record["pairs"],
            "trimmed_pairs": paired["pairs"],
            "raw_fastq_gz_bytes": sum(x["bytes"] for x in record["files"]),
            "trimmed_files": [
                {"path": str(p), "sha256": checksum(p), "bytes": p.stat().st_size}
                for p in reads
            ],
            "full_read_depth": True,
            "self_study": False,
        }
    )
    for mode, k in MODES:
        report["status"] = f"sketch/search {run} {mode}"
        save()
        sig = folder / f"{mode}.scaled10.sig.zip"
        call(
            base
            + [
                "sketch",
                "dna" if mode == "dna" else "translate",
                *reads,
                "--merge",
                run,
                "-p",
                f"{mode},k={k},scaled=10",
                "-o",
                sig,
            ],
            run + "-" + mode + "-sketch",
        )
        call(
            base
            + [
                "sig",
                "downsample",
                sig,
                "--scaled",
                "100",
                "-o",
                folder / f"{mode}.scaled100.sig.zip",
            ],
            run + "-" + mode + "-100",
        )
        for scale in (10, 100):
            target = folder / f"{mode}.scaled{scale}.sig.zip"
            sigs = list(sourmash.load_file_as_signatures(str(target)))
            if len(sigs) != 1 or sigs[0].name != run or sigs[0].minhash.scaled != scale:
                raise ValueError("Run sketch identity/scale mismatch")
            output = folder / f"{mode}.scaled{scale}.raw.csv"
            call(
                base
                + [
                    "scripts",
                    "manysearch",
                    qdir / f"{mode}.scaled{scale}.sig.zip",
                    target,
                    "-o",
                    output,
                    "-k",
                    str(k),
                    "-s",
                    str(scale),
                    "-m",
                    "DNA" if mode == "dna" else mode,
                    "-c",
                    cores,
                    "-t",
                    "0.1",
                    "-N",
                    "--ignore-abundance",
                ],
                f"{run}-{mode}-{scale}-manysearch",
            )
            retained = folder / f"{mode}.scaled{scale}.retained.csv"
            with output.open() as f, retained.open("w") as out:
                reader = csv.DictReader(f)
                writer = csv.DictWriter(
                    out,
                    [
                        "query_name",
                        "match_name",
                        "containment",
                        "intersect_hashes",
                        "ksize",
                        "scaled",
                        "moltype",
                    ],
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                n = kept = 0
                for hit in reader:
                    n += 1
                    shared = Decimal(hit["intersect_hashes"])
                    if (
                        not shared.is_finite()
                        or shared < 0
                        or shared != shared.to_integral_value()
                    ):
                        raise ValueError("Invalid native shared-hash count")
                    if shared >= 2 and Decimal(hit["containment"]) >= Decimal("0.1"):
                        writer.writerow(hit)
                        kept += 1
            report["results"].append(
                {
                    "run": run,
                    "study": row["study_accession"],
                    "mode": mode,
                    "scaled": scale,
                    "raw_match_rows": n,
                    "retained_match_rows": kept,
                    "signature_zip_bytes": target.stat().st_size,
                    "raw_csv": str(output),
                    "raw_csv_sha256": checksum(output),
                    "retained_csv": str(retained),
                    "retained_csv_sha256": checksum(retained),
                    "target_empty": not len(sigs[0].minhash),
                    "validation": "candidate occurrence only; not independently sequence-validated",
                }
            )
            save()
report["status"] = "completed"
save()
(work / "execution.complete.json").write_text(json.dumps(report, indent=2) + "\n")
