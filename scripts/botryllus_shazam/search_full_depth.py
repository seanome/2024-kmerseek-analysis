"""Search only validated uncapped full-depth queries against preserved references."""

import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import sourmash
from full_query_provenance import validate, sha

root, code, out = map(Path, sys.argv[1:])
discovery = root / "full-depth-discovery-20261023"
provenance = validate(root, discovery, code / "ena-runs.tsv")
for p, v in [("sourmash", "4.9.4"), ("sourmash_plugin_branchwater", "0.9.15")]:
    if importlib.metadata.version(p) != v:
        raise ValueError("Pinned versions required")
out.mkdir(exist_ok=False)
report = dict(
    provenance=provenance,
    job_id=os.environ["SLURM_JOB_ID"],
    commands=[],
    results=[],
    queries=[],
)
base = [sys.executable, "-m", "sourmash"]
cores = os.environ["SLURM_CPUS_PER_TASK"]


def save():
    p = out / "progress.json.partial"
    p.write_text(json.dumps(report, indent=2) + "\n")
    p.replace(out / "progress.json")


def call(args, label):
    started = time.monotonic()
    with (out / (label + ".log")).open("w") as f:
        subprocess.run(
            base + list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, check=True
        )
    report["commands"].append(
        dict(argv=list(map(str, args)), seconds=time.monotonic() - started)
    )
    save()


settings = [("dna", 31), ("protein", 7), ("dayhoff", 12), ("hp", 30)]
for mode, k in settings:
    query = discovery / ("extensions.fasta" if mode == "dna" else "peptides.fasta")
    q = out / f"queries.{mode}.scaled5.sig.zip"
    call(
        [
            "sketch",
            "dna" if mode == "dna" else "protein",
            query,
            "--singleton",
            "-p",
            f"{mode},k={k},scaled=5",
            "-o",
            q,
        ],
        "query-" + mode,
    )
    sigs = list(sourmash.load_file_as_signatures(str(q)))
    expected = (
        provenance["extension_records"]
        if mode == "dna"
        else provenance["unique_peptides"]
    )
    if len(sigs) != expected:
        raise ValueError("Sketching dropped full-depth query records")
    for scale in ([10, 100] if mode == "dna" else [5, 10, 100]):
        report["queries"].append(
            dict(
                mode=mode,
                scaled=scale,
                records=len(sigs),
                nonempty=sum(len(s.minhash.downsample(scaled=scale)) > 0 for s in sigs),
                eligible_two_hashes=sum(
                    len(s.minhash.downsample(scaled=scale)) >= 2 for s in sigs
                ),
            )
        )
    targets = []
    if mode != "dna":
        targets.append(
            (
                "swissprot",
                5,
                root
                / f"primary-search-pilot-20261023/swissprot.{mode}.scaled5.sig.zip",
            )
        )
    # Reuse validated target sketches, never historical query sketches.
    external = root / "external-tunicate-pilot-20261023"
    completed = json.loads((external / "execution.complete.json").read_text())
    if completed["status"] != "completed":
        raise ValueError("External target preparation incomplete")
    for run in completed["runs"]:
        for scale in (10, 100):
            targets.append(
                (
                    run["run"],
                    scale,
                    external / run["run"] / f"{mode}.scaled{scale}.sig.zip",
                )
            )
    report["archive_scope"] = (
        "Four completed external target runs; full 5776-run collection is not indexed"
    )
    for db, scale, target in targets:
        native = out / f"{db}.{mode}.scaled{scale}.raw.csv"
        call(
            [
                "scripts",
                "manysearch",
                q,
                target,
                "-c",
                cores,
                "-m",
                "DNA" if mode == "dna" else mode,
                "-k",
                str(k),
                "-s",
                str(scale),
                "-t",
                "0.1",
                "-N",
                "--ignore-abundance",
                "-o",
                native,
            ],
            f"{db}-{mode}-{scale}",
        )
        with native.open() as f:
            rows = sum(1 for _ in csv.DictReader(f))
        report["results"].append(
            dict(
                database=db,
                mode=mode,
                scaled=scale,
                raw_rows=rows,
                csv=str(native),
                sha256=sha(native),
                target_sha256=sha(target),
                interpretation="Unfiltered candidate rows; independent validation and metadata join required",
            )
        )
        save()
report["status"] = "completed"
save()
(out / "execution.complete.json").write_text(json.dumps(report, indent=2) + "\n")
