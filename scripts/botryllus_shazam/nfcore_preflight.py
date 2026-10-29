"""Run the pinned workflow on paired test reads before any experimental input."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

SINGLETONS = {
    "--no_poly_g",
    "--quality",
    "--length",
    "--stringency",
    "--adapter",
    "--adapter2",
    "--cores",
    "--paired",
    "--gzip",
}


def trim_tokens(text):
    match = re.search(r"^\s*trim_galore\s+(.+?)(?=\n\s*\n|\Z)", text, re.M | re.S)
    if not match:
        raise ValueError("No generated Trim Galore command found")
    tokens = shlex.split(match.group(0).replace("\\\n", " "))
    for option in SINGLETONS:
        if tokens.count(option) > 1:
            raise ValueError("Repeated singleton option: " + option)
    return tokens


def run(root, code, nextflow, pipeline):
    params = json.loads((code / "nfcore-params.json").read_text())
    if "--no_poly_g" in shlex.split(params["extra_trimgalore_args"]):
        raise ValueError(
            "Do not supply pipeline-managed --no_poly_g in extra arguments"
        )
    folder = root / "nfcore/preflight"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "passed.json"
    marker.unlink(missing_ok=True)
    count = 32
    # Controlled paired reads: known sequence, adapter and matching read identifiers.
    sequence = ("ACGTCAGTGCATGACTGACCTAGTCGATCG" * 5)[:120] + "AGATCGGAAGAGC"
    for mate in (1, 2):
        with gzip.open(folder / f"test_{mate}.fastq.gz", "wt") as f:
            for i in range(count):
                f.write(
                    f"@test{i}/{mate}\n{sequence}\n+\n" + "I" * len(sequence) + "\n"
                )
    sheet = folder / "samplesheet.csv"
    with sheet.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "fastq_1", "fastq_2", "strandedness"])
        writer.writerow(
            [
                "preflight",
                folder / "test_1.fastq.gz",
                folder / "test_2.fastq.gz",
                "forward",
            ]
        )
    argv = [
        str(nextflow),
        "run",
        str(pipeline),
        "-profile",
        "singularity",
        "-c",
        str(code / "nfcore.config"),
        "-params-file",
        str(code / "nfcore-params.json"),
        "-work-dir",
        str(folder / "work"),
        "-resume",
        "--input",
        str(sheet),
        "--outdir",
        str(folder / "results"),
        "--transcript_fasta",
        str(root / "references/botznik-transcripts.fa"),
        "--gtf",
        str(root / "references/nc_nsi_nf.gtf"),
    ]
    with (folder / "workflow.log").open("w") as log:
        subprocess.run(
            argv, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    trace = folder / "results/pipeline_info/execution_trace.txt"
    with trace.open() as f:
        tasks = [
            r for r in csv.DictReader(f, delimiter="\t") if ":TRIMGALORE " in r["name"]
        ]
    if (
        len(tasks) != 1
        or tasks[0]["status"] not in ("COMPLETED", "CACHED")
        or tasks[0]["exit"] != "0"
    ):
        raise ValueError("Expected one successfully executed real trimming task")
    command = Path(tasks[0]["workdir"]) / ".command.sh"
    tokens = trim_tokens(command.read_text())
    outputs = sorted((folder / "results/trimgalore").rglob("*_val_*.fq.gz"))
    if len(outputs) != 2:
        raise ValueError("Expected two trimmed mate files")
    records = []
    mate_ids = []
    for output in outputs:
        with gzip.open(output, "rt") as f:
            lines = list(f)
        if len(lines) != count * 4:
            raise ValueError("Unexpected number of trimmed test reads")
        if any(len(lines[i + 1].strip()) < 62 for i in range(0, len(lines), 4)):
            raise ValueError("Unexpected trimmed read lengths")
        ids = [lines[i].strip().rsplit("/", 1)[0] for i in range(0, len(lines), 4)]
        mate_ids.append(ids)
        if any(
            len(lines[i + 1].strip()) != len(lines[i + 3].strip())
            for i in range(0, len(lines), 4)
        ):
            raise ValueError("Sequence and quality lengths differ")
        records.append(dict(path=str(output), reads=count))
    if mate_ids[0] != mate_ids[1]:
        raise ValueError("Trimmed mate identifiers differ")
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            code / "nfcore.config",
            code / "nfcore-params.json",
            command,
            pipeline / "modules/nf-core/trimgalore/main.nf",
        ]
    }
    marker.write_text(
        json.dumps(
            dict(
                status="passed",
                synthetic_fixture=True,
                commands=argv,
                generated_trim_argv=tokens,
                sha256=hashes,
                outputs=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Preflight passed: real pinned workflow, generated command, 32 pairs and both trimmed outputs"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["root", "code", "nextflow", "pipeline"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.root, a.code, a.nextflow, a.pipeline)
