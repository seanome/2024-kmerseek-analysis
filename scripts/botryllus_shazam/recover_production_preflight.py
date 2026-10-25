"""Reuse successful measurements after the diagnosed gene-ID mapping failure."""

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

code, previous, out = map(Path, sys.argv[1:])
root = Path(os.environ["SCRATCH"]) / "botryllus-shazam-PEC-20261023"
py = Path(os.environ["GROUP_HOME"]) / "opt/oyster-shazam-20261023/bin/python"
labels = ["count", "kallisto-index", "quantify", "alignment", "pfam"]
for label in labels:
    if not re.search(r"Exit status: 0\s*$", (previous / (label + ".time")).read_text()):
        raise ValueError("Cannot reuse incomplete measurement: " + label)
if (
    "Reference requires unresolved or length-aware gene aggregation"
    not in (previous / "gene-map.stderr").read_text()
):
    raise ValueError("Unexpected prior failure; diagnose before recovery")
job = previous.name.rsplit("-", 1)[1]
accounting = subprocess.check_output(
    [
        "sacct",
        "--noheader",
        "--parsable2",
        "-j",
        job,
        "--format=JobID,State,Elapsed,MaxRSS,ExitCode",
    ],
    text=True,
)
batch = [
    r
    for r in csv.reader(accounting.splitlines(), delimiter="|")
    if r[0].endswith(".batch")
]
if len(batch) != 1 or batch[0][1] != "FAILED" or batch[0][4] != "1:0":
    raise ValueError("Prior allocation state differs from diagnosed gene-map failure")
peak = batch[0][3]
if not re.fullmatch(r"\d+K", peak):
    raise ValueError("Missing prior scheduler peak")
library = json.loads((previous / "largest-library.json").read_text())
out.mkdir(exist_ok=False)
os.chdir(out)
for name in [
    *(label + ".time" for label in labels),
    "largest-library.json",
    "botznik-chr.fa",
    library["run"] + ".quant",
]:
    Path(name).symlink_to((previous / name).resolve())
Path("prior-sacct.tsv").write_text(accounting)
for label, args in [
    (
        "gene-map",
        [
            py,
            code / "scripts/botryllus_shazam/nextflow_stages.py",
            "genes",
            root / "references/botznik-transcripts.fa",
            root / "references/nc_nsi_nf.gtf",
            library["run"] + ".quant",
        ],
    ),
    (
        "checksums",
        [
            "sha256sum",
            "botznik-chr.fa",
            Path(os.environ["SCRATCH"]) / "data/pfam/Pfam-A.hmm",
            root / "references/botznik-transcripts.fa",
            root / "references/nc_nsi_nf.gtf",
        ],
    ),
]:
    with open(label + ".stdout", "w") as stdout, open(label + ".stderr", "w") as stderr:
        subprocess.run(
            ["/usr/bin/time", "-v", "-o", label + ".time", *map(str, args)],
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
Path("complete.json").write_text(
    json.dumps(
        dict(
            genome_matches_all=False,
            library=library["run"],
            pairs=library["pairs"],
            recovered_from=str(previous),
            recovered_scheduler_peak_kb=int(peak[:-1]),
            prior_allocation_state="FAILED 1:0",
            diagnosed_failure="Exact gene IDs incorrectly required a single GTF isoform; successful count/index/quantification/alignment/Pfam measurements reused",
            gene_mapping="Completed with exact gene IDs; multiple quantified sequences per gene still rejected",
        ),
        indent=2,
    )
    + "\n"
)
