"""Measure real inputs and check matching genome and Pfam reference assets in Slurm."""

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

code, out = map(Path, sys.argv[1:])
root = Path(os.environ["SCRATCH"]) / "botryllus-shazam-PEC-20261023"
de = Path(os.environ["GROUP_HOME"]) / "opt/botryllus-dekupl-1.3.3-20261024"
py = Path(os.environ["GROUP_HOME"]) / "opt/oyster-shazam-20261023/bin/python"
out.mkdir(exist_ok=False)
os.chdir(out)


def run(label, args):
    with open(label + ".stdout", "w") as stdout, open(label + ".stderr", "w") as stderr:
        subprocess.run(
            ["/usr/bin/time", "-v", "-o", label + ".time", *map(str, args)],
            stdout=stdout,
            stderr=stderr,
            check=True,
        )


url = "https://botryllus.stanford.edu/botryllusgenome/download/botznik-chr.fa"
run(
    "download-genome",
    ["curl", "--fail", "--location", "--retry", "3", "--output", "botznik-chr.fa", url],
)
from Bio import SeqIO
from Bio.Seq import Seq

# The GTF uses chromosome coordinates; reconstruct every annotated transcript.
genome = {r.id: str(r.seq).upper() for r in SeqIO.parse("botznik-chr.fa", "fasta")}
reference = {
    r.id: str(r.seq).upper()
    for r in SeqIO.parse(root / "references/botznik-transcripts.fa", "fasta")
}
exons = {}
for line in (root / "references/nc_nsi_nf.gtf").open():
    if line.startswith("#"):
        continue
    f = line.rstrip().split("\t")
    if f[2] != "exon":
        continue
    a = dict(re.findall(r'(\S+) "([^"]*)"', f[8]))
    exons.setdefault(a["gene_id"], []).append((f[0], int(f[3]), int(f[4]), f[6]))
matched, mismatches = [], []
for gene, expected in reference.items():
    spans = sorted(exons.get(gene, []), key=lambda x: x[1])
    if not spans or any(
        c not in genome or b > len(genome[c]) for c, a, b, strand in spans
    ):
        mismatches.append({"gene": gene, "reason": "missing or out-of-range exons"})
        continue
    actual = "".join(genome[c][a - 1 : b] for c, a, b, strand in spans)
    if spans[0][3] == "-":
        actual = str(Seq(actual).reverse_complement())
    if actual == expected:
        matched.append(gene)
    else:
        mismatches.append(
            {
                "gene": gene,
                "reference_length": len(expected),
                "reconstructed_length": len(actual),
            }
        )
report = {
    "url": url,
    "transcripts": len(reference),
    "exact_matches": len(matched),
    "mismatches": mismatches,
    "contigs": {k: len(v) for k, v in genome.items()},
}
Path("genome-check.json").write_text(json.dumps(report, indent=2) + "\n")
# Record all mismatches; the launch gate is separate so other measurements still run.
selected = list(reference.items())[:500]
Path("transcripts500.fa").write_text(
    "".join(">" + k + "\n" + v + "\n" for k, v in selected)
)
run(
    "alignment",
    [
        py.parent / "minimap2",
        "-x",
        "splice",
        "-c",
        "--cs=long",
        "--secondary=yes",
        "-t",
        "8",
        "botznik-chr.fa",
        "transcripts500.fa",
    ],
)
run(
    "pfam",
    [
        "/share/software/user/open/hmmer/3.4/bin/hmmscan",
        "--cpu",
        "8",
        "--cut_ga",
        "--domtblout",
        "domains.domtblout",
        Path(os.environ["SCRATCH"]) / "data/pfam/Pfam-A.hmm",
        root / "annotation/pilot500.faa",
    ],
)
records = json.loads((root / "trimmed-inputs.validated.json").read_text())
r = max(records, key=lambda r: r["pairs"])
Path("largest-library.json").write_text(json.dumps(r, indent=2) + "\n")
block = (
    (code / "pipelines/botryllus-shazam/main.nf")
    .read_text()
    .split("process COUNT_KMERS {", 1)[1]
    .split("process JOIN_KMERS {", 1)[0]
    .split('"""')[1]
)
for a, b in {
    "${params.dekupl}": str(de),
    "${task.cpus}": "8",
    "${reads[0]}": r["files"][0]["path"],
    "${reads[1]}": r["files"][1]["path"],
    "${run}": r["run"],
}.items():
    block = block.replace(a, b)
assert "${" not in block
Path("count.command.sh").write_text(block)
run("count", ["bash", "-euo", "pipefail", "count.command.sh"])
run(
    "kallisto-index",
    [
        de / "share/dekupl/bin/kallisto",
        "index",
        "-i",
        "reference.idx",
        root / "references/botznik-transcripts.fa",
    ],
)
run(
    "quantify",
    [
        de / "share/dekupl/bin/kallisto",
        "quant",
        "-i",
        "reference.idx",
        "-o",
        r["run"] + ".quant",
        *[x["path"] for x in r["files"]],
    ],
)
run(
    "gene-map",
    [
        py,
        code / "scripts/botryllus_shazam/nextflow_stages.py",
        "genes",
        root / "references/botznik-transcripts.fa",
        root / "references/nc_nsi_nf.gtf",
        r["run"] + ".quant",
    ],
)
run(
    "checksums",
    [
        "sha256sum",
        "botznik-chr.fa",
        Path(os.environ["SCRATCH"]) / "data/pfam/Pfam-A.hmm",
        root / "references/botznik-transcripts.fa",
        root / "references/nc_nsi_nf.gtf",
    ],
)
Path("complete.json").write_text(
    json.dumps(
        {"genome_matches_all": not mismatches, "library": r["run"], "pairs": r["pairs"]}
    )
    + "\n"
)
