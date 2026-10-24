"""Check upstream DE-kupl counting and gene counting on a known small input."""

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import random
import subprocess
import sys

root, image = map(Path, sys.argv[1:])
out = root / "dekupl-setup/smoke"
out.mkdir(exist_ok=False)
reads = out / "reads"
reads.mkdir()
rng = random.Random(91)
dna = "".join(rng.choice("ACGT") for _ in range(1200))
transcript = out / "transcripts.fa"
transcript.write_text(">T1\n" + dna + "\n")
mapfile = out / "tx2gene.tsv"
mapfile.write_text("T1\tG1\n")
names = ["S1", "S2", "S3", "S4"]
expected = {}
comp = str.maketrans("ACGT", "TGCA")
for sample in names:
    counts = Counter()
    for mate in (1, 2):
        with gzip.open(reads / f"{sample}_{mate}.fastq.gz", "wt") as f:
            for i in range(240):
                start = (i % 6) * 120
                seq = (
                    dna[start : start + 100]
                    if mate == 1
                    else dna[start + 130 : start + 230].translate(comp)[::-1]
                )
                f.write(f"@r{i}/{mate}\n{seq}\n+\n" + ("I" * len(seq)) + "\n")
                for j in range(len(seq) - 30):
                    k = seq[j : j + 31]
                    counts[min(k, k.translate(comp)[::-1])] += 1
    expected[sample] = counts
config = dict(
    fastq_dir=str(reads),
    kmer_length=31,
    lib_type="unstranded",
    output_dir=str(out / "results"),
    masking="nomask",
    ref_kallisto=str(transcript),
    transcript_to_gene=str(mapfile),
    tmp_dir=str(Path(__import__("os").environ["L_SCRATCH_JOB"])),
    diff_method="DESeq2",
    dekupl_counter=dict(min_recurrence=2, min_recurrence_abundance=5),
    diff_analysis=dict(
        condition=dict(A="Control", B="Treatment"),
        pvalue_threshold=0.05,
        log2fc_threshold=0,
    ),
    samples=[
        dict(name=n, condition="Control" if i < 2 else "Treatment")
        for i, n in enumerate(names)
    ],
)
p = out / "config.json"
p.write_text(json.dumps(config, indent=2) + "\n")
raw = out / "results/kmer_counts/raw-counts.tsv.gz"
gene = out / "results/gene_expression/kallisto/gene_counts.tsv.gz"
cmd = [
    "apptainer",
    "exec",
    str(image),
    "snakemake",
    "-s",
    "/dekupl/Snakefile",
    "--configfile",
    str(p),
    "-j",
    "4",
    "--resources",
    "ram=12000",
    str(raw),
    str(gene),
]
subprocess.run(cmd, cwd=out, check=True)
rows = list(csv.DictReader(gzip.open(raw, "rt"), delimiter="\t"))
if not rows:
    raise ValueError("Upstream DE-kupl produced no toy k-mers")
for row in rows:
    for n in names:
        if int(row[n]) != expected[n][row["tag"]]:
            raise ValueError("Canonical k-mer toy count mismatch")
genes = list(csv.DictReader(gzip.open(gene, "rt"), delimiter="\t"))
if len(genes) != 1 or genes[0].get("feature") != "G1":
    raise ValueError("Toy gene mapping differs")
(root / "dekupl-setup/validated.json").write_text(
    json.dumps(
        dict(
            fixture=True,
            raw_kmers_checked=len(rows),
            samples=4,
            gene_rows=len(genes),
            command=cmd,
            interpretation="Software test only; not experimental results",
        ),
        indent=2,
    )
    + "\n"
)
