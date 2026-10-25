"""Known-answer read recruitment, extension and Branchwater checks inside Slurm."""

import csv
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys

root, code, out = map(Path, sys.argv[1:])
out.mkdir(exist_ok=False)
binary = out / "recruit"
subprocess.run(
    ["rustc", "-O", str(code / "recruit_dekupl_reads.rs"), "-o", str(binary)],
    check=True,
)
rng = random.Random(8723)
seq = "".join(rng.choice("ACGT") for _ in range(150))
seed = seq[10:41]
rc = seed.translate(str.maketrans("ACGT", "TGCA"))[::-1]
(out / "seeds.tsv").write_text(seed + "\tDKfixture\n")
for mate in (1, 2):
    with gzip.open(out / f"r{mate}.fq.gz", "wt") as f:
        for i in range(25):
            s = seq if mate == 1 and i < 24 else "A" * 150
            f.write(f"@read{i}/{mate}\n{s}\n+\n" + "I" * 150 + "\n")

args = [
    str(binary),
    str(out / "seeds.tsv"),
    str(out / "r1.fq.gz"),
    str(out / "r2.fq.gz"),
    str(out / "matched1.fq.gz"),
    str(out / "matched2.fq.gz"),
    str(out / "hits.tsv.gz"),
]
counts = json.loads(subprocess.check_output(args, text=True))
assert counts == dict(input_pairs=25, recruited_pairs=24, occurrences=24), counts
with gzip.open(out / "hits.tsv.gz", "rt") as f:
    hits = list(csv.DictReader(f, delimiter="\t"))
assert len(hits) == 24 and all(
    r["query_id"] == "DKfixture" and r["mate"] == "1" and r["position_0based"] == "10"
    for r in hits
)
with gzip.open(out / "matched2.fq.gz", "rt") as f:
    assert (
        sum(1 for _ in f) == 96
    )  # Non-matching mates are retained once with the matching pair.
(out / "reads.txt").write_text(
    str(out / "matched1.fq.gz") + "\n" + str(out / "matched2.fq.gz") + "\n"
)
(out / "anchors.tsv").write_text("anchor\n" + seed + "\n" + rc + "\n")
compactors = root.parent / "oyster-shazam-20261023/software/compactors"
subprocess.run(
    [
        str(compactors),
        "--all_anchors",
        "--kmer_len",
        "31",
        "--num_threads",
        "2",
        "--reads_buffer_gb",
        "1",
        str(out / "reads.txt"),
        str(out / "anchors.tsv"),
        str(out / "compactors.tsv"),
    ],
    check=True,
)
with (out / "compactors.tsv").open() as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
assert rows and any(
    len(r["compactor"]) > 31 and seed in r["compactor"] for r in rows
), rows
(out / "toy.fna").write_text(">DKfixture\n" + seq + "\n")
with (out / "input.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["name", "genome_filename", "protein_filename"])
    w.writerow(["toy", str(out / "toy.fna"), ""])
base = [sys.executable, "-m", "sourmash", "scripts"]
subprocess.run(
    [
        *base,
        "manysketch",
        str(out / "input.csv"),
        "--singleton",
        "-p",
        "dna,k=31,scaled=1",
        "-c",
        "2",
        "-o",
        str(out / "toy.sig.zip"),
    ],
    check=True,
)
subprocess.run(
    [
        *base,
        "manysearch",
        str(out / "toy.sig.zip"),
        str(out / "toy.sig.zip"),
        "-m",
        "DNA",
        "-k",
        "31",
        "-s",
        "1",
        "-t",
        "0",
        "-A",
        "-N",
        "-c",
        "2",
        "-o",
        str(out / "self.csv"),
    ],
    check=True,
)
with (out / "self.csv").open() as f:
    rows = list(csv.DictReader(f))
assert (
    len(rows) == 1
    and rows[0]["query_name"] == "DKfixture"
    and float(rows[0]["containment"]) == 1
), rows
(out / "passed.json").write_text(
    json.dumps(
        dict(
            status="passed",
            synthetic_fixture=True,
            recruitment=counts,
            compactors_extended_seed=True,
            branchwater_manysketch_manysearch_self=True,
            job=os.environ.get("SLURM_JOB_ID"),
        ),
        indent=2,
    )
    + "\n"
)
