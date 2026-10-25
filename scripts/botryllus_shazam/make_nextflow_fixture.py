"""Generate deterministic software-test reads, never experimental data."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

out = Path(sys.argv[1]).resolve()
out.mkdir()
rng = random.Random(7301)
seqs = {
    f"T{i:03d}": "".join(rng.choice("ACGT") for _ in range(330)) for i in range(100)
}
(out / "reference.fa").write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
(out / "reference.gtf").write_text(
    "".join(
        f'chr\tfixture\texon\t1\t330\t.\t+\t.\tgene_id "G{i:03d}"; transcript_id "{t}";\n'
        for i, t in enumerate(seqs)
    )
)
records = []
samples = []
contrasts = []
for contrast in range(3):
    libraries = []
    for genotype in range(3):
        for cond in ("Control", "Treatment"):
            run = f"S{contrast}{genotype}{cond[0]}"
            count = 0
            files = []
            handles = [gzip.open(out / f"{run}_{m}.fastq.gz", "wt") for m in (1, 2)]
            for i, (t, seq) in enumerate(seqs.items()):
                n = max(20, int(rng.gammavariate(4, 20))) * (
                    4 if cond == "Treatment" and i < 2 else 1
                )
                for repeat in range(n):
                    start = rng.randrange(100)
                    a, b = (
                        seq[start : start + 100],
                        seq[start + 130 : start + 230].translate(
                            str.maketrans("ACGT", "TGCA")
                        )[::-1],
                    )
                    for mate, read in enumerate((a, b), 1):
                        handles[mate - 1].write(
                            f"@{run}_{count}/{mate}\n{read}\n+\n"
                            + "I" * len(read)
                            + "\n"
                        )
                    count += 1
            for h in handles:
                h.close()
            for mate in (1, 2):
                p = out / f"{run}_{mate}.fastq.gz"
                files.append(
                    dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                )
            records.append(dict(run=run, pairs=count, files=files))
            samples.append(run)
            libraries.append(dict(run=run, genotype=f"G{genotype}", condition=cond))
    contrasts.append(
        dict(
            name=f"fixture{contrast}", genotypes=["G0", "G1", "G2"], libraries=libraries
        )
    )
(out / "samples.tsv").write_text("run\n" + "\n".join(samples) + "\n")
(out / "design.json").write_text(
    json.dumps(
        dict(
            samples_sha256=hashlib.sha256(
                (out / "samples.tsv").read_bytes()
            ).hexdigest(),
            contrasts=contrasts,
        )
    )
)
(out / "inputs.json").write_text(json.dumps(records))
(out / "paper.tsv").write_text("gene_id\tcategory\nG000\tsoftware fixture\n")
(out / "annotations.tsv").write_text(
    "record_id\tname\nfixture-target\tsoftware fixture\n"
)
with (out / "targets.tsv").open("w") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["database", "kind", "mode", "k", "scaled", "paths"])
    for mode, k in [("dna", 31), ("protein", 7), ("dayhoff", 12), ("hp", 30)]:
        w.writerow(
            [
                "fixture-target",
                "reads",
                mode,
                k,
                5,
                ";".join(x["path"] for x in records[0]["files"]),
            ]
        )
params = {
    key: str(out / name)
    for key, name in [
        ("inputs", "inputs.json"),
        ("samples", "samples.tsv"),
        ("design", "design.json"),
        ("reference", "reference.fa"),
        ("gtf", "reference.gtf"),
        ("paper", "paper.tsv"),
        ("annotations", "annotations.tsv"),
        ("targets", "targets.tsv"),
    ]
}
params.update(expected_libraries=18, outdir=str(out / "results"))
(out / "params.json").write_text(json.dumps(params))
print(out / "params.json")
