"""Stream native counts into balanced genotype units and deterministic partitions."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

raw, genes, design_file, out = map(Path, sys.argv[1:5])
assays = sys.argv[5:] or ["genes", "kmers"]
if any(a not in {"genes", "kmers"} for a in assays) or len(set(assays)) != len(assays):
    raise ValueError("Expected genes and/or kmers assays")
design = json.loads(design_file.read_text())
out.mkdir(exist_ok=False)
for contrast in design["contrasts"]:
    units = [(g, c) for g in contrast["genotypes"] for c in ("Control", "Treatment")]
    names = {unit: f"U{i:02d}_{unit[1]}" for i, unit in enumerate(units, 1)}
    mapping = {r["run"]: (r["genotype"], r["condition"]) for r in contrast["libraries"]}
    for assay, source in [("genes", genes), ("kmers", raw)]:
        if assay not in assays:
            continue
        folder = out / contrast["name"] / assay
        folder.mkdir(parents=True)
        with (folder / "units.tsv").open("w") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["unit", "genotype", "condition"])
            w.writerows([names[u], *u] for u in units)
        chunks = 1 if assay == "genes" else 32
        handles = [
            gzip.open(folder / f"part-{i:03d}.tsv.gz", "wt") for i in range(chunks)
        ]
        writers = [csv.writer(h, delimiter="\t", lineterminator="\n") for h in handles]
        header = ["tag", *[names[u] for u in units]]
        for w in writers:
            w.writerow(header)
        tested = 0
        with gzip.open(folder / "pooled.tsv.gz", "wt") as merged, gzip.open(
            source, "rt"
        ) as f:
            allwriter = csv.writer(merged, delimiter="\t", lineterminator="\n")
            allwriter.writerow(header)
            reader = csv.reader(f, delimiter="\t")
            columns = next(reader)
            index = {name: i for i, name in enumerate(columns)}
            if not set(mapping) <= set(index):
                raise ValueError("Native count matrix missing libraries")
            for row in reader:
                sums = {u: 0.0 for u in units}
                for run, unit in mapping.items():
                    sums[unit] += float(row[index[run]])
                values = [int(round(sums[u])) for u in units]
                if sum(v >= 5 for v in values) < 2:
                    continue
                line = [row[0], *values]
                allwriter.writerow(line)
                part = (
                    int.from_bytes(hashlib.sha256(row[0].encode()).digest()[:8], "big")
                    % chunks
                )
                writers[part].writerow(line)
                tested += 1
        for h in handles:
            h.close()
        (folder / "eligible.json").write_text(
            json.dumps(
                dict(
                    features=tested,
                    partitions=chunks,
                    unit_count=len(units),
                    filter="count >=5 in >=2 pooled genotype/condition units",
                    design_sha256=hashlib.sha256(design_file.read_bytes()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
