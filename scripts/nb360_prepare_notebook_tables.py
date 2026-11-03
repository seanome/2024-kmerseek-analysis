#!/usr/bin/env python3
"""Copy the small result tables the notebook reads, and keep only primary-cutoff landing rows.

Runs on Sherlock, where the full landing tables are (drosophila gbmr7 k11 alone is 10 GB):

    python3 scripts/nb360_prepare_notebook_tables.py <run>/results <staging dir>

then the staging dir is copied to notebooks/360_data/results. The full tables stay on Sherlock;
`source_manifest.tsv` records each one's sha256, size and row counts.
"""

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

PRIMARY_E = 1.0
SMALL = ["reads/true_counts.tsv", "distance/{o}.seed_blastp.tsv",
         "standard/{o}.standard.calls.tsv", "standard/{o}.standard.summary.tsv"]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("source", type=Path, help="the run's results/ directory")
ap.add_argument("dest", type=Path, help="where to write the notebook's copy")
args = ap.parse_args()

args.dest.mkdir(parents=True, exist_ok=True)
shutil.copy2(args.source / "summary.tsv", args.dest / "summary.tsv")
manifest = []
for org_dir in sorted(p for p in args.source.iterdir() if p.is_dir()):
    o = org_dir.name
    for rel in SMALL:
        rel = rel.format(o=o)
        (args.dest / o / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(org_dir / rel, args.dest / o / rel)
    (args.dest / o / "landings").mkdir(parents=True, exist_ok=True)
    for summ in sorted((org_dir / "landings").glob("*.summary.tsv")):
        shutil.copy2(summ, args.dest / o / "landings" / summ.name)
    for source in sorted((org_dir / "landings").glob("*.landings.tsv.gz")):
        target = args.dest / o / "landings" / source.name.replace(".landings.", ".primary_landings.")
        # Column 7 is region_evalue; the header row passes through.
        cmd = (f"pigz -dc '{source}' | awk -F'\\t' 'NR==1{{print; next}} {{n++}} $7<={PRIMARY_E}{{k++; print}}"
               f" END{{print n+0, k+0 > \"/dev/stderr\"}}' | gzip -c > '{target}'")
        res = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True, stderr=subprocess.PIPE, universal_newlines=True)
        total, kept = map(int, res.stderr.split())
        sha = hashlib.sha256()
        with source.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                sha.update(block)
        manifest.append(dict(source=f"{o}/landings/{source.name}", sha256=sha.hexdigest(),
                             bytes=source.stat().st_size, rows=total, primary_rows=kept,
                             notebook_file=str(target.relative_to(args.dest))))
        print(source.name, total, kept, flush=True)
with (args.dest / "source_manifest.tsv").open("w") as fh:
    w = csv.DictWriter(fh, manifest[0].keys(), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(manifest)
print(f"{len(manifest)} full landing tables inventoried")
