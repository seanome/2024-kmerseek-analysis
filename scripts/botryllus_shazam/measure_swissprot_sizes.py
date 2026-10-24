"""Measure actual Swiss-Prot FASTA and per-encoding sketch ZIP bytes."""

import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = root / "swissprot-size-comparison-20261023"
out.mkdir(exist_ok=False)
pilot = root / "primary-search-pilot-20261023"
source = root / "references/uniprot_sprot.fasta.gz"
expected = json.loads((pilot / "execution.complete.json").read_text())
with source.open("rb") as f:
    source_hash = hashlib.file_digest(f, "sha256").hexdigest()
if source_hash != expected["source_sha256"]["swissprot"]:
    raise ValueError("Swiss-Prot snapshot changed")
raw_bytes = 0
raw_records = 0
raw_hash = hashlib.sha256()
with gzip.open(source, "rb") as f:
    for line in f:
        raw_bytes += len(line)
        raw_records += line.startswith(b">")
        raw_hash.update(line)
if raw_records != 575748:
    raise ValueError("Swiss-Prot reference record count changed")
rows = []
manifest = pilot / "swissprot.fromfile.csv"
for mode, k in [("protein", 7), ("dayhoff", 12), ("hp", 30)]:
    for scale in [5, 10, 100]:
        target = (
            pilot / f"swissprot.{mode}.scaled5.sig.zip"
            if scale == 5
            else out / f"swissprot.{mode}.scaled{scale}.sig.zip"
        )
        if scale != 5:
            args = [
                sys.executable,
                "-m",
                "sourmash",
                "scripts",
                "manysketch",
                str(manifest),
                "--singleton",
                "-c",
                os.environ["SLURM_CPUS_PER_TASK"],
                "-p",
                f"{mode},k={k},scaled={scale}",
                "-o",
                str(target),
            ]
            with (out / f"{mode}.scaled{scale}.log").open("w") as log:
                subprocess.run(
                    ["/usr/bin/time", "-v", *args],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        with target.open("rb") as f:
            checksum = hashlib.file_digest(f, "sha256").hexdigest()
        rows.append(
            dict(
                alphabet=mode,
                ksize_aa=k,
                scaled=scale,
                bytes=target.stat().st_size,
                path=str(target),
                sha256=checksum,
            )
        )
report = dict(
    source=str(source),
    source_sha256=source_hash,
    reference_records=raw_records,
    raw_fasta_bytes=raw_bytes,
    raw_fasta_sha256=raw_hash.hexdigest(),
    download_fasta_gz_bytes=source.stat().st_size,
    settings=rows,
    sourmash="4.9.4",
    branchwater="0.9.15",
    scope="Same 575748 Swiss-Prot records; exact full-record sequence and header maps in primary-search-pilot-20261023; separate search copy masks unsupported residues as X; per-record ZIP sketch collection sizes, excludes searchable indexes.",
    note="Scaled 10 and 100 are measurements for storage comparison; production Swiss-Prot searches remain scaled 5.",
)
with (out / "sizes.tsv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    w.writeheader()
    w.writerows(rows)
(out / "sizes.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
