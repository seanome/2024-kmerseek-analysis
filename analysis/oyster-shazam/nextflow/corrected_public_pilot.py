"""Compare corrected 2000-query sketches with one complete, identically trimmed run."""

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from Bio import SeqIO
import sourmash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peptide_protocol import export, PROTOCOL
from sketch_corrected_peptides import build, SETTINGS
from artifacts import sha, write_new


def pilot(root, queries, executable):
    root.mkdir()
    query_meta = json.loads((queries / "validated.json").read_text())
    assert query_meta["protocol"] == PROTOCOL
    for collection in query_meta["collections"]:
        assert sha(queries / collection["file"]) == collection["sha256"]
    group = Path(os.environ["GROUP_SCRATCH"])
    historical = group / "oyster-stopfree-public-pilot-linear-20261025"
    meta = json.loads((historical / "validated.json").read_text())["reference_metadata"]
    urls = meta["fastq_ftp"].split(";")
    md5s = meta["fastq_md5"].split(";")
    assert len(urls) == len(md5s)
    cutadapt = (
        Path(os.environ["GROUP_HOME"]) / "opt/oyster-mvp/envs/shazam/bin/cutadapt"
    )
    trim = (
        Path(os.environ["GROUP_HOME"])
        / "opt/oyster-mvp/sw/TrimGalore-0.6.10/trim_galore"
    )
    reference_fasta = root / "reference.fa"
    source_files = []
    with reference_fasta.open("w") as fasta:
        for i, (url, expected) in enumerate(zip(urls, md5s)):
            raw = root / f"reads-{i}.fq.gz"
            with (root / f"download-{i}.log").open("w") as log:
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--retry",
                        "3",
                        "--speed-limit",
                        "1024",
                        "--speed-time",
                        "60",
                        "-o",
                        str(raw),
                        "https://" + url,
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            with raw.open("rb") as handle:
                assert hashlib.file_digest(handle, "md5").hexdigest() == expected
            with (root / f"trim-{i}.log").open("w") as log:
                subprocess.run(
                    [
                        str(trim),
                        "--path_to_cutadapt",
                        str(cutadapt),
                        "--cores",
                        "1",
                        "--quality",
                        "20",
                        "--length",
                        "20",
                        "--gzip",
                        "--output_dir",
                        str(root),
                        str(raw),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            trimmed = root / f"reads-{i}_trimmed.fq.gz"
            count = 0
            with gzip.open(trimmed, "rt") as handle:
                for j, record in enumerate(SeqIO.parse(handle, "fastq")):
                    fasta.write(f">read:{i}:{j}\n{record.seq}\n")
                    count += 1
            source_files.append(
                dict(
                    url=url,
                    md5=expected,
                    raw_sha256=sha(raw),
                    trimmed_sha256=sha(trimmed),
                    trimmed_records=count,
                )
            )
    # Check the exact reference membership/DNA against the historical pilot sample.
    expected = {}
    sample = group / "oyster-min20-pilot-20261025/sample.tsv"
    with sample.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["group"] == "reference":
                expected[row["id"]] = row["dna_sha256"]
    with reference_fasta.open() as handle:
        observed = {
            r.id: hashlib.sha256(str(r.seq).encode()).hexdigest()
            for r in SeqIO.parse(handle, "fasta")
        }
    assert observed == expected, "Reference differs from original pilot DNA"
    export(reference_fasta, root / "translation")
    build(root / "translation", root / "sketches", executable, 4)
    measurements = []
    for mol, k, flags in SETTINGS:
        for scale in (10, 100):
            refs = list(
                sourmash.load_file_as_signatures(
                    str(root / "sketches" / f"{mol}-{scale}.sig.gz")
                )
            )
            combined = set()
            for s in refs:
                combined.update(s.minhash.hashes)
            qs = list(
                sourmash.load_file_as_signatures(str(queries / f"{mol}-{scale}.sig.gz"))
            )
            assert len(qs) == len({s.name for s in qs}) == 2000
            for label in ("splash", "dekupl"):
                selected = [
                    s for s in qs if s.name.startswith("splash:") == (label == "splash")
                ]
                hits = {
                    s.name: len(set(s.minhash.hashes) & combined)
                    for s in selected
                    if set(s.minhash.hashes) & combined
                }
                assert len(selected) == 1000
                measurements.append(
                    dict(
                        group=label,
                        moltype=mol,
                        k=k,
                        scaled=scale,
                        queries=len(selected),
                        nonempty=sum(bool(len(s.minhash)) for s in selected),
                        matches=len(hits),
                        match_intersections=hits,
                        reference_hashes=len(combined),
                    )
                )
    write_new(
        root / "corrected-public-pilot.json",
        dict(
            protocol=PROTOCOL,
            scope="1000 systematic first-batch full-depth SPLASH extensions and 1000 systematic DE-kupl contigs against one complete public run; candidate sequence matches only",
            reference=meta["run_accession"],
            reference_records=len(observed),
            reference_matches_historical_dna=True,
            sources=source_files,
            query_validation_sha256=sha(queries / "validated.json"),
            measurements=measurements,
        ),
    )


if __name__ == "__main__":
    pilot(*(Path(a).resolve() for a in sys.argv[1:]))
