"""Bounded corrected-protocol pilot: Branchwater segments, original-ID aggregation.

Run on Sherlock compute after renewed access. This holds the pilot in memory;
archive-scale streaming and translated Nextflow integration are pending.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import sourmash
from sourmash.save_load import SaveSignaturesToLocation
from peptide_protocol import PROTOCOL

SETTINGS = [
    ("protein", 7, {"is_protein": True}),
    ("dayhoff", 12, {"dayhoff": True}),
    ("hp", 30, {"hp": True}),
]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(source, output, executable, threads):
    version = importlib.metadata.version("sourmash_plugin_branchwater")
    if version != "0.9.15":
        raise ValueError(f"Require Branchwater 0.9.15; found {version}")
    # Resolve sourmash alongside this interpreter to avoid testing a different install.
    import sys

    if executable.resolve() != (Path(sys.executable).parent / "sourmash").resolve():
        raise ValueError("sourmash must be from this Python environment")
    meta = json.loads((source / "complete.json").read_text())
    if meta["protocol"] != PROTOCOL:
        raise ValueError("Wrong translation protocol")
    for name in ("origins.jsonl", "segments.jsonl", "peptides.fa"):
        if sha(source / name) != meta["outputs"][name]:
            raise ValueError("Translation output changed")
    origins = [
        json.loads(line) for line in (source / "origins.jsonl").read_text().splitlines()
    ]
    segments = [
        json.loads(line)
        for line in (source / "segments.jsonl").read_text().splitlines()
    ]
    ids = [row["origin_id"] for row in origins]
    if len(ids) != len(set(ids)) or len(ids) != meta["origins"]:
        raise ValueError("Original membership mismatch")
    expected = {row["segment_id"]: row for row in segments}
    if len(expected) != len(segments) or any(
        row["origin_id"] not in ids for row in segments
    ):
        raise ValueError("Segment membership mismatch")
    output.mkdir()
    merged = {
        mol: {name: sourmash.MinHash(n=0, ksize=k, scaled=10, **flags) for name in ids}
        for mol, k, flags in SETTINGS
    }
    seen = set()
    if segments:
        with (output / "inputs.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "genome_filename", "protein_filename"])
            writer.writerow(["segments", "", str((source / "peptides.fa").resolve())])
        command = [
            str(executable),
            "scripts",
            "manysketch",
            str(output / "inputs.csv"),
            "--singleton",
            "-c",
            str(threads),
            "-o",
            str(output / "segments.zip"),
        ]
        for mol, k, _ in SETTINGS:
            command += ["-p", f"{mol},k={k},scaled=10"]
        with (output / "manysketch.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        for sig in sourmash.load_file_as_signatures(str(output / "segments.zip")):
            mol = sig.minhash.moltype.lower()
            pair = sig.name, mol
            if sig.name not in expected or mol not in merged or pair in seen:
                raise ValueError("Unexpected or duplicate segment sketch")
            seen.add(pair)
            row = expected[sig.name]
            k, flags = next((k, flags) for m, k, flags in SETTINGS if m == mol)
            check = sourmash.MinHash(n=0, ksize=k, scaled=10, **flags)
            check.add_protein(row["peptide"])
            if check != sig.minhash:
                raise ValueError(
                    "Branchwater differs from peptide API, including internal X handling"
                )
            merged[mol][row["origin_id"]].merge(sig.minhash)
    if seen != {(name, mol) for name in expected for mol, _, _ in SETTINGS}:
        raise ValueError("Missing segment sketches")
    summaries = []
    for mol, _, _ in SETTINGS:
        for scale in (10, 100):
            destination = output / f"{mol}-{scale}.sig.gz"
            with SaveSignaturesToLocation(str(destination)) as save:
                for name in ids:
                    save.add(
                        sourmash.SourmashSignature(
                            merged[mol][name].downsample(scaled=scale), name=name
                        )
                    )
            saved = list(sourmash.load_file_as_signatures(str(destination)))
            if [s.name for s in saved] != ids:
                raise ValueError("Aggregated original membership mismatch")
            summaries.append(
                dict(
                    file=destination.name,
                    sha256=sha(destination),
                    origins=len(saved),
                    empty=sum(not len(s.minhash) for s in saved),
                )
            )
    (output / "validated.json").write_text(
        json.dumps(
            dict(
                protocol=PROTOCOL,
                branchwater=version,
                translation_manifest_sha256=sha(source / "complete.json"),
                segment_comparisons=len(seen),
                internal_X_segments=sum("X" in row["peptide"] for row in segments),
                collections=summaries,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sourmash", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    build(args.source, args.output, args.sourmash, args.threads)
