"""Independently verify new query chunks and rebuild the small reference with full versions."""

import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import importlib.metadata
from Bio import SeqIO
from Bio.Seq import Seq
import sourmash
from artifacts import sha, write_new
from audit_completed import dna_records, load, require
from corrected_public_pilot import pilot


def independent(dna):
    settings = [
        ("protein", 7, {"is_protein": True}),
        ("dayhoff", 12, {"dayhoff": True}),
        ("hp", 30, {"hp": True}),
    ]
    result = {
        mol: sourmash.MinHash(n=0, ksize=k, scaled=10, **flags)
        for mol, k, flags in settings
    }
    for strand in (Seq(dna), Seq(dna).reverse_complement()):
        for offset in range(3):
            n = (len(strand) - offset) // 3
            if n <= 0:
                continue
            peptide = str(strand[offset : offset + 3 * n].translate(table=1))
            # Independent boundary implementation. X is included by [^*].
            for match in re.finditer(r"[^*]{30,}", peptide):
                segment = match.group()
                for mol, k, _ in settings:
                    for i in range(len(segment) - k + 1):
                        result[mol].add_protein(segment[i : i + k])
    return result


def verify(manifest_path, archive_path, output, executable):
    output.mkdir()
    require(bool(os.environ.get("SLURM_JOB_ID")), "Requires Slurm")
    manifest = load(manifest_path)
    require(manifest["originals"] == 380867, "Incomplete build")
    require(
        manifest["protocol"] == "six-frame-standard-stop-only-X-preserved-min30-v1",
        "Wrong protocol",
    )
    build = manifest_path.parent
    require(
        sha(build / "build-inputs.json") == manifest["build_inputs_sha256"],
        "Build provenance changed",
    )
    meta = load(build / "build-inputs.json")
    for path, expected in meta["source_files_sha256"].items():
        require(sha(path) == expected, "Original source changed")
    replay = Path(os.environ["GROUP_SCRATCH"]) / "oyster-first-public-linear-20261025"
    records = dna_records(replay, load(replay / "manifest.json"))
    scratch = Path(os.environ["L_SCRATCH_JOB"]) / "replacement-audit"
    scratch.mkdir()
    count = 0
    for item in manifest["chunks"]:
        bundle = build / item["file"]
        require(sha(bundle) == item["sha256"], "Chunk changed")
        with tarfile.open(bundle) as archive:
            archive.extractall(scratch, filter="data")
        chunk = scratch / item["file"].removesuffix(".tar.gz")
        expected = {}
        with (chunk / "original.fa").open() as handle:
            for record in SeqIO.parse(handle, "fasta"):
                require(
                    str(record.seq) == records.pop(record.id),
                    "Original DNA or ID mismatch",
                )
                expected[record.id] = independent(str(record.seq))
        receipt = load(chunk / "sketches/validated.json")
        require(receipt == item["validation"], "Sketch receipt changed")
        for collection in receipt["collections"]:
            path = chunk / "sketches" / collection["file"]
            require(sha(path) == collection["sha256"], "Collection changed")
            mol, scale = collection["file"].removesuffix(".sig.gz").split("-")
            found = set()
            for sig in sourmash.load_file_as_signatures(str(path)):
                require(
                    sig.name not in found and sig.name in expected,
                    "Duplicate or unknown original",
                )
                require(
                    sig.minhash
                    == expected[sig.name][mol].downsample(scaled=int(scale)),
                    "Independent word hashes differ",
                )
                found.add(sig.name)
            require(
                found == set(expected), "Missing originals including empty sketches"
            )
        count += len(expected)
        shutil.rmtree(chunk)
        print("VERIFIED", count, flush=True)
    require(not records and count == 380867, "Incomplete original coverage")
    # Rebuild only the small corrected reference to record every runtime dependency.
    imported = scratch / "imported"
    imported.mkdir()
    with tarfile.open(archive_path) as archive:
        members = [
            m for m in archive if m.name.startswith("./imported/corrected-sketches/")
        ]
        require(bool(members), "Missing pilot query inputs")
        archive.extractall(imported, members=members, filter="data")
    pilot(
        output / "fresh-reference", imported / "imported/corrected-sketches", executable
    )
    for i in (0, 1):
        log = (output / "fresh-reference" / f"trim-{i}.log").read_text()
        require(
            "Trim Galore version: 0.6.10" in log and "Cutadapt version: 5.2" in log,
            "Unexpected trimming tools",
        )
    code = Path(__file__).resolve().parents[1]
    names = [
        "peptide_protocol.py",
        "sketch_corrected_peptides.py",
        "nextflow/corrected_public_pilot.py",
        "nextflow/verify_replacements.py",
        "nextflow/artifacts.py",
        "nextflow/audit_completed.py",
    ]
    write_new(
        output / "verified-provenance.json",
        dict(
            protocol=manifest["protocol"],
            original_queries=count,
            query_manifest_sha256=sha(manifest_path),
            query_build_inputs_sha256=sha(build / "build-inputs.json"),
            source_code_sha256={n: sha(code / n) for n in names},
            tool_versions={
                n: importlib.metadata.version(n)
                for n in ("sourmash", "sourmash_plugin_branchwater", "biopython")
            },
            python=sys.version,
            argv=sys.argv,
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            reference_pilot_sha256=sha(
                output / "fresh-reference/corrected-public-pilot.json"
            ),
            reference_sketch_manifest_sha256=sha(
                output / "fresh-reference/sketches/validated.json"
            ),
            reference_translation_manifest_sha256=sha(
                output / "fresh-reference/translation/complete.json"
            ),
            reference_tool_versions={"trim_galore": "0.6.10", "cutadapt": "5.2"},
            validation="Every query hash independently rebuilt one word at a time; corrected reference freshly rebuilt with runtime versions",
            scope="380867 audited queries and SRR5945967; historical outputs remain unchanged and their missing provenance remains unknown",
            production_rollout=False,
        ),
    )


if __name__ == "__main__":
    verify(*(Path(a).resolve() for a in sys.argv[1:]))
