"""Rebuild corrected query collections with provenance; preserve every attempt.

This covers the first audited SPLASH batch and all selected DE-kupl contigs.
It does not replace historical files or claim archive-wide coverage.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from artifacts import sha, write_new
from audit_completed import dna_records, load, require

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peptide_protocol import export, PROTOCOL
from sketch_corrected_peptides import build


def code_record(root):
    names = [
        "peptide_protocol.py",
        "sketch_corrected_peptides.py",
        "nextflow/build_provenance.py",
        "nextflow/audit_completed.py",
        "nextflow/artifacts.py",
    ]
    return {name: sha(root / name) for name in names}


def run(output, executable, threads):
    require(bool(os.environ.get("SLURM_JOB_ID")), "Requires Slurm")
    code = Path(__file__).resolve().parents[1]
    source = Path(os.environ["GROUP_SCRATCH"]) / "oyster-first-public-linear-20261025"
    before = code_record(code)
    versions = {
        n: importlib.metadata.version(n)
        for n in ("sourmash", "sourmash_plugin_branchwater", "biopython")
    }
    require(versions["sourmash_plugin_branchwater"] == "0.9.15", "Wrong Branchwater")
    source_manifest = load(source / "manifest.json")
    records = dna_records(source, source_manifest)
    require(len(records) == 380867, "Wrong audited source membership")
    # Bind every source file actually read by dna_records, including the selected table.
    de = Path(os.environ["SCRATCH"]) / "oyster-dekupl-queries-20261024"
    selection = load(de / "selection.json")
    paths = [
        source / "manifest.json",
        de / "selection.json",
        de / "all-significant-contigs.tsv.gz",
        Path(selection["source"]),
        Path(os.environ["SCRATCH"])
        / "oyster-full-extensions-20261023/splash-trimmed-all/batch-000/compactors.tsv",
    ]
    sources = {str(p): sha(p) for p in paths}
    provenance = dict(
        protocol=PROTOCOL,
        code_sha256=before,
        source_files_sha256=sources,
        tools=versions,
        python=sys.version,
        argv=sys.argv,
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        source_scope="First audited SPLASH batch and all selected DE-kupl contigs; upstream historical generation is not re-certified",
    )
    write_new(output / "build-inputs.json", provenance)
    scratch = Path(os.environ["L_SCRATCH_JOB"]) / "corrected-chunks"
    scratch.mkdir()
    entries = []
    items = list(records.items())
    started = time.monotonic()
    for start in range(0, len(items), 2000):
        number = start // 2000
        chunk = scratch / f"chunk-{number:04d}"
        chunk.mkdir()
        fasta = chunk / "original.fa"
        with fasta.open("x") as handle:
            for name, dna in items[start : start + 2000]:
                handle.write(f">{name}\n{dna}\n")
        export(fasta, chunk / "translation")
        build(chunk / "translation", chunk / "sketches", executable, threads)
        receipt = load(chunk / "sketches/validated.json")
        # Keep chunks as a few archives rather than thousands of shared-filesystem files.
        target = output / f"chunk-{number:04d}.tar.gz"
        partial = target.with_suffix(".gz.partial")
        with partial.open("xb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz") as archive:
                archive.add(chunk, arcname=chunk.name)
        os.link(partial, target)
        partial.unlink()
        entries.append(
            dict(
                file=target.name,
                sha256=sha(target),
                first_origin=items[start][0],
                origins=receipt["collections"][0]["origins"],
                validation=receipt,
            )
        )
        # A published chunk and its original sequences are durable before cleanup.
        shutil.rmtree(chunk)
        print(
            json.dumps(
                dict(
                    chunk=number,
                    originals=min(start + 2000, len(items)),
                    elapsed_seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
    require(sum(x["origins"] for x in entries) == len(records), "Missing originals")
    require(code_record(code) == before, "Code changed during build")
    require({str(p): sha(p) for p in paths} == sources, "Source changed during build")
    write_new(
        output / "completed.json",
        dict(
            protocol=PROTOCOL,
            originals=len(records),
            chunks=entries,
            build_inputs_sha256=sha(output / "build-inputs.json"),
            status="corrected-query-replacements-built",
            historical_build_provenance="unknown; historical artifacts unchanged",
            reference_provenance="pending",
            production_rollout=False,
            seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3]))
