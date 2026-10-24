"""Refuse historical or incomplete inputs for full-depth searches."""

import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate(root, discovery, manifest):
    with Path(manifest).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = {r["run_accession"] for r in rows}
    trimmed = json.loads((root / "trimmed-inputs.validated.json").read_text())
    if (
        len(rows) != 64
        or len(expected) != 64
        or len(trimmed) != 64
        or {r["run"] for r in trimmed} != expected
    ):
        raise ValueError("Full-depth queries require every one of the 64 libraries")
    completion = json.loads((discovery / "discovery.validated.json").read_text())
    selection = json.loads((discovery / "all-significant/selection.json").read_text())
    if (
        selection["extension_cap"] is not None
        or selection["rule"] != "pval_opt_corrected < 0.05"
    ):
        raise ValueError("Capped or different anchor-selection rule")
    source = discovery / "full.after_correction.all_anchors.tsv"
    if sha(source) != selection["source_sha256"]:
        raise ValueError("SPLASH tested population changed")
    with source.open() as stream:
        all_rows = list(csv.DictReader(stream, delimiter="\t"))
    expected_anchors = {
        r["anchor"] for r in all_rows if float(r["pval_opt_corrected"]) < 0.05
    }
    with (discovery / "all-significant/selected_anchors.tsv").open() as stream:
        selected_rows = list(csv.DictReader(stream, delimiter="\t"))
    selected = {r["anchor"] for r in selected_rows}
    if (
        selected != expected_anchors
        or len(selected_rows) != len(selected)
        or len(all_rows) != selection["tested_anchors"]
        or len(selected) != completion["selected_anchors"]
    ):
        raise ValueError(
            "Selected anchors do not equal the complete strict adjusted-P set"
        )
    for name, digest in completion["outputs"].items():
        if sha(discovery / name) != digest:
            raise ValueError("Full-depth discovery output changed: " + name)
    for name, count in [
        ("peptides.fasta", completion["unique_peptides"]),
        ("extensions.fasta", completion["extension_records"]),
    ]:
        with (discovery / name).open() as f:
            if sum(line.startswith(">") for line in f) != count:
                raise ValueError("Full-depth FASTA record count differs")
    return dict(
        scope="full-depth 64-library discovery; every adjusted P < 0.05 anchor; no cap",
        libraries=64,
        selected_anchors=len(selected),
        discovery_sha256=sha(discovery / "discovery.validated.json"),
        source_sha256=selection["source_sha256"],
        query_sha256={
            n: sha(discovery / n) for n in ("peptides.fasta", "extensions.fasta")
        },
        unique_peptides=completion["unique_peptides"],
        extension_records=completion["extension_records"],
    )
