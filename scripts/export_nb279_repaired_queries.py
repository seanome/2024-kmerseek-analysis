"""Export minima from repaired query summaries without rerunning searches.

Read one species and alphabet setting at a time. Conventional-tool minima are copied
unchanged, and every input file is fingerprinted in the output manifest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

NAME = re.compile(
    r"^[^.]+\.(?P<alphabet>.+)\.k(?P<ksize>\d+)\.s(?P<scaled>\d+)"
    r"\.lc(?P<mask>true|false)\.(?P<extension>exact|extend-c[0-9.]+)\.queries\.tsv$"
)
SPECIES = ("human", "worm", "yeast", "borrelia", "botryllus")
FIELDS = [
    "alphabet",
    "ksize",
    "scaled",
    "low_complexity_mask",
    "extension",
    "accession",
    "min_region_evalue",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_queries(path: Path, known: set[str]) -> dict[str, float] | None:
    """Validate one repaired batch; None means the setting refused its fit."""
    with path.open() as handle:
        header = handle.readline().rstrip("\n")
        if header.startswith("#nofit\t"):
            return None
        if header != "query_name\tmin_region_evalue":
            raise ValueError(f"Unexpected query header in {path}: {header!r}")
        best: dict[str, float] = {}
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2 or row[0] not in known:
                raise ValueError(f"Invalid or unknown query in {path}: {row!r}")
            value = float(row[1])
            if math.isnan(value) or value < 0:
                raise ValueError(f"Invalid E-value in {path}: {row[1]!r}")
            best[row[0]] = min(best.get(row[0], math.inf), value)
    return best


def export_species(species: str, queries: Path, conventional: Path, out: Path) -> dict:
    """Write one species' unchanged comparator minima and repaired kmerseek minima."""
    tool_file = conventional / f"{species}_protein_tool_min_evalue.csv.gz"
    with gzip.open(tool_file, "rt") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["accession", "phmmer", "jackhmmer", "mmseqs2"]:
            raise ValueError(f"Unexpected conventional-tool columns: {tool_file}")
        accessions = [row["accession"] for row in reader]
    known = set(accessions)
    if not known or len(known) != len(accessions):
        raise ValueError(f"Empty or duplicate conventional query set: {tool_file}")

    groups = defaultdict(list)
    for path in sorted((queries / species).glob("*.queries.tsv")):
        match = NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognised query batch name: {path}")
        key = (
            match["alphabet"],
            int(match["ksize"]),
            int(match["scaled"]),
            match["mask"],
            match["extension"],
        )
        groups[key].append(path)
    if not groups:
        raise ValueError(f"No repaired query files for {species}")

    output = out / f"{species}_protein_kmerseek_min_evalue.csv.gz"
    refused, sources, n_rows, searched = [], [], 0, 0
    with gzip.open(output, "wt", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for key, paths in sorted(groups.items()):
            best: dict[str, float] = {}
            states = set()
            for path in paths:
                batch = read_queries(path, known)
                states.add(batch is not None)
                sources.append(
                    {
                        "file": f"{species}/{path.name}",
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
                if batch is not None:
                    for accession, value in batch.items():
                        best[accession] = min(best.get(accession, math.inf), value)
            if len(states) != 1:
                raise ValueError(f"Setting both refused and searched: {species} {key}")
            if states == {False}:
                refused.append(key)
                continue
            searched += 1
            for accession, value in sorted(best.items()):
                writer.writerow([*key, accession, repr(value)])
                n_rows += 1
    refused_file = out / f"{species}_kmerseek_unfitted.csv"
    with refused_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS[:5])
        writer.writerows(refused)
    tool_copy = out / tool_file.name
    shutil.copyfile(tool_file, tool_copy)
    return {
        "n_proteins": len(known),
        "n_source_files": len(sources),
        "n_settings_searched": searched,
        "n_settings_refused": len(refused),
        "n_output_rows": n_rows,
        "sources": sources,
        "outputs": {p.name: sha256(p) for p in (output, refused_file, tool_copy)},
        "conventional_source": str(tool_file),
        "conventional_sha256": sha256(tool_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-root", type=Path, required=True)
    parser.add_argument("--conventional-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-directories", type=Path, required=True)
    args = parser.parse_args()
    source_directories = json.loads(args.source_directories.read_text())
    for species in SPECIES:
        paths = list((args.queries_root / species).glob("*.queries.tsv"))
        expected = source_directories[species]
        if (
            len(paths) != expected["files"]
            or sum(p.stat().st_size for p in paths) != expected["bytes"]
        ):
            raise ValueError(f"Downloaded query files are incomplete for {species}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "description": "Repaired query minima from the completed ladder-0.4-midi run; no searches rerun",
        "source_directories": source_directories,
        "species": {},
    }
    for species in SPECIES:
        record = export_species(
            species, args.queries_root, args.conventional_root, args.output_root
        )
        manifest["species"][species] = record
        print(
            species, {k: v for k, v in record.items() if k.startswith("n_")}, flush=True
        )
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
