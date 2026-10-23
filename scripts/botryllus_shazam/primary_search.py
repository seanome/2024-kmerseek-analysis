"""Orchestrate actual Sourmash/Branchwater Rust sketching and bulk retrieval."""

import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from Bio import SeqIO


def run(root, out, query):
    if (
        importlib.metadata.version("sourmash") != "4.9.4"
        or importlib.metadata.version("sourmash_plugin_branchwater") != "0.9.15"
    ):
        raise ValueError("Pinned Sourmash/Branchwater versions required")
    versions = {"sourmash": "4.9.4", "branchwater": "0.9.15"}
    commands = []
    cores = str(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

    def call(args, label):
        start = time.monotonic()
        with (out / (label + ".log")).open("w") as f:
            subprocess.run(
                [
                    "/usr/bin/time",
                    "-v",
                    sys.executable,
                    "-m",
                    "sourmash",
                    *map(str, args),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        commands.append(
            dict(argv=list(map(str, args)), seconds=time.monotonic() - start)
        )

    sources = {}
    references = {}
    for db, path in [
        ("botryllus", root / "references/botryllus_new_genome_protein.fasta"),
        ("swissprot", root / "references/uniprot_sprot.fasta.gz"),
    ]:
        sources[db] = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        target = out / (db + ".search.faa")
        mapping = out / (db + ".record-map.tsv")
        mask = out / (db + ".masked-positions.tsv")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f, target.open("w") as dest, mapping.open(
            "w"
        ) as mp, mask.open("w") as masked:
            w = csv.writer(mp, delimiter="\t")
            w.writerow(
                [
                    "record_id",
                    "original_id",
                    "original_description",
                    "sequence_sha256",
                    "length_aa",
                ]
            )
            mw = csv.writer(masked, delimiter="\t")
            mw.writerow(
                ["record_id", "position_1based", "original_residue", "search_residue"]
            )
            count = 0
            for count, r in enumerate(SeqIO.parse(f, "fasta"), 1):
                rid = f"BSP{count:06d}" if db == "botryllus" else f"SWP{count:07d}"
                seq = str(r.seq)
                search = []
                for pos, aa in enumerate(seq.upper(), 1):
                    if aa not in "ACDEFGHIKLMNPQRSTVWYX":
                        mw.writerow([rid, pos, aa, "X"])
                        aa = "X"
                    search.append(aa)
                dest.write(f">{rid}\n" + "".join(search) + "\n")
                w.writerow(
                    [
                        rid,
                        r.id,
                        r.description,
                        hashlib.sha256(seq.encode()).hexdigest(),
                        len(seq),
                    ]
                )
        if db == "botryllus" and count != 45339:
            raise ValueError("Unexpected Botryllus reference population")
        references[db] = dict(path=str(target), records=count)
    sources["queries"] = hashlib.file_digest(query.open("rb"), "sha256").hexdigest()
    nq = sum(1 for _ in SeqIO.parse(query, "fasta"))
    # A sequence FASTA remains separate from its signatures and all metadata tables.
    for db, path in [
        ("queries", query),
        *[(db, Path(x["path"])) for db, x in references.items()],
    ]:
        manifest = out / (db + ".fromfile.csv")
        with manifest.open("w") as f:
            w = csv.writer(f)
            w.writerow(["name", "genome_filename", "protein_filename"])
            w.writerow([db, "", str(path)])
        for mode, k in [("protein", 7), ("dayhoff", 12), ("hp", 30)]:
            target = out / f"{db}.{mode}.scaled5.sig.zip"
            call(
                [
                    "scripts",
                    "manysketch",
                    manifest,
                    "--singleton",
                    "-c",
                    cores,
                    "-p",
                    f"{mode},k={k},scaled=5",
                    "-o",
                    target,
                ],
                f"{db}.{mode}.sketch",
            )
    results = []
    for db in references:
        for mode, k in [("protein", 7), ("dayhoff", 12), ("hp", 30)]:
            for scaled in ([5] if db == "swissprot" else [5, 10, 100]):
                output = out / f"{db}.{mode}.scaled{scaled}.csv"
                call(
                    [
                        "scripts",
                        "manysearch",
                        out / f"queries.{mode}.scaled5.sig.zip",
                        out / f"{db}.{mode}.scaled5.sig.zip",
                        "-c",
                        cores,
                        "-m",
                        mode,
                        "-k",
                        str(k),
                        "-s",
                        str(scaled),
                        "-t",
                        "0.1",
                        "-N",
                        "--ignore-abundance",
                        "-o",
                        output,
                    ],
                    f"{db}.{mode}.scaled{scaled}.search",
                )
                with output.open() as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames
                    rows = sum(1 for _ in reader)
                results.append(
                    dict(
                        database=db,
                        mode=mode,
                        ksize=k,
                        scaled=scaled,
                        csv=str(output),
                        csv_columns=columns,
                        raw_match_rows=rows,
                        sha256=hashlib.file_digest(
                            output.open("rb"), "sha256"
                        ).hexdigest(),
                    )
                )
    # This is execution completion, not validation of retrieval semantics or function.
    report = dict(
        versions=versions,
        input_scope=(
            "historical pilot peptide queries"
            if "checkpoints/pilot" in str(query)
            else "full-depth peptide queries"
        ),
        query_records=nq,
        references=references,
        source_sha256=sources,
        commands=commands,
        results=results,
        validation_status="pending independent comparison against sourmash search and sampled-hash arithmetic",
        filter="native containment >=0.1; raw rows retained, >=2 shared-hash filter applied only after validation",
    )
    (out / "execution.complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Primary Sourmash execution complete; independent validation required",
        flush=True,
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
