"""Full-depth upstream discovery, uncapped significant-anchor extension and translation."""

import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from select_anchors import select


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def command(argv, out, label):
    start = time.monotonic()
    with (out / (label + ".log")).open("w") as log:
        result = subprocess.run(
            ["/usr/bin/time", "-v", *map(str, argv)],
            cwd=out,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    (out / (label + ".command.json")).write_text(
        json.dumps(
            dict(
                argv=list(map(str, argv)),
                exit_code=result.returncode,
                seconds=time.monotonic() - start,
            ),
            indent=2,
        )
        + "\n"
    )
    result.check_returncode()


def run(root, software, out):
    records = json.loads((root / "trimmed-inputs.validated.json").read_text())
    manifest = list(
        csv.DictReader((Path(__file__).parent / "ena-runs.tsv").open(), delimiter="\t")
    )
    if {r["run"] for r in records} != {r["run_accession"] for r in manifest} or len(
        records
    ) != 64:
        raise ValueError(
            "Full-depth discovery requires every one of the 64 validated runs"
        )
    out.mkdir(exist_ok=False)
    (out / "resource-model.json").write_text(
        json.dumps(
            dict(
                job_id=os.environ.get("SLURM_JOB_ID"),
                cpus=16,
                memory_GB=96,
                time_hours=48,
                evidence="Six-library oyster pilot: 6 million input pairs, 3m24s, 2.2 GB peak RSS for SPLASH plus top-300 Compactors. Botryllus raw maximum 654672879 pairs, about 109 times the reads. KMC bounded at 24 GB; 64 bins. Full-depth all-anchor Compactors has no measured history; 48h and 96 GB are conservative model requests, not guarantees.",
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "software-sha256.json").write_text(
        json.dumps(
            {
                name: sha(software / name)
                for name in ["splash", "compactors", "satc", "satc_merge", "sig_anch"]
            },
            indent=2,
        )
        + "\n"
    )
    input_lines = []
    read_lines = []
    for record in sorted(records, key=lambda x: x["run"]):
        run_id = record["run"]
        merged = out / (run_id + ".fastq.gz")
        for source in record["files"]:
            if sha(Path(source["path"])) != source["sha256"]:
                raise ValueError("Validated trimmed source changed: " + source["path"])
        partial = merged.with_suffix(".gz.partial")
        with gzip.open(partial, "wb", compresslevel=1) as dest:
            for source in record["files"]:
                with gzip.open(source["path"], "rb") as f:
                    shutil.copyfileobj(f, dest, 1024 * 1024)
        partial.rename(merged)
        input_lines.append(f"{run_id} {merged}\n")
        read_lines.append(str(merged) + "\n")
    (out / "input.txt").write_text("".join(input_lines))
    (out / "reads.txt").write_text("".join(read_lines))
    command(
        [
            software / "splash",
            "--bin_path",
            software,
            "--outname_prefix",
            "full",
            "--anchor_len",
            "31",
            "--target_len",
            "31",
            "--gap_len",
            "0",
            "--fdr_threshold",
            "0.05",
            "--without_compactors",
            "--keep_significant_anchors_satc",
            "--dump_sample_anchor_target_count_txt",
            "--satc_merge_dump_format",
            "satc",
            "--keep_top_n_target_entropy",
            "0",
            "--keep_top_n_effect_size_bin",
            "0",
            "--n_threads_stage_1",
            "1",
            "--n_threads_stage_1_internal",
            "16",
            "--n_threads_stage_2",
            "16",
            "--n_bins",
            "64",
            "--kmc_max_mem_GB",
            "24",
            "--tmp_dir",
            Path(os.environ["L_SCRATCH_JOB"]) / "botryllus-splash",
            out / "input.txt",
        ],
        out,
        "splash",
    )
    selection = select(
        out / "full.after_correction.all_anchors.tsv", out / "all-significant"
    )
    selected = out / "all-significant/selected_anchors.tsv"
    (out / "splash.validated.json").write_text(json.dumps(selection, indent=2) + "\n")
    if selection["selected_anchors"]:
        command(
            [
                software / "compactors",
                "--num_threads",
                "16",
                "--reads_buffer_gb",
                "24",
                out / "reads.txt",
                selected,
                out / "compactors.tsv",
            ],
            out,
            "compactors",
        )
    else:
        (out / "compactors.tsv").write_text("anchor\tcompactor\tid\tparent_id\n")
    sys.path.insert(0, str(root / "checkpoints/pilot/code/reused"))
    from core import translate_segments

    anchors = {r["anchor"] for r in csv.DictReader(selected.open(), delimiter="\t")}
    peptides = {}
    n = 0
    extended = set()
    emitted = set()
    with (out / "extensions.fasta").open("w") as dna, (
        out / "peptide_membership.tsv"
    ).open("w") as mf:
        w = csv.DictWriter(
            mf,
            fieldnames=[
                "peptide_id",
                "extension_id",
                "anchor",
                "strand",
                "frame",
                "nt_start",
                "nt_end",
                "length_aa",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        w.writeheader()
        for n, row in enumerate(
            csv.DictReader((out / "compactors.tsv").open(), delimiter="\t"), 1
        ):
            seq = row["compactor"]
            anchor = row["anchor"]
            eid = f"EXT{n:09d}"
            if anchor not in anchors or anchor not in seq or set(seq) - set("ACGTN"):
                raise ValueError("Invalid Compactors output record")
            emitted.add(anchor)
            if len(seq) > len(anchor):
                extended.add(anchor)
            dna.write(f">{eid} anchor={anchor}\n{seq}\n")
            for fragment in translate_segments(seq):
                peptide = fragment.pop("peptide")
                pid = peptides.setdefault(peptide, f"PEP{len(peptides)+1:09d}")
                w.writerow(
                    dict(
                        peptide_id=pid,
                        extension_id=eid,
                        anchor=anchor,
                        length_aa=len(peptide),
                        **fragment,
                    )
                )
    with (out / "peptides.fasta").open("w") as f:
        for peptide, pid in peptides.items():
            f.write(f">{pid}\n{peptide}\n")
    with (out / "anchors_without_extension.tsv").open("w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["anchor", "reason"])
        for anchor in sorted(anchors - extended):
            w.writerow(
                [
                    anchor,
                    (
                        "no_compactor_record_emitted"
                        if anchor not in emitted
                        else "only_unextended_record_emitted"
                    ),
                ]
            )
    completion = dict(
        rule="pval_opt_corrected < 0.05; no top-N cap",
        selected_anchors=len(anchors),
        anchors_with_output=len(emitted),
        anchors_extended=len(extended),
        anchors_without_extension=len(anchors - extended),
        extension_records=n,
        unique_peptides=len(peptides),
        inference="Unsupervised composition; no treatment abundance claim",
        outputs={
            p.name: sha(p)
            for p in [
                out / "compactors.tsv",
                out / "extensions.fasta",
                out / "peptides.fasta",
                out / "peptide_membership.tsv",
                out / "anchors_without_extension.tsv",
            ]
        },
    )
    partial = out / "discovery.validated.json.partial"
    partial.write_text(json.dumps(completion, indent=2) + "\n")
    partial.rename(out / "discovery.validated.json")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
