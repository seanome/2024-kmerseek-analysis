"""Prepare a separate 64-library run without changing recovered inputs or old results."""

import csv
from frozen_samples import restore_frozen_samples
import gzip
import json
import os
from pathlib import Path
import sys

code, preflight, reference_check, launch = map(Path, sys.argv[1:])
root = Path(os.environ["SCRATCH"]) / "botryllus-shazam-PEC-20261023"
checks = json.loads((preflight / "complete.json").read_text())
reference_evidence = json.loads((reference_check / "verified.json").read_text())
if reference_evidence["status"] != "direct sequence searches passed":
    raise ValueError("Reference process checks have not passed")
if not (preflight / "transcript-to-gene.tsv").is_file():
    raise ValueError("Real reference gene-identifier mapping has not passed")
records = json.loads((root / "trimmed-inputs.validated.json").read_text())
if len(records) != 64 or len({r["run"] for r in records}) != 64:
    raise ValueError("Exactly 64 distinct validated libraries required")
launch.mkdir(exist_ok=False)
reports = code / "reports/botryllus-shazam"
inputs = launch / "inputs"
inputs.mkdir()
restore_frozen_samples(
    reports / "samples.tsv",
    reports / "abundance-design-20261024.json",
    inputs / "samples.tsv",
)
(inputs / "reference-check.json").write_text(
    json.dumps(reference_evidence, indent=2) + "\n"
)
# Use all uniquely resolved published hub memberships and all predefined immune genes.
with (root / "figure4-inputs/immune-gene-sets.tsv").open() as f:
    memberships = list(csv.DictReader(f, delimiter="\t"))
for row in memberships:
    row.update(paper_time="immune model", paper_direction="predefined signed score")
hubs = json.loads((root / "figure4-inputs/hub-gene-memberships.json").read_text())
for h in hubs:
    if h["status"] == "unique exact label within published comparison":
        memberships.append(
            dict(
                gene_id=h["gene_ids"][0],
                category=h["hub"],
                paper_time=h["time"],
                paper_direction=h["direction"],
                paper_label=h["paper_label"],
                paper_group=h["paper_group"],
                cohort=h["cohort"],
            )
        )
fields = sorted(set().union(*(r.keys() for r in memberships)))
with (inputs / "paper.tsv").open("w") as f:
    w = csv.DictWriter(f, fields, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(memberships)
external = root / "external-tunicate-pilot-20261023"
proteome = root / "checkpoints/pilot/results/botryllus_candidate_annotated.fasta"
swiss = root / "references/uniprot_sprot.fasta.gz"
with (inputs / "targets.tsv").open("w") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["database", "kind", "mode", "k", "scaled", "paths"])
    for db, path in [("swissprot", swiss), ("botryllus", proteome)]:
        for mode, k in [("protein", 7), ("dayhoff", 12), ("hp", 30)]:
            w.writerow([db, "protein", mode, k, 5, path])
    with (external / "external-runs.tsv").open() as src:
        external_rows = list(csv.DictReader(src, delimiter="\t"))
    for row in external_rows:
        run = row["run_accession"]
        paths = [
            external / run / "trimmed" / f"{run}_{i}_val_{i}.fq.gz" for i in [1, 2]
        ]
        if not all(p.is_file() for p in paths):
            raise ValueError("Unprepared external run: " + run)
        for mode, k in [("dna", 31), ("protein", 7), ("dayhoff", 12), ("hp", 30)]:
            for scale in [10, 100]:
                w.writerow([run, "reads", mode, k, scale, ";".join(map(str, paths))])
# FASTA descriptions and run metadata are reference annotations, not query functions.
with (inputs / "annotations.tsv").open("w") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["record_id", "description", "source", "metadata"])
    for path, kind in [
        (swiss, "Swiss-Prot FASTA description"),
        (proteome, "provisional Botryllus FASTA description"),
    ]:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as src:
            for line in src:
                if line.startswith(">"):
                    name, _, desc = line[1:].rstrip().partition(" ")
                    w.writerow([name, desc, kind, ""])
    for row in external_rows:
        w.writerow(
            [
                row["run_accession"],
                row["sample_title"],
                "external RNA-seq metadata",
                json.dumps(row),
            ]
        )
params = dict(
    inputs=str(root / "trimmed-inputs.validated.json"),
    design=str(reports / "abundance-design-20261024.json"),
    samples=str(inputs / "samples.tsv"),
    reference=str(root / "references/botznik-transcripts.fa"),
    gtf=str(root / "references/nc_nsi_nf.gtf"),
    targets=str(inputs / "targets.tsv"),
    paper=str(inputs / "paper.tsv"),
    annotations=str(inputs / "annotations.tsv"),
    genome=str((preflight / "botznik-chr.fa").resolve()),
    pfam_hmm=str(Path(os.environ["SCRATCH"]) / "data/pfam/Pfam-A.hmm"),
    hmmscan="/share/software/user/open/hmmer/3.4/bin/hmmscan",
    minimap2="/share/software/user/open/minimap2/2.30/bin/minimap2",
    outdir=str(launch / "results"),
    expected_libraries=64,
)
(launch / "params.json").write_text(json.dumps(params, indent=2) + "\n")
(launch / "scope.json").write_text(
    json.dumps(
        dict(
            libraries=64,
            pairs=sum(r["pairs"] for r in records),
            external_runs=[r["run_accession"] for r in external_rows],
            external_scope="Only the explicitly prepared runs, not the entire tunicate archive",
            reference_checks=str(inputs / "reference-check.json"),
            annotation_coordinate_status=reference_evidence[
                "annotation_coordinate_status"
            ],
            paper_memberships=len(memberships),
            ambiguous_or_unresolved_hubs=[
                h
                for h in hubs
                if h["status"] != "unique exact label within published comparison"
            ],
        ),
        indent=2,
    )
    + "\n"
)
