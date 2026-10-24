"""Native DE-kupl counts plus genotype-blocked external abundance models."""

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from full_query_provenance import sha

root, code, prefix = map(Path, sys.argv[1:])
out = root / "abundance-20261024"
out.mkdir(exist_ok=False)
setup = json.loads((root / "dekupl-setup/validated.json").read_text())
if not setup.get("fixture") or setup.get("raw_kmers_checked", 0) < 1:
    raise ValueError("DE-kupl software check missing")
if not (root / "dekupl-setup/blocked-model-check/validated.txt").exists():
    raise ValueError("Blocked model software test has not passed")
design = json.loads((code / "abundance-design-20261024.json").read_text())
if sha(code / "samples.tsv") != design["samples_sha256"]:
    raise ValueError("Frozen sample design changed")
trimmed = json.loads((root / "trimmed-inputs.validated.json").read_text())
with (code / "samples.tsv").open() as f:
    metadata = {r["run"]: r for r in csv.DictReader(f, delimiter="\t")}
if len(trimmed) != 64 or {r["run"] for r in trimmed} != set(metadata):
    raise ValueError("Require the same 64 validated trimmed inputs as SPLASH")
report = dict(
    job_id=os.environ["SLURM_JOB_ID"],
    status="validating inputs",
    commands=[],
    input_files=[],
    design_sha256=sha(code / "abundance-design-20261024.json"),
    assays="DE-kupl canonical k31 counts (nomask), Kallisto gene counts, external genotype-blocked DESeq2",
    inference="No native condition-only treatment test",
)


def save():
    p = out / "progress.json.partial"
    p.write_text(json.dumps(report, indent=2) + "\n")
    p.replace(out / "progress.json")


def call(args, label, cwd=None):
    start = time.monotonic()
    with (out / (label + ".log")).open("w") as log:
        subprocess.run(
            ["/usr/bin/time", "-v", *map(str, args)],
            cwd=cwd or out,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report["commands"].append(
        dict(argv=list(map(str, args)), seconds=time.monotonic() - start)
    )
    save()


reads = out / "reads"
reads.mkdir()
for record in trimmed:
    for mate, source in enumerate(record["files"], 1):
        p = Path(source["path"])
        if sha(p) != source["sha256"]:
            raise ValueError("SPLASH/DE trimmed input checksum differs")
        (reads / f"{record['run']}_{mate}.fastq.gz").symlink_to(p)
        report["input_files"].append(source)
# Match only explicit transcript IDs, or a gene FASTA ID with exactly one annotated transcript.
gtf = root / "references/nc_nsi_nf.gtf"
fasta = root / "references/botznik-transcripts.fa"
tx = {}
gene_tx = {}
with gtf.open() as f:
    for line in f:
        if line.startswith("#"):
            continue
        cols = line.rstrip().split("\t")
        if len(cols) != 9:
            raise ValueError("Invalid reference GTF")
        attrs = dict(re.findall(r'(\S+) "([^"]*)"', cols[8]))
        g = attrs.get("gene_id")
        t = attrs.get("transcript_id")
        if g and t:
            if t in tx and tx[t] != g:
                raise ValueError("Ambiguous transcript-to-gene mapping")
            tx[t] = g
            gene_tx.setdefault(g, set()).add(t)
mapfile = out / "transcript-to-gene.tsv"
mapped_genes = set()
seen = set()
mapping_types = {}
with fasta.open() as f, mapfile.open("w") as dest:
    w = csv.writer(dest, delimiter="\t", lineterminator="\n")
    for line in f:
        if not line.startswith(">"):
            continue
        name = line[1:].split()[0].split("|")[0]
        if name in seen:
            raise ValueError("Duplicate quantification reference ID")
        seen.add(name)
        if name in tx:
            g = tx[name]
            kind = "explicit transcript ID"
        elif name in gene_tx and len(gene_tx[name]) == 1:
            g = name
            kind = "gene FASTA ID with one annotated transcript"
        else:
            raise ValueError("Unresolved quantification reference ID: " + name)
        if g in mapped_genes:
            raise ValueError(
                "Multiple reference transcripts per gene require length-aware aggregation; not implemented"
            )
        mapped_genes.add(g)
        mapping_types[kind] = mapping_types.get(kind, 0) + 1
        w.writerow([name, g])
report["reference"] = dict(
    fasta_sha256=sha(fasta),
    gtf_sha256=sha(gtf),
    mapping_sha256=sha(mapfile),
    mapping_types=mapping_types,
    namespace="Supplied predicted Botryllus reference genes; not automatically equivalent to new-proteome or Swiss-Prot gene names",
)
config = dict(
    fastq_dir=str(reads),
    kmer_length=31,
    lib_type="unstranded",
    output_dir=str(out / "counts"),
    masking="nomask",
    ref_kallisto=str(fasta),
    transcript_to_gene=str(mapfile),
    tmp_dir=os.environ["L_SCRATCH_JOB"],
    diff_method="DESeq2",
    dekupl_counter=dict(min_recurrence=2, min_recurrence_abundance=5),
    diff_analysis=dict(
        condition=dict(A="Control", B="Treatment"),
        pvalue_threshold=0.05,
        log2fc_threshold=0,
    ),
    samples=[
        dict(name=n, condition=metadata[n]["treatment"]) for n in sorted(metadata)
    ],
)
conf = out / "config.json"
conf.write_text(json.dumps(config, indent=2) + "\n")
report["status"] = "native k-mer and gene counting"
save()
raw = out / "counts/kmer_counts/raw-counts.tsv.gz"
genes = out / "counts/gene_expression/kallisto/gene_counts.tsv.gz"
call(
    [
        prefix / "bin/snakemake",
        "-s",
        prefix / "share/dekupl/Snakefile",
        raw,
        genes,
        "--configfile",
        conf,
        "-j",
        os.environ["SLURM_CPUS_PER_TASK"],
        "--resources",
        "ram=90000",
    ],
    "native-counts",
)
report["count_outputs"] = {str(p): sha(p) for p in (raw, genes)}
report["status"] = "pooling genotype units"
save()
call(
    [
        sys.executable,
        code / "pool_abundance.py",
        raw,
        genes,
        code / "abundance-design-20261024.json",
        out / "models",
    ],
    "pool-counts",
)
for contrast in design["contrasts"]:
    for assay in ("genes", "kmers"):
        folder = out / "models" / contrast["name"] / assay
        nf = folder / "normalization.tsv"
        with nf.open("w") as f:
            subprocess.run(
                [
                    str(prefix / "share/dekupl/bin/computeNF"),
                    str(folder / "pooled.tsv.gz"),
                ],
                stdout=f,
                check=True,
            )
        call(
            [
                prefix / "bin/Rscript",
                code / "blocked_abundance.R",
                folder,
                contrast["name"],
                assay,
            ],
            "model-" + contrast["name"] + "-" + assay,
        )
report["status"] = (
    "completed abundance tests; SPLASH gene/junction crosswalk still required"
)
save()
(out / "abundance-tests.complete.json").write_text(json.dumps(report, indent=2) + "\n")
