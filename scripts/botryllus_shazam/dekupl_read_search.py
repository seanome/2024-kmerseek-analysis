"""Recover exact DE-kupl seed-containing read pairs, extend, and search with Branchwater."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

COMP = str.maketrans("ACGT", "TGCA")


def canonical(seq):
    if len(seq) != 31 or set(seq) - set("ACGT"):
        raise ValueError("Require an unambiguous DNA 31-mer")
    return min(seq, seq.translate(COMP)[::-1])


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def select(models):
    seeds = {}
    sources = []
    for path in sorted(models.glob("*/kmers/all-features.tsv.gz")):
        sources.append(dict(path=str(path), sha256=sha(path)))
        with gzip.open(path, "rt") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    q = float(row["padj"])
                except ValueError:
                    continue
                if not math.isfinite(q) or q >= 0.05:
                    continue
                seq = canonical(row["tag"])
                seeds.setdefault(seq, []).append(
                    dict(
                        contrast=path.parents[1].name,
                        padj=q,
                        log2foldchange=row["log2FoldChange"],
                        tested_kmer=row["tag"],
                    )
                )
                if len(seeds) > 10000000:
                    raise ValueError(
                        "More than 10 million selected seeds: partition recruitment before launch"
                    )
    if len(sources) != 3:
        raise ValueError(
            "Expected the three completed frozen genotype-blocked contrasts"
        )
    return seeds, sources


def run(root, code, out):
    import sourmash
    from sourmash.save_load import SaveSignaturesToLocation
    from Bio.Seq import Seq

    versions = {
        p: importlib.metadata.version(p)
        for p in ["sourmash", "sourmash_plugin_branchwater"]
    }
    if versions != {"sourmash": "4.9.4", "sourmash_plugin_branchwater": "0.9.15"}:
        raise ValueError("Require tested pinned Sourmash / Branchwater versions")
    abundance = root / "abundance-20261024"
    completed = abundance / "abundance-tests.complete.json"
    json.loads(completed.read_text())
    inputs = json.loads((root / "trimmed-inputs.validated.json").read_text())
    if len(inputs) != 64 or len({r["run"] for r in inputs}) != 64:
        raise ValueError("Require all 64 validated trimmed input libraries")
    seeds, sources = select(abundance / "models")
    out.mkdir(exist_ok=False)
    report = dict(
        scope="Full-depth genotype/collection-blocked DE-kupl adjusted P < 0.05, no top-N cap",
        status="preparing",
        job_id=os.environ["SLURM_JOB_ID"],
        versions=versions,
        model_sources=sources,
        abundance_completion_sha256=sha(completed),
        distinct_seeds=len(seeds),
        commands=[],
        recruitment=[],
        results=[],
        query_coverage=[],
        external_scope="Four prepared external tunicate runs; the full 5776-run collection remains unavailable for search",
    )

    def save():
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")

    def call(argv, label):
        start = time.monotonic()
        with (out / (label + ".log")).open("w") as f:
            subprocess.run(
                ["/usr/bin/time", "-v", *map(str, argv)],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report["commands"].append(
            dict(
                argv=list(map(str, argv)),
                seconds=time.monotonic() - start,
                log=label + ".log",
            )
        )
        save()

    seed_ids = {s: "DK" + hashlib.sha256(s.encode()).hexdigest()[:24] for s in seeds}
    if len(set(seed_ids.values())) != len(seeds):
        raise ValueError("Query ID collision")
    (out / "seeds.tsv").write_text(
        "".join(s + "\t" + seed_ids[s] + "\n" for s in sorted(seeds))
    )
    (out / "seed-metadata.json").write_text(
        json.dumps(
            {seed_ids[s]: dict(sequence=s, tests=seeds[s]) for s in seeds}, indent=2
        )
        + "\n"
    )
    if not seeds:
        report.update(
            status="completed",
            interpretation="No DE-kupl seeds passed adjusted P < 0.05; no external searches performed",
        )
        save()
        (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    binary = Path(os.environ["L_SCRATCH_JOB"]) / "recruit-dekupl-reads"
    call(
        ["rustc", "-O", code / "recruit_dekupl_reads.rs", "-o", binary],
        "compile-recruiter",
    )
    read_folder = out / "recruited-reads"
    read_folder.mkdir()

    def recruit(record):
        name = record["run"]
        paths = [Path(x["path"]) for x in record["files"]]
        for p, meta in zip(paths, record["files"]):
            if sha(p) != meta["sha256"]:
                raise ValueError("Trimmed input checksum differs")
        dest = [read_folder / f"{name}_{m}.fastq.gz" for m in (1, 2)]
        mapping = read_folder / f"{name}.hits.tsv.gz"
        args = [
            str(binary),
            str(out / "seeds.tsv"),
            *map(str, paths),
            *map(str, dest),
            str(mapping),
        ]
        with (read_folder / f"{name}.time.txt").open("w") as log:
            raw = subprocess.check_output(
                ["/usr/bin/time", "-v", *args], stderr=log, text=True
            )
        counts = json.loads(raw)
        if counts["input_pairs"] != record["pairs"]:
            raise ValueError("Recruitment did not scan the complete library")
        return dict(
            run=name,
            **counts,
            reads=[str(x) for x in dest],
            mapping=str(mapping),
            mapping_sha256=sha(mapping),
        )

    report["status"] = "exact read-pair recruitment"
    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        report["recruitment"] = list(pool.map(recruit, inputs))
    all_reads = [p for r in report["recruitment"] for p in r["reads"]]
    (out / "reads.txt").write_text("\n".join(all_reads) + "\n")
    # Both orientations are seeds; exact supporting reads retain their sample and mate IDs.
    anchors = out / "anchors.tsv"
    anchors.write_text(
        "anchor\n"
        + "\n".join(sorted({x for s in seeds for x in [s, s.translate(COMP)[::-1]]}))
        + "\n"
    )
    compactors = root.parent / "oyster-shazam-20261023/software/compactors"
    report["status"] = "Compactors extensions"
    save()
    call(
        [
            compactors,
            "--all_anchors",
            "--kmer_len",
            "31",
            "--num_threads",
            os.environ["SLURM_CPUS_PER_TASK"],
            "--reads_buffer_gb",
            "24",
            out / "reads.txt",
            anchors,
            out / "compactors.tsv",
        ],
        "compactors",
    )
    dna = {seed_ids[s]: s for s in seeds}
    origins = {seed_ids[s]: seed_ids[s] for s in seeds}
    extensions = []
    with (out / "compactors.tsv").open() as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t"), 1):
            seed = canonical(row["anchor"])
            seq = row["compactor"].upper()
            if seed not in seeds or row["anchor"] not in seq or set(seq) - set("ACGTN"):
                raise ValueError("Unsupported Compactors output")
            qid = f"DEX{i:09d}"
            dna[qid] = seq
            origins[qid] = seed_ids[seed]
            extensions.append(dict(query=qid, seed=seed_ids[seed], **row))
    (out / "extension-membership.json").write_text(
        json.dumps(extensions, indent=2) + "\n"
    )
    report["extension_limits"] = (
        "Compactors defaults retain branching up to 1000 compactors per seed and 20 children per step; unsupported/short/unextended seeds remain direct DNA queries. Pooled extension is not a sample-specific transcript or proof of a full coding sequence."
    )
    dna_file = out / "queries.fna"
    dna_file.write_text("".join(f">{k}\n{v}\n" for k, v in dna.items()))
    fragments = {}
    fragment_parent = {}
    for qid, seq in dna.items():
        for strand, oriented in [("+", seq), ("-", seq.translate(COMP)[::-1])]:
            for frame in range(3):
                peptide = str(
                    Seq(
                        oriented[frame : len(oriented) - (len(oriented) - frame) % 3]
                    ).translate()
                )
                for part in peptide.replace("X", "*").split("*"):
                    if len(part) < 7:
                        continue
                    fid = f"FRAG{len(fragments)+1:012d}"
                    fragments[fid] = part
                    fragment_parent[fid] = qid
    protein_file = out / "fragments.faa"
    protein_file.write_text("".join(f">{k}\n{v}\n" for k, v in fragments.items()))
    (out / "fragment-membership.json").write_text(json.dumps(fragment_parent) + "\n")
    base = [sys.executable, "-m", "sourmash", "scripts"]
    cores = os.environ["SLURM_CPUS_PER_TASK"]
    archive = json.loads(
        (root / "external-tunicate-pilot-20261023/execution.complete.json").read_text()
    )
    if archive["status"] != "completed":
        raise ValueError("External target preparation incomplete")
    for mode, k in [("dna", 31), ("protein", 7), ("dayhoff", 12), ("hp", 30)]:
        source = dna_file if mode == "dna" else protein_file
        csvfile = out / f"{mode}.input.csv"
        with csvfile.open("w") as f:
            w = csv.writer(f)
            w.writerow(["name", "genome_filename", "protein_filename"])
            w.writerow(
                [
                    "dekupl",
                    str(source) if mode == "dna" else "",
                    str(source) if mode != "dna" else "",
                ]
            )
        rawzip = out / f"{mode}.fragments.sig.zip"
        grouped = {
            qid: sourmash.MinHash(
                n=0,
                ksize=k,
                scaled=5,
                is_protein=mode == "protein",
                dayhoff=mode == "dayhoff",
                hp=mode == "hp",
            )
            for qid in dna
        }
        eligible = {qid for qid, seq in dna.items() if mode == "dna" and len(seq) >= k}
        if mode != "dna":
            eligible = {
                fragment_parent[fid] for fid, seq in fragments.items() if len(seq) >= k
            }
        if eligible:
            call(
                [
                    *base,
                    "manysketch",
                    csvfile,
                    "--singleton",
                    "-p",
                    f"{mode},k={k},scaled=5",
                    "-c",
                    cores,
                    "-o",
                    rawzip,
                ],
                f"manysketch-{mode}",
            )
            seen = set()
            for sig in sourmash.load_file_as_signatures(str(rawzip)):
                name = sig.name
                qid = name if mode == "dna" else fragment_parent[name]
                if name in seen:
                    raise ValueError("Duplicate sketch ID")
                if len(seen) < 10:
                    expected_mh = sig.minhash.copy_and_clear()
                    if mode == "dna":
                        expected_mh.add_sequence(dna[name], force=True)
                    else:
                        expected_mh.add_protein(fragments[name])
                    if set(expected_mh.hashes) != set(sig.minhash.hashes):
                        raise ValueError(
                            "Branchwater hash parity differs from independent MinHash validation"
                        )
                seen.add(name)
                grouped[qid].add_many(list(sig.minhash.hashes))
            expected = (
                set(dna)
                if mode == "dna"
                else {fid for fid, seq in fragments.items() if len(seq) >= k}
            )
            if expected - seen:
                raise ValueError("Eligible source records dropped by manysketch")
        queryzip = out / f"queries.{mode}.sig.zip"
        with SaveSignaturesToLocation(str(queryzip)) as dest:
            for qid, mh in grouped.items():
                dest.add(sourmash.SourmashSignature(mh, name=qid))
        loaded = set()
        for signature in sourmash.load_file_as_signatures(str(queryzip)):
            if signature.name in loaded or signature.name not in grouped:
                raise ValueError("Grouped query names changed")
            loaded.add(signature.name)
            if set(signature.minhash.hashes) != set(grouped[signature.name].hashes):
                raise ValueError("Grouped query hashes changed on serialization")
        if loaded != set(dna):
            raise ValueError("Grouped serialization dropped empty DNA queries")
        for scale in ([10, 100] if mode == "dna" else [5, 10, 100]):
            report["query_coverage"].append(
                dict(
                    mode=mode,
                    scaled=scale,
                    queries=len(dna),
                    long_enough=len(eligible),
                    nonempty=sum(
                        bool(len(m.downsample(scaled=scale))) for m in grouped.values()
                    ),
                    source="DNA seeds and supported extensions; six-frame stop-free hashes grouped under original DNA query",
                )
            )
        targets = []
        if mode != "dna":
            targets.append(
                (
                    "swissprot",
                    5,
                    root
                    / f"primary-search-pilot-20261023/swissprot.{mode}.scaled5.sig.zip",
                )
            )
        for r in archive["runs"]:
            for scale in (10, 100):
                targets.append(
                    (
                        r["run"],
                        scale,
                        root
                        / f"external-tunicate-pilot-20261023/{r['run']}/{mode}.scaled{scale}.sig.zip",
                    )
                )
        for db, scale, target in targets:
            if not any(len(m.downsample(scaled=scale)) for m in grouped.values()):
                report["results"].append(
                    dict(
                        database=db,
                        mode=mode,
                        scaled=scale,
                        status="unsearchable: no query hashes; not a negative biological result",
                    )
                )
                continue
            output = out / f"{db}.{mode}.scaled{scale}.csv"
            call(
                [
                    *base,
                    "manysearch",
                    queryzip,
                    target,
                    "-m",
                    "DNA" if mode == "dna" else mode,
                    "-k",
                    str(k),
                    "-s",
                    str(scale),
                    "-t",
                    "0",
                    "-A",
                    "-N",
                    "--ignore-abundance",
                    "-c",
                    cores,
                    "-o",
                    output,
                ],
                f"manysearch-{db}-{mode}-{scale}",
            )
            with output.open() as f:
                rows = sum(1 for _ in csv.DictReader(f))
            report["results"].append(
                dict(
                    database=db,
                    mode=mode,
                    scaled=scale,
                    status="completed candidate retrieval",
                    rows=rows,
                    csv=str(output),
                    sha256=sha(output),
                    target_sha256=sha(target),
                    query_sha256=sha(queryzip),
                    interpretation="Any nonzero overlap; candidates require sequence confirmation and chance controls",
                )
            )
            save()
    report["query_origins"] = origins
    report["status"] = "completed"
    save()
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["root", "code", "out"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.root, a.code, a.out)
