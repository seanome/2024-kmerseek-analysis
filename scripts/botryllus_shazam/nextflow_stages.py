"""File-level operations for the Botryllus DSL2 workflow."""

import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys

PROTOCOL = "six-frame-stop-only-X-min30-v1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def fasta(path):
    records = {}
    name = None
    with Path(path).open() as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in records:
                    raise ValueError("Duplicate FASTA identifier: " + name)
                records[name] = ""
            elif line.strip():
                if name is None:
                    raise ValueError("Sequence before FASTA header")
                records[name] += line.strip()
    return records


def write_fasta(path, records):
    Path(path).write_text("".join(f">{k}\n{v}\n" for k, v in records.items()))


def pair(run, r1, r2, h1, h2, count, validator):
    if [sha(r1), sha(r2)] != [h1, h2]:
        raise ValueError("Trimmed input fingerprint changed")
    result = json.loads(
        subprocess.check_output([str(Path(validator).resolve()), r1, r2], text=True)
    )
    if result["pairs"] != int(count):
        raise ValueError("Trimmed input pair count changed")
    save(run + ".validated.json", dict(run=run, **result, sha256=[h1, h2]))


def check_recruit(validation, counts):
    original = json.loads(Path(validation).read_text())
    recruited = json.loads(Path(counts).read_text())
    if original["pairs"] != recruited["input_pairs"]:
        raise ValueError("Recruitment did not scan every validated pair")


def genes(reference, gtf, *quantifications):
    tx, gene_tx = {}, {}
    for line in Path(gtf).read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9:
            raise ValueError("Invalid GTF")
        attrs = dict(re.findall(r'(\S+) "([^"]*)"', fields[8]))
        g, t = attrs.get("gene_id"), attrs.get("transcript_id")
        if g and t:
            if t in tx and tx[t] != g:
                raise ValueError("Ambiguous transcript mapping")
            tx[t] = g
            gene_tx.setdefault(g, set()).add(t)
    mapping = {}
    mapped_genes = set()
    for identifier in fasta(reference):
        name = identifier.split("|")[0]
        gene = tx.get(name)
        if gene is None and len(gene_tx.get(name, ())) == 1:
            gene = name
        if gene is None or gene in mapped_genes or name in mapping:
            raise ValueError(
                "Reference requires unresolved or length-aware gene aggregation"
            )
        mapping[name] = gene
        mapped_genes.add(gene)
    Path("transcript-to-gene.tsv").write_text(
        "".join(t + "\t" + g + "\n" for t, g in mapping.items())
    )
    matrix = {g: [] for g in mapping.values()}
    names = []
    for folder in quantifications:
        names.append(Path(folder).name.removesuffix(".quant"))
        counts = {}
        with (Path(folder) / "abundance.tsv").open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                t = row["target_id"].split("|")[0]
                count = float(row["est_counts"])
                if (
                    t not in mapping
                    or t in counts
                    or not math.isfinite(count)
                    or count < 0
                ):
                    raise ValueError("Invalid Kallisto record")
                counts[t] = count
        if set(counts) != set(mapping):
            raise ValueError("Kallisto reference population changed")
        for t, g in mapping.items():
            # Preserve the installed DE-kupl conversion (truncate per-gene estimates).
            matrix[g].append(int(counts[t]))
    if len(names) != len(set(names)):
        raise ValueError("Duplicate quantification sample")
    with gzip.open("genes.tsv.gz", "wt") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["tag", *names])
        w.writerows([g, *counts] for g, counts in matrix.items())


def seeds(*models):
    selected = {}
    contrasts = set()
    for model in models:
        with gzip.open(Path(model) / "all-features.tsv.gz", "rt") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                contrasts.add(row["contrast"])
                value = row["padj"]
                if value in ("NA", "", "nan"):
                    continue
                q = float(value)
                if not math.isfinite(q) or not 0 <= q <= 1:
                    raise ValueError("Invalid adjusted P")
                if q >= 0.05:
                    continue
                seq = row["tag"]
                if len(seq) != 31 or set(seq) - set("ACGT"):
                    raise ValueError("Invalid selected 31-mer")
                seq = min(seq, seq.translate(str.maketrans("ACGT", "TGCA"))[::-1])
                selected.setdefault(seq, []).append(
                    dict(
                        contrast=row["contrast"],
                        padj=q,
                        log2foldchange=row["log2FoldChange"],
                        tested_kmer=row["tag"],
                    )
                )
    if len(models) != 3 or len(contrasts) != 3:
        raise ValueError("Require three complete contrasts")
    out = Path("selected")
    out.mkdir()
    ids = {
        s: "DK" + hashlib.sha256(s.encode()).hexdigest()[:24] for s in sorted(selected)
    }
    (out / "seeds.tsv").write_text("".join(s + "\t" + ids[s] + "\n" for s in ids))
    save(out / "seeds.json", selected)
    save(out / "ids.json", ids)
    for contrast in sorted(contrasts):
        folder = out / "contrasts" / contrast
        folder.mkdir(parents=True)
        subset = {
            seq: [t for t in tests if t["contrast"] == contrast]
            for seq, tests in selected.items()
        }
        subset = {seq: tests for seq, tests in subset.items() if tests}
        save(folder / "seeds.json", subset)
        save(folder / "ids.json", {seq: ids[seq] for seq in subset})
    save(
        out / "sources.json",
        {
            str(Path(p) / "all-features.tsv.gz"): sha(Path(p) / "all-features.tsv.gz")
            for p in models
        },
    )


def merge(selected, binary):
    from dekupl_contigs import assemble

    p = Path(selected)
    assemble(
        json.loads((p / "seeds.json").read_text()),
        json.loads((p / "ids.json").read_text()),
        Path(binary),
        Path("native"),
    )


def splash(software, cpus, reads):
    out = Path("splash")
    out.mkdir()
    inputs = sorted(Path(reads).glob("*"))
    groups = {}
    for p in inputs:
        run = p.name.split("_")[0]
        groups.setdefault(run, []).append(p.resolve())
    with (out / "input.txt").open("w") as manifest:
        for run, paths in groups.items():
            if len(paths) != 2:
                raise ValueError("SPLASH requires two mates per library")
            dest = out / (run + ".fastq.gz")
            # Concatenated gzip members preserve both mates without recompression.
            with dest.open("wb") as f:
                import shutil

                for p in paths:
                    with p.open("rb") as src:
                        shutil.copyfileobj(src, f, 1024 * 1024)
            manifest.write(f"{run} {dest.resolve()}\n")
    cmd = [
        str(Path(software) / "splash"),
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
        cpus,
        "--n_threads_stage_2",
        cpus,
        "--n_bins",
        "64",
        "--kmc_max_mem_GB",
        "24",
        "--tmp_dir",
        str((out / "tmp").resolve()),
        str((out / "input.txt").resolve()),
    ]
    subprocess.run(cmd, cwd=out, check=True)


def compact(source, reads, selected, binary, cpus):
    out = Path("context")
    out.mkdir()
    if source == "splash":
        with (Path(selected) / "selected_anchors.tsv").open() as f:
            seqs = {r["anchor"] for r in csv.DictReader(f, delimiter="\t")}
        ids = {
            s: "SA" + hashlib.sha256(s.encode()).hexdigest()[:24] for s in sorted(seqs)
        }
        anchors = seqs
    else:
        ids = json.loads((Path(selected) / "ids.json").read_text())
        anchors = set(ids) | {
            s.translate(str.maketrans("ACGT", "TGCA"))[::-1] for s in ids
        }
    (out / "anchors.tsv").write_text(
        "anchor\n" + "".join(s + "\n" for s in sorted(anchors))
    )
    paths = sorted(Path(reads).glob("*"))
    (out / "reads.txt").write_text("".join(str(p.resolve()) + "\n" for p in paths))
    if anchors:
        subprocess.run(
            [
                binary,
                "--all_anchors",
                "--kmer_len",
                "31",
                "--num_threads",
                cpus,
                "--reads_buffer_gb",
                "24",
                str(out / "reads.txt"),
                str(out / "anchors.tsv"),
                str(out / "compactors.tsv"),
            ],
            check=True,
        )
    else:
        (out / "compactors.tsv").write_text("anchor\tcompactor\tid\tparent_id\n")
    dna = {qid: s for s, qid in ids.items()}
    origins = {qid: dict(kind="selected seed", source=source, seed=qid) for qid in dna}
    with (out / "compactors.tsv").open() as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t"), 1):
            s, seq = row["anchor"], row["compactor"]
            canonical = (
                min(s, s.translate(str.maketrans("ACGT", "TGCA"))[::-1])
                if source == "dekupl"
                else s
            )
            if canonical not in ids or s not in seq or set(seq) - set("ACGTN"):
                raise ValueError("Compactor has no selected seed support")
            qid = ("SX" if source == "splash" else "DX") + f"{i:012d}"
            dna[qid] = seq
            origins[qid] = dict(
                kind="read-supported context",
                source=source,
                seed=ids[canonical],
                compactor=row,
            )
    write_fasta(out / "dna.fasta", dna)
    save(out / "origins.json", origins)


def translate(source, context, *native_contexts):
    from dekupl_translation import translate_context

    dna = fasta(Path(context) / "dna.fasta")
    origins = json.loads((Path(context) / "origins.json").read_text())
    if source == "dekupl":
        for row in [
            row
            for folder in native_contexts
            for row in json.loads((Path(folder) / "contig-metadata.json").read_text())[
                "contigs"
            ]
        ]:
            if row["query"] in dna:
                raise ValueError("Duplicate query")
            dna[row["query"]] = row["sequence"]
            origins[row["query"]] = dict(
                kind="native mergeTags contig", source=source, **row
            )
    peptides, membership, coordinates = translate_context(dna)
    out = Path("queries")
    out.mkdir()
    write_fasta(out / "dna.fasta", dna)
    write_fasta(out / "peptides.faa", peptides)
    save(out / "origins.json", origins)
    save(out / "membership.json", membership)
    save(out / "coordinates.json", coordinates)
    save(
        out / "protocol.json",
        dict(protocol=PROTOCOL, source=source, dna=len(dna), peptides=len(peptides)),
    )


def check_versions():
    for name, version in [
        ("sourmash", "4.9.4"),
        ("sourmash_plugin_branchwater", "0.9.15"),
    ]:
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned version required: " + name)


def sketch(queries, mode, k, cpus):
    import sourmash
    from sourmash.save_load import SaveSignaturesToLocation

    check_versions()
    k = int(k)
    folder = Path(queries)
    dna = fasta(folder / "dna.fasta")
    peptides = fasta(folder / "peptides.faa")
    records = dna if mode == "dna" else peptides
    membership = json.loads((folder / "membership.json").read_text())
    kwargs = dict(
        n=0,
        ksize=k,
        scaled=5,
        is_protein=mode == "protein",
        dayhoff=mode == "dayhoff",
        hp=mode == "hp",
    )
    grouped = {name: sourmash.MinHash(**kwargs) for name in dna}
    eligible = {n: s for n, s in records.items() if len(s) >= k}
    if eligible:
        write_fasta("eligible.fasta", eligible)
        with open("sketch.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(["name", "genome_filename", "protein_filename"])
            w.writerow(
                [
                    "query",
                    str(Path("eligible.fasta").resolve()) if mode == "dna" else "",
                    str(Path("eligible.fasta").resolve()) if mode != "dna" else "",
                ]
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sourmash",
                "scripts",
                "manysketch",
                "sketch.csv",
                "--singleton",
                "-p",
                f"{mode},k={k},scaled=5",
                "-c",
                cpus,
                "-o",
                "fragments.sig.zip",
            ],
            check=True,
        )
        seen = set()
        for sig in sourmash.load_file_as_signatures("fragments.sig.zip"):
            name = sig.name.split()[0]
            if name not in eligible or name in seen:
                raise ValueError("Unexpected sketch record")
            seen.add(name)
            expected = sourmash.MinHash(**kwargs)
            if mode == "dna":
                expected.add_sequence(eligible[name], force=True)
            else:
                expected.add_protein(eligible[name])
            if set(expected.hashes) != set(sig.minhash.hashes):
                raise ValueError("Branchwater and Sourmash hash disagreement")
            grouped[name if mode == "dna" else membership[name]].add_many(
                list(sig.minhash.hashes)
            )
        if seen != set(eligible):
            raise ValueError("Branchwater omitted eligible records")
    with SaveSignaturesToLocation("query.sig.zip") as out:
        for name, mh in grouped.items():
            out.add(sourmash.SourmashSignature(mh, name=name))
    save(
        "coverage.json",
        dict(
            protocol=PROTOCOL,
            mode=mode,
            k=k,
            queries=len(dna),
            eligible_fragments=len(eligible),
            nonempty={
                str(s): sum(bool(len(m.downsample(scaled=s))) for m in grouped.values())
                for s in (5, 10, 100)
            },
        ),
    )


def search(query, target, protocol, source, database, mode, k, scale, cpus):
    import sourmash

    check_versions()
    meta = json.loads(Path(protocol).read_text())
    if meta.get("protocol") != PROTOCOL or meta.get("sha256") != sha(target):
        raise ValueError(
            "Target protocol or checksum differs; rebuild in a separate versioned directory"
        )
    if target.endswith(".sbt.zip"):
        raise ValueError("SBT is prohibited")
    signatures = list(sourmash.load_file_as_signatures(query))
    n = sum(bool(len(s.minhash.downsample(scaled=int(scale)))) for s in signatures)
    if n:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sourmash",
                "scripts",
                "manysearch",
                query,
                target,
                "-m",
                "DNA" if mode == "dna" else mode,
                "-k",
                k,
                "-s",
                scale,
                "-t",
                "0",
                "-N",
                "--ignore-abundance",
                "-c",
                cpus,
                "-o",
                "raw-matches.csv",
            ],
            check=True,
        )
    else:
        Path("matches.csv").write_text(
            "query_name,match_name,intersect_hashes,containment\n"
        )
    if n:
        with open("raw-matches.csv") as src, open("matches.csv", "w") as dest:
            reader = csv.DictReader(src)
            if not reader.fieldnames or "intersect_hashes" not in reader.fieldnames:
                raise ValueError("Search output schema changed")
            writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if int(row["intersect_hashes"]) > 0:
                    writer.writerow(row)
    else:
        Path("raw-matches.csv").write_text(Path("matches.csv").read_text())
    with open("matches.csv") as f:
        rows = sum(1 for _ in csv.DictReader(f))
    save(
        "search.json",
        dict(
            protocol=PROTOCOL,
            source=source,
            database=database,
            mode=mode,
            scaled=int(scale),
            query_sha256=sha(query),
            target_sha256=sha(target),
            matches_sha256=sha("matches.csv"),
            rows=rows,
            searchable_queries=n,
            status="completed" if n else "unsearchable",
            rule="any nonzero overlap; candidate matches only",
        ),
    )


def target(database, kind, mode, k, scale, validator, *sources):
    """Keep one signature per reference protein or per external RNA-seq run."""
    import sourmash
    from sourmash.save_load import SaveSignaturesToLocation
    from Bio import SeqIO
    from dekupl_translation import translate_context

    check_versions()
    k, scale = int(k), int(scale)
    kwargs = dict(
        n=0,
        ksize=k,
        scaled=scale,
        is_protein=mode == "protein",
        dayhoff=mode == "dayhoff",
        hp=mode == "hp",
    )
    if kind not in ("protein", "reads") or (kind == "protein" and mode == "dna"):
        raise ValueError("Unsupported target kind/mode")
    if kind == "reads":
        if len(sources) != 2:
            raise ValueError("External target requires two paired trimmed files")
        subprocess.run([str(Path(validator).resolve()), *sources], check=True)
    mh = sourmash.MinHash(**kwargs)
    records = 0
    seen = set()
    with SaveSignaturesToLocation("target.sig.zip") as dest:
        for path in sources:
            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rt") as f:
                for record in SeqIO.parse(f, "fasta" if kind == "protein" else "fastq"):
                    records += 1
                    seq = str(record.seq)
                    if kind == "protein":
                        if record.id in seen:
                            raise ValueError("Duplicate reference protein ID")
                        seen.add(record.id)
                        one = sourmash.MinHash(**kwargs)
                        for stretch in seq.split("*"):
                            if len(stretch) >= 30:
                                one.add_protein(stretch)
                        dest.add(sourmash.SourmashSignature(one, name=record.id))
                    elif mode == "dna":
                        mh.add_sequence(seq, force=True)
                    else:
                        fragments, _, _ = translate_context({"read": seq})
                        for peptide in fragments.values():
                            mh.add_protein(peptide)
        if kind == "reads":
            dest.add(sourmash.SourmashSignature(mh, name=database))
    if not records:
        raise ValueError("Empty target input")
    save(
        "target.protocol.json",
        dict(
            protocol=PROTOCOL,
            sha256=sha("target.sig.zip"),
            database=database,
            kind=kind,
            mode=mode,
            k=k,
            scaled=scale,
            records=records,
            granularity="per protein" if kind == "protein" else "per run",
            inputs={str(p): sha(p) for p in sources},
            compressed_input_bytes=sum(Path(p).stat().st_size for p in sources),
            sketch_bytes=Path("target.sig.zip").stat().st_size,
        ),
    )


def evidence(reference, mapping, paper, annotations, *folders):
    """Retain exact gene links and both discovery sources without assigning function."""
    from figure4_recovery import encode, exact_gene_map

    with open(mapping) as f:
        tx = dict(csv.reader(f, delimiter="\t"))
    query_sources = {}
    matches = []
    models = {}
    source_protocols = set()
    search_sources = set()
    alignment_files = []
    pfam_files = []
    for folder in map(Path, folders):
        if (folder / "origins.json").exists():
            source_protocols.add(
                json.loads((folder / "protocol.json").read_text())["source"]
            )
            origins = json.loads((folder / "origins.json").read_text())
            for name, seq in fasta(folder / "dna.fasta").items():
                if name in query_sources:
                    raise ValueError("Query ID collision between discovery sources")
                query_sources[name] = dict(sequence=seq, origin=origins[name])
        elif (folder / "matches.csv").exists():
            meta = json.loads((folder / "search.json").read_text())
            if meta.get("matches_sha256") != sha(folder / "matches.csv"):
                raise ValueError("Search result checksum changed before evidence join")
            search_sources.add(meta["source"])
            with (folder / "matches.csv").open() as f:
                matches.extend(
                    {
                        **r,
                        "source": meta["source"],
                        "database": meta["database"],
                        "mode": meta["mode"],
                        "scaled": meta["scaled"],
                    }
                    for r in csv.DictReader(f)
                )
        elif (folder / "context.paf").exists():
            alignment_files.append(
                dict(
                    path=str(folder / "context.paf"), sha256=sha(folder / "context.paf")
                )
            )
        elif (folder / "domains.domtblout").exists():
            pfam_files.append(
                dict(
                    path=str(folder / "domains.domtblout"),
                    sha256=sha(folder / "domains.domtblout"),
                )
            )
        elif (folder / "all-features.tsv.gz").exists():
            with gzip.open(folder / "all-features.tsv.gz", "rt") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    if row["assay"] == "genes":
                        models.setdefault(row["tag"], []).append(row)
    if source_protocols != {"splash", "dekupl"} or search_sources != {
        "splash",
        "dekupl",
    }:
        raise ValueError(
            "Both discovery sources and their searches must finish before joining"
        )
    seeds = {}
    for qid, row in query_sources.items():
        origin = row["origin"]
        if origin["kind"] == "selected seed":
            seeds.setdefault(encode(row["sequence"]), []).append(qid)
    hits = exact_gene_map(Path(reference), tx, set(seeds))
    query_genes = {}
    for key, qids in seeds.items():
        for qid in qids:
            query_genes[qid] = sorted(hits.get(key, set()))
    for qid, row in query_sources.items():
        origin = row["origin"]
        parent = origin.get("seed", origin.get("representative_seed"))
        row["genes"] = query_genes.get(parent, [])
        row["gene_link"] = (
            "exact seed containment; context is not a confirmed transcript"
        )
    with open(paper) as f:
        paper_rows = list(csv.DictReader(f, delimiter="\t"))
    if paper_rows and "gene_id" not in paper_rows[0]:
        raise ValueError("Paper membership requires gene_id")
    annotation_rows = {}
    with open(annotations) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["record_id"] in annotation_rows:
                raise ValueError("Duplicate annotation record")
            annotation_rows[row["record_id"]] = row
    for row in matches:
        if row["query_name"] not in query_sources:
            raise ValueError("Match query has no origin")
        row["query_genes"] = query_sources[row["query_name"]]["genes"]
        row["reference_annotation"] = annotation_rows.get(
            row["match_name"], {"status": "unknown"}
        )
        row["interpretation"] = (
            "Candidate overlap; reference Pfam/GO is not a query domain or function"
        )
    ids = (
        set(models)
        | {r["gene_id"] for r in paper_rows}
        | {g for row in query_sources.values() for g in row["genes"]}
    )
    from collections import defaultdict

    gene_sequences, gene_matches, paper_membership = (
        defaultdict(dict),
        defaultdict(list),
        defaultdict(list),
    )
    for qid, row in query_sources.items():
        for gid in row["genes"]:
            gene_sequences[gid][qid] = row
    for row in matches:
        for gid in row["query_genes"]:
            gene_matches[gid].append(row)
    for row in paper_rows:
        paper_membership[row["gene_id"]].append(row)
    result = [
        dict(
            gene_id=gid,
            paper_membership=paper_membership[gid],
            tests=models.get(gid, []),
            sequences=gene_sequences[gid],
            matches=gene_matches[gid],
        )
        for gid in sorted(ids)
    ]
    save(
        "gene-evidence.json",
        dict(
            protocol=PROTOCOL,
            status="completed evidence join",
            reference_sha256=sha(reference),
            mapping_sha256=sha(mapping),
            paper_sha256=sha(paper),
            annotations_sha256=sha(annotations),
            genes=result,
            unmatched_sequences={
                q: r for q, r in query_sources.items() if not r["genes"]
            },
            direct_query_pfam=pfam_files or "not measured",
            context_alignment=alignment_files or "not measured",
            sequence_confirmation="Exact seed containment; PAF alignments and query Pfam calls require separate interpretation",
        ),
    )


if __name__ == "__main__":
    actions = {
        f.__name__: f
        for f in (
            pair,
            check_recruit,
            genes,
            seeds,
            merge,
            splash,
            compact,
            translate,
            sketch,
            search,
            target,
            evidence,
        )
    }
    actions[sys.argv[1]](*sys.argv[2:])
