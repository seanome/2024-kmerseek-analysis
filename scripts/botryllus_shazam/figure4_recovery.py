"""Join completed SPLASH/DE-kupl evidence to the published immune gene IDs."""

import argparse, csv, gzip, hashlib, json, math
from collections import defaultdict, Counter
from pathlib import Path


def encode(seq):
    f = r = 0
    for i, c in enumerate(seq.upper()):
        if c not in "ACGT":
            raise ValueError("Non-DNA query")
        n = "ACGT".index(c)
        f = (f << 2) | n
        r |= (3 - n) << (2 * i)
    return min(f, r)


def windows(seq, k=31):
    f = r = valid = 0
    mask = (1 << (2 * k)) - 1
    for c in seq.upper():
        n = "ACGT".find(c)
        if n < 0:
            f = r = valid = 0
            continue
        f = ((f << 2) | n) & mask
        r = (r >> 2) | ((3 - n) << (2 * (k - 1)))
        valid += 1
        if valid >= k:
            yield min(f, r)


def exact_gene_map(fasta, tx_to_gene, keys):
    hits = defaultdict(set)
    name = None
    sequence = []

    def scan():
        if name is None:
            return
        if name not in tx_to_gene:
            raise ValueError(
                "Reference ID absent from validated transcript-to-gene map: " + name
            )
        gene = tx_to_gene[name]
        for key in windows("".join(sequence)):
            if key in keys:
                hits[key].add(gene)

    with fasta.open() as f:
        for line in f:
            if line.startswith(">"):
                scan()
                name = line[1:].split()[0].split("|")[0]
                sequence = []
            else:
                sequence.append(line.strip())
    scan()
    return hits


def run(root, inputs, out):
    abundance = root / "abundance-20261024"
    splash = root / "splash-abundance-sequence-evidence-20261024"
    complete = json.loads((abundance / "abundance-tests.complete.json").read_text())
    json.loads((splash / "evidence.complete.json").read_text())
    fasta = root / "references/botznik-transcripts.fa"
    if (
        hashlib.sha256(fasta.read_bytes()).hexdigest()
        != complete["reference"]["fasta_sha256"]
    ):
        raise ValueError("Reference changed")
    tx = dict(csv.reader((abundance / "transcript-to-gene.tsv").open(), delimiter="\t"))
    sets = list(
        csv.DictReader((inputs / "immune-gene-sets.tsv").open(), delimiter="\t")
    )
    if len(sets) != 146 or len({r["gene_id"] for r in sets}) != 146:
        raise ValueError("Paper gene-set membership differs")
    for row in sets:
        row.update(paper_time="immune model", paper_direction="predefined signed score")
    hubs = json.loads((inputs / "hub-gene-memberships.json").read_text())
    for h in hubs:
        if h["status"] == "unique exact label within published comparison":
            sets.append(
                dict(
                    gene_id=h["gene_ids"][0],
                    gene_name=h["paper_label"],
                    role="hub: " + h["hub"],
                    paper_time=h["time"],
                    paper_direction=h["direction"],
                )
            )
    de = {}
    significant = {}
    keys = set()
    for folder in sorted((abundance / "models").iterdir()):
        contrast = folder.name
        with gzip.open(folder / "genes/all-features.tsv.gz", "rt") as f:
            de[contrast] = {r["tag"]: r for r in csv.DictReader(f, delimiter="\t")}
        with gzip.open(folder / "kmers/all-features.tsv.gz", "rt") as f:
            records = []
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    q = float(row["padj"])
                except ValueError:
                    continue
                if not math.isfinite(q) or q >= 0.05:
                    continue
                if len(row["tag"]) != 31:
                    raise ValueError("Unexpected DE-kupl k-mer length")
                key = encode(row["tag"])
                keys.add(key)
                records.append((key, row))
            significant[contrast] = records
    if len(keys) > 10000000:
        raise ValueError(
            "More than ten million significant k-mers; use a partitioned mapping job"
        )
    mapped = exact_gene_map(fasta, tx, keys)
    splash_hits = defaultdict(set)
    with gzip.open(splash / "anchor-gene-abundance-evidence.tsv.gz", "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["reference_gene"]:
                splash_hits[(row["contrast"], row["reference_gene"])].add(row["anchor"])
    out.mkdir(exist_ok=False)
    kmers = defaultdict(set)
    ambiguous = defaultdict(set)
    with gzip.open(out / "significant-kmer-gene-mappings.tsv.gz", "wt") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "contrast",
                "kmer",
                "padj",
                "log2foldchange",
                "mapping_status",
                "reference_genes",
            ]
        )
        for contrast, records in significant.items():
            for key, row in records:
                genes = mapped.get(key, set())
                status = (
                    "unique gene"
                    if len(genes) == 1
                    else "multiple genes" if genes else "no exact transcript match"
                )
                for gene in genes:
                    (kmers if len(genes) == 1 else ambiguous)[contrast, gene].add(
                        row["tag"]
                    )
                w.writerow(
                    [
                        contrast,
                        row["tag"],
                        row["padj"],
                        row["log2FoldChange"],
                        status,
                        ";".join(sorted(genes)),
                    ]
                )
    summaries = []
    with (out / "paper-gene-recovery.tsv").open("w") as f:
        columns = [
            "contrast",
            "gene_id",
            "gene_name",
            "role",
            "paper_time",
            "paper_direction",
            "splash_anchors",
            "unique_gene_DE_kmers",
            "ambiguous_DE_kmers",
            "gene_padj",
            "gene_log2foldchange",
            "gene_status",
        ]
        w = csv.DictWriter(f, columns, delimiter="\t")
        w.writeheader()
        for contrast in de:
            counters = Counter()
            for gene in sets:
                gid = gene["gene_id"]
                r = de[contrast].get(gid, {})
                sa = len(splash_hits[contrast, gid])
                dk = len(kmers[contrast, gid])
                am = len(ambiguous[contrast, gid])
                w.writerow(
                    dict(
                        contrast=contrast,
                        **gene,
                        splash_anchors=sa,
                        unique_gene_DE_kmers=dk,
                        ambiguous_DE_kmers=am,
                        gene_padj=r.get("padj", "NA"),
                        gene_log2foldchange=r.get("log2FoldChange", "NA"),
                        gene_status=r.get(
                            "status", "not tested: absent or below count filter"
                        )
                    )
                )
                if gene["role"] in ("proimmune", "cell_repair"):
                    counters["immune_model_genes"] += 1
                    counters["SPLASH_recovered_immune_genes"] += bool(sa)
                    counters["DEkupl_uniquely_recovered_immune_genes"] += bool(dk)
                    counters["both_immune"] += bool(sa and dk)
            summaries.append(dict(contrast=contrast, **counters))
    record = dict(
        status="completed",
        scope="Exact paper gene IDs; full-depth SPLASH unique gene assignments and significant DE-kupl 31-mers with exact transcript matches",
        summary=summaries,
        significant_unique_kmers=len(keys),
        input_completion_sha256=hashlib.sha256(
            (abundance / "abundance-tests.complete.json").read_bytes()
        ).hexdigest(),
        limitations="Transcript matches do not localize genomic junctions; ambiguous genes excluded from unique recovery. Significant k-mer abundance need not imply gene-level DE. No original Figure 4E score reproduction inferred.",
    )
    (out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("inputs", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.root, a.inputs, a.out)
