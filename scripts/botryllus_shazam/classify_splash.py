"""Join SPLASH anchors to gene DE and sequence-alignment evidence, without a splicing verdict."""

import bisect
from collections import defaultdict, Counter
import csv
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from full_query_provenance import validate, sha


def blocks_and_introns(start, cigar):
    blocks = []
    introns = []
    position = start
    begin = start
    for count, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        n = int(count)
        if op == "N":
            if position > begin:
                blocks.append((begin, position))
            introns.append((position, position + n))
            position += n
            begin = position
        elif op in ("M", "D", "=", "X"):
            position += n
    if position > begin:
        blocks.append((begin, position))
    return blocks, introns


def union_length(intervals):
    total = 0
    end = None
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += right - left
            end = right
        elif right > end:
            total += right - end
            end = right
    return total


def annotation(path):
    exons = defaultdict(set)
    transcripts = defaultdict(set)
    gene_names = {}
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip().split("\t")
            if len(c) != 9:
                raise ValueError("Invalid GTF")
            a = dict(re.findall(r'(\S+) "([^"]*)"', c[8]))
            g = a.get("gene_id")
            t = a.get("transcript_id")
            if g:
                gene_names[g] = a.get("gene_name", g)
            if c[2] == "exon" and g:
                interval = (int(c[3]) - 1, int(c[4]))
                exons[c[0]].add((*interval, g))
                if t:
                    transcripts[(c[0], t, g)].add(interval)
    junctions = defaultdict(set)
    for (chrom, t, g), values in transcripts.items():
        ordered = sorted(values)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] > left[1]:
                junctions[(chrom, left[1], right[0])].add(g)
    index = {}
    for chrom, values in exons.items():
        values = sorted(values)
        maxima = []
        peak = -1
        for a, b, g in values:
            peak = max(peak, b)
            maxima.append(peak)
        index[chrom] = (values, [x[0] for x in values], maxima)
    return index, junctions, gene_names


def genes_for_blocks(index, chrom, blocks):
    if chrom not in index:
        return set()
    values, starts, maxima = index[chrom]
    overlaps = defaultdict(list)
    for left, right in blocks:
        i = bisect.bisect_left(starts, right) - 1
        while i >= 0 and maxima[i] > left:
            a, b, g = values[i]
            if b > left:
                overlaps[g].append((max(a, left), min(b, right)))
            i -= 1
    return {g for g, intervals in overlaps.items() if union_length(intervals) >= 31}


def run(root, code, genome, aligner):
    discovery = root / "full-depth-discovery-20261023"
    proof = validate(root, discovery, code / "ena-runs.tsv")
    abundance = root / "abundance-20261024"
    completed = json.loads((abundance / "abundance-tests.complete.json").read_text())
    gtf = root / "references/nc_nsi_nf.gtf"
    if sha(gtf) != completed["reference"]["gtf_sha256"]:
        raise ValueError("Gene models differ from abundance reference")
    out = root / "splash-abundance-sequence-evidence-20261024"
    out.mkdir(exist_ok=False)
    index, junctions, names = annotation(gtf)
    with genome.open() as f:
        chromosomes = {line[1:].split()[0] for line in f if line.startswith(">")}
    if not set(index) <= chromosomes:
        raise ValueError("Genome and GTF chromosome identifiers differ")
    paf = out / "extensions-to-genome.paf"
    command = [
        str(aligner),
        "-x",
        "splice",
        "-c",
        "--cs=long",
        "--secondary=yes",
        "-t",
        os.environ["SLURM_CPUS_PER_TASK"],
        str(genome),
        str(discovery / "extensions.fasta"),
    ]
    with paf.open("w") as f, (out / "minimap2.log").open("w") as log:
        subprocess.run(command, stdout=f, stderr=log, check=True)
    header_to_anchor = {}
    sequences = {}
    q = None
    with (discovery / "extensions.fasta").open() as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split()
                q = parts[0]
                header_to_anchor[q] = next(
                    x.split("=", 1)[1] for x in parts[1:] if x.startswith("anchor=")
                )
                sequences[q] = ""
            else:
                sequences[q] += line.strip()
    anchors = [
        r["anchor"]
        for r in csv.DictReader(
            (discovery / "all-significant/selected_anchors.tsv").open(), delimiter="\t"
        )
    ]
    evidence = {a: dict(genes=set(), flags=set(), extensions=set()) for a in anchors}
    for q, a in header_to_anchor.items():
        evidence[a]["extensions"].add(q)
    with paf.open() as f:
        for line in f:
            c = line.rstrip().split("\t")
            q = c[0]
            length = int(c[1])
            coverage = (int(c[3]) - int(c[2])) / length
            identity = int(c[9]) / int(c[10]) if int(c[10]) else 0
            if int(c[11]) < 20 or coverage < 0.9 or identity < 0.95:
                continue
            tags = {x.split(":", 2)[0]: x.split(":", 2)[2] for x in c[12:]}
            if "cg" not in tags:
                raise ValueError("Missing requested CIGAR")
            blocks, introns = blocks_and_introns(int(c[7]), tags["cg"])
            genes = genes_for_blocks(index, c[5], blocks)
            item = evidence[header_to_anchor[q]]
            item["genes"].update(genes)
            if introns:
                known = set().union(
                    *(
                        junctions.get((c[5], left, right), set())
                        for left, right in introns
                    )
                )
                item["flags"].add(
                    "annotated splice-junction alignment candidate"
                    if known & genes
                    else "unannotated gapped alignment candidate"
                )
            if any(x in tags.get("cs", "") for x in ("*", "+", "-")):
                item["flags"].add("mismatch or indel alignment candidate")
            if not introns:
                item["flags"].add("contiguous alignment")
    design = json.loads((code / "abundance-design-20261024.json").read_text())
    summaries = {}
    with gzip.open(out / "anchor-gene-abundance-evidence.tsv.gz", "wt") as f:
        columns = [
            "anchor",
            "contrast",
            "extension_records",
            "reference_gene",
            "reference_gene_label",
            "gene_status",
            "gene_padj",
            "gene_log2foldchange",
            "sequence_alignment_evidence",
            "splicing_inference",
            "composition_inference",
        ]
        writer = csv.DictWriter(f, columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for contrast in design["contrasts"]:
            source = (
                abundance / "models" / contrast["name"] / "genes/all-features.tsv.gz"
            )
            with gzip.open(source, "rt") as stream:
                de = {r["tag"]: r for r in csv.DictReader(stream, delimiter="\t")}
            count = Counter()
            for anchor, item in evidence.items():
                gene = next(iter(item["genes"])) if len(item["genes"]) == 1 else ""
                row = de.get(gene, {})
                status = row.get(
                    "status",
                    (
                        "not tested: no unique gene assignment"
                        if not gene
                        else "not tested: absent or below count filter"
                    ),
                )
                flags = "; ".join(sorted(item["flags"])) or (
                    "no emitted extension"
                    if not item["extensions"]
                    else "no qualifying unique genomic alignment"
                )
                writer.writerow(
                    dict(
                        anchor=anchor,
                        contrast=contrast["name"],
                        extension_records=len(item["extensions"]),
                        reference_gene=gene,
                        reference_gene_label=names.get(gene, ""),
                        gene_status=status,
                        gene_padj=row.get("padj", "NA"),
                        gene_log2foldchange=row.get("log2FoldChange", "NA"),
                        sequence_alignment_evidence=flags,
                        splicing_inference="candidate alignment evidence only; no differential junction-usage test",
                        composition_inference="sample-dependent SPLASH significance; not a treatment-specific composition test",
                    )
                )
                count[status] += 1
            summaries[contrast["name"]] = dict(count)
    record = dict(
        status="gene-DE overlap and candidate sequence evidence completed",
        discovery=proof,
        abundance_complete_sha256=sha(abundance / "abundance-tests.complete.json"),
        genome_sha256=sha(genome),
        gtf_sha256=sha(gtf),
        aligner_sha256=sha(aligner),
        command=command,
        thresholds=dict(
            min_mapq=20,
            min_query_coverage=0.9,
            min_identity=0.95,
            min_gene_exon_overlap_nt=31,
        ),
        contrasts=summaries,
        interpretation="Gene DE, SPLASH composition and candidate splice/mismatch evidence are independent axes. No definitive differential splicing, causal treatment composition or variant call is made.",
    )
    (out / "evidence.complete.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
