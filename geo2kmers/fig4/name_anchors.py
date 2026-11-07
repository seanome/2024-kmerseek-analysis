"""Name the genes behind the SPLASH anchors that sit inside one gene cluster, and say what each probe measures.

Usage: name_anchors.py <splash_hits.tsv> <probes.tsv.gz> <probe_design.tsv.gz> <GPL.txt> <groups.tsv>
                       <signature_proteins.tsv> <dataset> <comparison> <out prefix>
Takes hits with features = 1. A 25-letter probe holds one anchor, its first 15 letters, so an anchor's targets are
the other probes that start the same way. Each probe's pre and post share is its mean intensity over the anchor's
probes. A probe set named JUC covers an exon-exon junction and one named PSR covers a stretch of exon. Writes
<prefix>.anchor_genes.tsv, one row per anchor, and <prefix>.anchor_probes.tsv, one row per probe. Names come from
the array's annotation and are used for nothing but naming.
"""

import csv
import gzip
import sys
from collections import defaultdict

import numpy as np

from geo2kmers.geo import read_table

ANCHOR = 15
hits_file, probes, design, platform, groups_file, signature, dataset, comparison, prefix = sys.argv[1:]


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def genes_in(assignment):
    """The gene symbols in one gene_assignment cell, whose parts are split by /// and fields by //."""
    names = [part.split(" // ")[1].strip() for part in assignment.split(" /// ") if part.count(" // ") >= 1]
    return sorted({n for n in names if n and n != "---"})


hits = [r for r in rows_of(hits_file) if r["comparison"] == comparison and int(r["features"]) == 1]
anchors = {r["anchor"]: r for r in hits}
groups = [r for r in rows_of(groups_file) if r["dataset"] == dataset and r["comparison"] == comparison]
pre = [r["sample"] for r in groups if r["group"] == "0"]
post = [r["sample"] for r in groups if r["group"] == "1"]

by_sequence = defaultdict(list)
with gzip.open(probes, "rt") as f:
    header = f.readline().rstrip("\n").split("\t")
    pre_columns, post_columns = [[header.index(s) for s in group] for group in (pre, post)]
    for line in f:
        fields = line.rstrip("\n").split("\t")
        if fields[0].upper()[:ANCHOR] in anchors:
            values = np.array(fields[1:], dtype=float)
            by_sequence[fields[0].upper()].append(
                (values[[c - 1 for c in pre_columns]], values[[c - 1 for c in post_columns]])
            )
measured = {s: tuple(np.mean([v[i] for v in held], axis=0).mean() for i in (0, 1)) for s, held in by_sequence.items()}

what = defaultdict(list)
with gzip.open(design, "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["sequence"] and r["sequence"].upper()[:ANCHOR] in anchors:
            what[r["sequence"].upper()].append(r)

_, columns, table = read_table(platform, "platform")
clusters = {c for rows in what.values() for r in rows for c in [r["transcript_cluster"]] if c}
annotation = {r[columns[0]]: r for r in table if r[columns[0]] in clusters}
signature_genes = {r["gene"] for r in rows_of(signature)}

probe_rows, gene_rows = [], []
for anchor, hit in anchors.items():
    sequences = sorted(s for s in measured if s[:ANCHOR] == anchor)
    totals = [sum(measured[s][i] for s in sequences) for i in (0, 1)]
    genes, sets, cluster_ids, regions = set(), set(), set(), set()
    for sequence in sequences:
        rows = what[sequence]
        cluster = sorted({r["transcript_cluster"] for r in rows if r["transcript_cluster"]})
        named = sorted({g for c in cluster for g in genes_in(annotation[c]["gene_assignment"])} if annotation else [])
        kinds = sorted({r["probe_set"][:3] for r in rows})
        probe_rows.append(
            {
                "dataset": dataset,
                "anchor": anchor,
                "probe": sequence,
                "target": sequence[ANCHOR:],
                "pre_mean_intensity": f"{measured[sequence][0]:.2f}",
                "post_mean_intensity": f"{measured[sequence][1]:.2f}",
                "pre_share": f"{measured[sequence][0] / totals[0]:.3f}",
                "post_share": f"{measured[sequence][1] / totals[1]:.3f}",
                "probe_sets": ",".join(sorted({r["probe_set"] for r in rows})),
                "kinds": ",".join(kinds),
                "transcript_clusters": ",".join(cluster),
                "genes": ",".join(named),
                "exons": ",".join(sorted({r["exon_id"] for r in rows if r["exon_id"]})),
                "junction_edges": ",".join(
                    sorted({f"{r['junction_start']}-{r['junction_stop']}" for r in rows if r["junction_start"]})
                ),
            }
        )
        genes |= set(named)
        sets |= {r["probe_set"] for r in rows}
        cluster_ids |= set(cluster)
        regions |= set(kinds)
    gene_rows.append(
        {
            "dataset": dataset,
            "anchor": anchor,
            "q": hit["q"],
            "pre_top_target_share": hit["pre_top_target_share"],
            "post_top_target_share": hit["post_top_target_share"],
            "genes": ",".join(sorted(genes)),
            "transcript_clusters": ",".join(sorted(cluster_ids)),
            "probes": len(sequences),
            "probe_kinds": ",".join(sorted(regions)),
            "probe_sets": len(sets),
            "in_signature": ",".join(sorted(genes & signature_genes)),
        }
    )
gene_rows.sort(key=lambda r: float(r["q"]))
for name, rows in (("anchor_genes", gene_rows), ("anchor_probes", probe_rows)):
    with open(f"{prefix}.{name}.tsv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
