"""Ask which SPLASH anchors change their target use between the groups, and what those anchors are.

Usage: splash_groups.py <scores.tsv> <dump folder> <groups.tsv> <dataset> <features.tsv.gz> <proteins.fasta>
                        <out folder>
SPLASH does not see the groups. For each significant anchor, a sample's value is the share of the anchor's
reads on its most common target. Mann-Whitney compares post with pre, corrected by Benjamini-Hochberg, and
100 shuffles of the groups show how many anchors pass by chance. features.tsv.gz lists sequence and feature:
an anchor with one feature is inside one probe set or gene cluster, as alternative splicing would be.
"""

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from geo2kmers.landing import land_probes

FDR, SHUFFLES, ANCHOR = 0.05, 100, 15
SUPPLEMENT_ONLY = {"full_model", "early_vs_rest"}
scores, dumps, groups_file, dataset, features_file, proteins, out = sys.argv[1:]
with open(scores) as f:
    significant = {r["anchor"]: float(r["pval_opt_corrected"]) for r in csv.DictReader(f, delimiter="\t")}
counts = defaultdict(lambda: defaultdict(dict))
for path in Path(dumps).glob("*.satc.dump"):
    with open(path) as f:
        for line in f:
            count, sequence, sample = line.split()
            anchor, target = sequence[:ANCHOR], sequence[ANCHOR:]
            if anchor in significant:
                counts[anchor][target][sample] = int(count)
comparisons = defaultdict(lambda: ([], []))
with open(groups_file) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["dataset"] == dataset and r["comparison"] not in SUPPLEMENT_ONLY:
            comparisons[r["comparison"]][int(r["group"])].append(r["sample"])
features = defaultdict(set)
with gzip.open(features_file, "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        sequence = r["sequence"].upper()
        for start in range(len(sequence) - 24):
            features[sequence[start : start + 25]].add(r["feature"])


def share_table(anchor, samples):
    """Each sample's share of this anchor's reads on the anchor's most common target."""
    targets = counts[anchor]
    top = max(targets, key=lambda t: sum(targets[t].values()))
    total = np.array([sum(targets[t].get(s, 0) for t in targets) for s in samples], dtype=float)
    on_top = np.array([targets[top].get(s, 0) for s in samples], dtype=float)
    return np.divide(on_top, total, out=np.full(len(samples), np.nan), where=total > 0)


def adjusted(p):
    """Benjamini-Hochberg q values."""
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(q, 1)
    return result


def test(matrix, labels):
    """p for each anchor, post against pre, on the samples with reads."""
    p = np.ones(len(matrix))
    for i, row in enumerate(matrix):
        pre, post = row[~labels & ~np.isnan(row)], row[labels & ~np.isnan(row)]
        if len(pre) >= 3 and len(post) >= 3 and np.ptp(np.r_[pre, post]) > 0:
            p[i] = mannwhitneyu(post, pre, alternative="two-sided").pvalue
    return p


anchors = sorted(counts)
sequences = {f"{a}{t}": a for a in anchors for t in counts[a]}
anchor_features = defaultdict(set)
for sequence, anchor in sequences.items():
    anchor_features[anchor] |= features.get(sequence, set())
feature_count = {a: len(anchor_features[a]) for a in anchors}
landed = defaultdict(set)
if sequences:
    probes = {s: s for s in sequences if len(s) == 25}
    for flip in (False, True):
        table = {s: (s.translate(str.maketrans("ACGT", "TGCA")) if flip else s) for s in probes}
        for row in land_probes(table, [proteins])[2]:
            landed[sequences[row[0]]].add(row[1])
rng = np.random.default_rng(1)
summary, detail = [], []
for comparison, (pre, post) in comparisons.items():
    samples = pre + post
    labels = np.array([False] * len(pre) + [True] * len(post))
    matrix = np.array([share_table(a, samples) for a in anchors]) if anchors else np.empty((0, len(samples)))
    q = adjusted(test(matrix, labels)) if anchors else np.array([])
    chance = (
        [int((adjusted(test(matrix, rng.permutation(labels))) < FDR).sum()) for _ in range(SHUFFLES)]
        if anchors
        else [0]
    )
    hits = [i for i in np.argsort(q) if q[i] < FDR]
    summary.append(
        {
            "dataset": dataset,
            "comparison": comparison,
            "significant_anchors": len(anchors),
            "anchors_inside_one_feature": sum(feature_count[a] == 1 for a in anchors),
            "anchors_differing_between_groups": len(hits),
            "mean_with_shuffled_groups": f"{np.mean(chance):.2f}",
            "highest_with_shuffled_groups": max(chance),
            "hits_inside_one_feature": sum(feature_count[anchors[i]] == 1 for i in hits),
            "hits_on_signature_proteins": sum(bool(landed[anchors[i]]) for i in hits),
        }
    )
    detail += [
        {
            "dataset": dataset,
            "comparison": comparison,
            "anchor": anchors[i],
            "q": f"{q[i]:.3g}",
            "pre_top_target_share": f"{np.nanmean(matrix[i][~labels]):.3f}",
            "post_top_target_share": f"{np.nanmean(matrix[i][labels]):.3f}",
            "features": feature_count[anchors[i]],
            "signature_proteins": ",".join(sorted(landed[anchors[i]])),
        }
        for i in hits
    ]
for name, rows in (("summary", summary), ("hits", detail)):
    with open(f"{out}/{dataset}.splash_{name}.tsv", "w", newline="") as f:
        fields = list(rows[0]) if rows else ["dataset"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
