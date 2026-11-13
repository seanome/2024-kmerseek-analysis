"""Score each array sample on the signature proteins and compare post with pre.

Usage: score.py <signature_proteins.tsv> <groups.tsv> <published_fig4.tsv> <platforms.tsv> <series folder>
                <landing folder> <out folder>
Each sample: a protein's value is the mean intensity_weight of the features landing on it. Each protein
is standardised across a series with StepMiner's threshold, (value - (threshold + 0.5)) / (3 x sd), as in
Voskoboynik et al. 2024 and PR 212, using the threshold of Sahoo et al. 2007. The score is -1 x the M1 mean + 1 x the M2 mean, or M1 alone for GSE27536.
A second score puts the weights in Ghosh et al. 2023's table, -1, 1 and 2, on the sums of clusters 13, 14 and 3,
as PR 212 found the published code does. Rule fixed before running: reproduced when the AUC is on the same side of 0.5 as the published AUC
and within 0.15 of it. The AUC treats the post group as the positive one.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo2kmers.geo import read_series_matrix
from geo2kmers.scoring import auc_and_p, passes, scores, standardise
from geo2kmers.sketch import probe_weights, to_float

signature, groups_file, published_file, platforms_file, series, landing, out = sys.argv[1:]
M1_ONLY = {"GSE27536"}
METHOD = "stepminer_modified_z"


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def verdict(auc, published):
    return "reproduced" if passes(auc, published) else "not reproduced"


signature_rows = rows_of(signature)
cluster_of = defaultdict(set)
for r in signature_rows:
    cluster_of[r["gene"]].add(int(r["cluster"]))
platform = {r["gse"]: r for r in rows_of(platforms_file)}
groups = defaultdict(lambda: ([], []))
for r in rows_of(groups_file):
    groups[r["dataset"], r["comparison"]][int(r["group"])].append(r["sample"])


def score_series(gse):
    """Return {sample: main score}, {sample: Ghosh score}, and how many signature proteins were used."""
    _, samples, rows = read_series_matrix(Path(series) / gse / f"{gse}_series_matrix.txt.gz")
    features = defaultdict(list)
    for r in rows_of(Path(landing) / f"{gse}.landing.tsv"):
        features[r["protein"]].append(r["feature"])
    weights = {}
    for column, sample in enumerate(samples):
        values = {name: to_float(row[column]) for name, row in rows.items()}
        weights[sample] = probe_weights({name: v for name, v in values.items() if v is not None})
    z = {}
    for protein, names in features.items():
        names = [n for n in names if n in rows]
        if names:
            scored = standardise(np.array([np.mean([weights[s].get(n, 0.0) for n in names]) for s in samples]))
            if scored is not None:
                z[protein] = scored
    main, ghosh = scores(z, cluster_of, gse in M1_ONLY)
    m1_used = sum(13 in cluster_of[p] for p in z)
    m2_used = sum(bool(cluster_of[p] & {14, 3}) for p in z)
    return dict(zip(samples, main)), dict(zip(samples, ghosh)), m1_used, m2_used


results, per_sample = [], []
cache = {}
for r in rows_of(published_file):
    if r["panel"] not in ("4C", "4D"):
        continue
    gse = r["dataset"]
    row = {
        "GSE": gse,
        "group_pair": f"{r['group0']} vs {r['group1']}",
        "n_pre": "",
        "n_post": "",
        "published_auc": r["auc"],
        "reproduced_auc": "",
        "mann_whitney_p": "",
        "share_of_probes_with_a_kmer_in_proteome": "",
        "m1_proteins_used": "",
        "m2_proteins_used": "",
        "auc_ghosh_weights": "",
        "score_method": METHOD,
        "verdict": "not covered by geo2kmers",
        "verdict_ghosh_weights": "not covered by geo2kmers",
    }
    if gse in platform and platform[gse]["type"] == "array":
        pre, post = groups[gse, r["comparison"]]
        if gse not in cache:
            cache[gse] = score_series(gse)
        main, ghosh, m1_used, m2_used = cache[gse]
        pre_scores = np.array([main[s] for s in pre])
        post_scores = np.array([main[s] for s in post])
        auc, p = auc_and_p(pre_scores, post_scores)
        ghosh_auc = auc_and_p(*(np.array([ghosh[s] for s in g]) for g in (pre, post)))[0]
        row |= {
            "n_pre": len(pre),
            "n_post": len(post),
            "reproduced_auc": f"{auc:.3f}",
            "mann_whitney_p": f"{p:.3g}",
            "share_of_probes_with_a_kmer_in_proteome": platform[gse]["share_of_probes_with_a_kmer_in_proteome"],
            "m1_proteins_used": m1_used,
            "m2_proteins_used": m2_used,
            "auc_ghosh_weights": f"{ghosh_auc:.3f}",
            "verdict": verdict(auc, float(r["auc"])),
            "verdict_ghosh_weights": verdict(ghosh_auc, float(r["auc"])),
        }
        per_sample += [(gse, r["comparison"], "pre", s, main[s]) for s in pre]
        per_sample += [(gse, r["comparison"], "post", s, main[s]) for s in post]
    results.append(row)
with open(f"{out}/auc.tsv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(results)
with open(f"{out}/sample_scores.tsv", "w") as f:
    f.write("gse\tcomparison\tgroup\tsample\tscore\n")
    f.writelines(f"{g}\t{c}\t{w}\t{s}\t{v:.6f}\n" for g, c, w, s, v in per_sample)
