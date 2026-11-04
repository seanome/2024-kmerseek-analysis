"""Score each array sample on the signature proteins and compare post with pre.

Usage: score.py <signature_proteins.tsv> <groups.tsv> <published_fig4.tsv> <platforms.tsv> <series folder>
                <landing folder> <out folder>
Each sample: a protein's value is the mean intensity_weight of the features landing on it. Each protein
is standardised across a series with StepMiner's threshold, (value - (threshold + 0.5)) / (3 x sd), as in
Sahoo et al. 2007. The score is -1 x the M1 mean + 1 x the M2 mean, or M1 alone for GSE27536.
A second score, weights -1, 1 and 2 on the sums of clusters 13, 14 and 3, is the formula of Ghosh et al.
2023. Rule fixed before running: reproduced when the AUC is on the same side of 0.5 as the published AUC
and within 0.15 of it. The AUC treats the post group as the positive one.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from geo2kmers.geo import read_series_matrix
from geo2kmers.sketch import probe_weights, to_float

signature, groups_file, published_file, platforms_file, series, landing, out = sys.argv[1:]
M1_ONLY = {"GSE27536"}
RULE_DISTANCE = 0.15
METHOD = "stepminer_modified_z"


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def stepminer_threshold(values):
    """Halfway between the two means of the best one-step fit of the sorted values (Sahoo et al. 2007)."""
    a = np.sort(values)
    if a[0] == a[-1]:
        return float(a[0])
    n = len(a)
    s, s2 = np.cumsum(a), np.cumsum(a * a)
    left_size = np.arange(1, n)
    left = s2[left_size - 1] - s[left_size - 1] ** 2 / left_size
    right = s2[-1] - s2[left_size - 1] - (s[-1] - s[left_size - 1]) ** 2 / (n - left_size)
    b = int(np.argmin(left + right)) + 1
    return float((a[:b].mean() + a[b:].mean()) / 2)


def standardise(values):
    sd = values.std(ddof=1)
    return None if sd == 0 else (values - (stepminer_threshold(values) + 0.5)) / (3 * sd)


def verdict(auc, published):
    close = abs(auc - published) <= RULE_DISTANCE and (auc > 0.5) == (published > 0.5)
    return "reproduced" if close else "not reproduced"


def auc_and_p(pre, post):
    u, p = mannwhitneyu(post, pre, alternative="two-sided")
    return u / (len(pre) * len(post)), p


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
    in_cluster = {c: [p for p in z if c in cluster_of[p]] for c in (13, 14, 3)}
    m2 = [p for p in z if cluster_of[p] & {14, 3}]
    zero = np.zeros(len(samples))
    mean_of = lambda proteins: np.mean([z[p] for p in proteins], axis=0) if proteins else zero
    sum_of = lambda proteins: np.sum([z[p] for p in proteins], axis=0) if proteins else zero
    main = -mean_of(in_cluster[13]) + (zero if gse in M1_ONLY else mean_of(m2))
    ghosh = -sum_of(in_cluster[13])
    if gse not in M1_ONLY:
        ghosh = ghosh + sum_of(in_cluster[14]) + 2 * sum_of(in_cluster[3])
    return dict(zip(samples, main)), dict(zip(samples, ghosh)), len(in_cluster[13]), len(m2)


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
        "landing_share": "",
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
            "landing_share": platform[gse]["landing_share"],
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
