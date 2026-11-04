"""How many comparisons would pass the rule by chance? Run after scoring.

Usage: chance.py <signature_proteins.tsv> <groups.tsv> <published_fig4.tsv> <auc.tsv> <series folder>
                 <landing folder> <out folder>
Two chance models, 1000 draws each, both scores, every series in the same draw:
  random features: each protein keeps its number of features, drawn at random from the array, so the
    landings say nothing. This does not match the features on expression level.
  shuffled labels: real landings, but the proteins' cluster labels are shuffled, so the signature says nothing.
Writes chance_test.tsv (how often at least the observed number of comparisons pass) and
chance_by_comparison.tsv (how often each comparison passes).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo2kmers.geo import read_series_matrix
from geo2kmers.scoring import auc_and_p, passes, scores, standardise
from geo2kmers.sketch import BACKGROUND_PERCENTILE, to_float

signature, groups_file, published_file, auc_file, series, landing, out = sys.argv[1:]
DRAWS = 1000
SEED = 11
M1_ONLY = {"GSE27536"}
MODELS = ("random features", "shuffled labels")
SCORES = ("planned", "ghosh_weights")


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


cluster_of = defaultdict(set)
for r in rows_of(signature):
    cluster_of[r["gene"]].add(int(r["cluster"]))
groups = defaultdict(lambda: ([], []))
for r in rows_of(groups_file):
    groups[r["dataset"], r["comparison"]][int(r["group"])].append(r["sample"])
published = [r for r in rows_of(published_file) if r["panel"] in ("4C", "4D")]
observed_rows = [r for r in rows_of(auc_file) if r["reproduced_auc"]]
observed = {"planned": sum(r["verdict"] == "reproduced" for r in observed_rows)}
observed["ghosh_weights"] = sum(r["verdict_ghosh_weights"] == "reproduced" for r in observed_rows)

rng = np.random.default_rng(SEED)
total = {(m, s): np.zeros(DRAWS, dtype=int) for m in MODELS for s in SCORES}
each = defaultdict(list)
for gse in sorted({r["GSE"] for r in observed_rows}):
    _, samples, rows = read_series_matrix(Path(series) / gse / f"{gse}_series_matrix.txt.gz")
    names = list(rows)
    place = {name: i for i, name in enumerate(names)}
    table = np.array([[to_float(x) if to_float(x) is not None else np.nan for x in rows[n]] for n in names])
    background = np.nanpercentile(table, BACKGROUND_PERCENTILE, axis=0)
    weights = np.nan_to_num(np.maximum(table - background, 0))
    features = defaultdict(list)
    for r in rows_of(Path(landing) / f"{gse}.landing.tsv"):
        if r["feature"] in place:
            features[r["protein"]].append(place[r["feature"]])
    z = {p: standardise(weights[v].mean(axis=0)) for p, v in features.items()}
    z = {p: v for p, v in z.items() if v is not None}
    proteins = list(z)
    labels = [cluster_of[p] for p in proteins]
    where = {s: i for i, s in enumerate(samples)}
    tests = []
    for r in published:
        if r["dataset"] == gse:
            pre, post = groups[gse, r["comparison"]]
            tests.append((r["comparison"], [where[s] for s in pre], [where[s] for s in post], float(r["auc"])))
    for draw in range(DRAWS):
        random_z = {}
        for p, v in features.items():
            scored = standardise(weights[rng.integers(0, len(names), size=len(v))].mean(axis=0))
            if scored is not None:
                random_z[p] = scored
        shuffled = dict(zip(proteins, (labels[i] for i in rng.permutation(len(labels)))))
        for model, values in zip(
            MODELS,
            (scores(random_z, cluster_of, gse in M1_ONLY), scores(z, shuffled, gse in M1_ONLY)),
        ):
            for name, score in zip(SCORES, values):
                for comparison, pre, post, printed in tests:
                    ok = passes(auc_and_p(score[pre], score[post])[0], printed)
                    total[model, name][draw] += ok
                    each[gse, comparison, model, name].append(ok)

with open(f"{out}/chance_test.tsv", "w") as f:
    f.write("model\tscore\tobserved_passing\tcomparisons\tmean_passing_by_chance\tshare_of_draws_at_least_observed\n")
    for (model, name), passing in total.items():
        share = float((passing >= observed[name]).mean())
        f.write(f"{model}\t{name}\t{observed[name]}\t{len(observed_rows)}\t{passing.mean():.2f}\t{share:.3f}\n")
with open(f"{out}/chance_by_comparison.tsv", "w") as f:
    f.write("gse\tcomparison\tmodel\tscore\tshare_of_draws_passing\n")
    for (gse, comparison, model, name), passing in sorted(each.items()):
        f.write(f"{gse}\t{comparison}\t{model}\t{name}\t{np.mean(passing):.3f}\n")
