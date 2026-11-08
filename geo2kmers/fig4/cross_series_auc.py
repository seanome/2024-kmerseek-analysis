"""Score one series' samples with k-mers chosen on another series on the same array.

Usage: cross_series_auc.py <train table> <train design> <scored table> <scored design> <series folder>
       <train dataset> <scored dataset> <scored comparison> <shuffles> <out prefix>
Tables and designs are kamrat_table.py's. The rules are in cross_series_plan.md. Writes <prefix>.cross.tsv, one row
per k-mer count, and .cross_samples.tsv and .cross_chance.tsv.
"""

import csv
import sys

import numpy as np

from geo2kmers import held_out as ho
from geo2kmers.people import person_of, sample_fields

SIZES = (100, 1000, 10000)
HEADLINE = 1000
(
    train_table,
    train_design,
    scored_table,
    scored_design,
    series,
    train_gse,
    scored_gse,
    comparison,
    shuffles,
    prefix,
) = sys.argv[1:]
shuffles = int(shuffles)


def read(table, design):
    with open(design) as f:
        labels = dict(line.split() for line in f)
    kmers, lines = [], []
    with open(table) as f:
        samples = f.readline().rstrip("\n").split("\t")[1:]
        for line in f:
            kmer, rest = line.split("\t", 1)
            kmers.append(kmer)
            lines.append(np.fromstring(rest, sep="\t", dtype=np.float32))
    values = np.log2(np.array(lines, dtype=np.float64) + 1)
    return kmers, samples, values, np.array([labels[s] == "post" for s in samples])


train_kmers, _, train_values, train_post = read(train_table, train_design)
scored_kmers, samples, scored_values, post = read(scored_table, scored_design)
where = {k: i for i, k in enumerate(scored_kmers)}
shared = [(i, where[k]) for i, k in enumerate(train_kmers) if k in where]
train_values = train_values[[i for i, _ in shared]]
scored_values = scored_values[[j for _, j in shared]]
# A k-mer with the same value in every sample of either series cannot be tested or standardised
varies = (train_values.std(axis=1) > 0) & (scored_values.std(axis=1) > 0)
constant = int((~varies).sum())
train_values, scored_values = train_values[varies], scored_values[varies]
fields = sample_fields(series, scored_gse)
people = np.array([person_of(scored_gse, fields[s]) for s in samples])

p, difference = ho.welch_p(train_values, train_values * train_values, ~train_post, train_post)
rows = np.lexsort((np.arange(len(p)), p))[: max(SIZES)]
signs = np.sign(difference)
everyone = np.ones(len(post), dtype=bool)
scores = ho.score_samples(scored_values, rows, signs[rows], everyone, everyone, SIZES)

rng = np.random.default_rng(1)
swaps = ho.arrangements(post, people, shuffles, rng)
up, down = np.flatnonzero(signs > 0), np.flatnonzero(signs <= 0)


def random_set(size):
    """Random k-mers with as many going up in training as among the chosen ones, so a shift of the whole scored
    series favours them as much as the chosen k-mers."""
    n_up = int((signs[rows[:size]] > 0).sum())
    picked = np.concatenate([rng.choice(up, n_up, replace=False), rng.choice(down, size - n_up, replace=False)])
    return rng.permutation(picked)


random_sets = [{size: random_set(min(size, len(p))) for size in SIZES} for _ in range(shuffles)]
random_scores = [
    {
        size: ho.score_samples(scored_values, r[size], signs[r[size]], everyone, everyone, (size,))[size]
        for size in SIZES
    }
    for r in random_sets
]

summary, chance = [], []
for size in SIZES:
    auc = ho.auc_of(scores[size], post)
    above, n_people = ho.people_post_above_pre(scores[size], post, people)
    by_swap = [(ho.auc_of(scores[size], s), ho.people_post_above_pre(scores[size], s, people)[0]) for s in swaps]
    by_kmers = [(ho.auc_of(r[size], post), ho.people_post_above_pre(r[size], post, people)[0]) for r in random_scores]
    for kind, values in (("labels swapped", by_swap), ("random k-mers", by_kmers)):
        chance += [
            {"kmers": size, "chance": kind, "draw": i, "auc": f"{a:.3f}", "people_post_above_pre": u}
            for i, (a, u) in enumerate(values)
        ]
    summary.append(
        {
            "trained_on": train_gse,
            "scored": scored_gse,
            "comparison": comparison,
            "kmers": size,
            "headline": size == HEADLINE,
            "auc": f"{auc:.3f}",
            "p_labels_swapped": f"{ho.p_from_shuffles(auc, [a for a, _ in by_swap]):.4f}",
            "swaps": len(by_swap),
            "median_auc_labels_swapped": f"{np.median([a for a, _ in by_swap]):.3f}",
            "p_random_kmers": f"{ho.p_from_shuffles(auc, [a for a, _ in by_kmers]):.4f}",
            "random_sets": len(by_kmers),
            "median_auc_random_kmers": f"{np.median([a for a, _ in by_kmers]):.3f}",
            "people_post_above_pre": above,
            "people": n_people,
            "people_p_labels_swapped": f"{ho.p_from_shuffles(above, [u for _, u in by_swap]):.4f}",
            "people_p_random_kmers": f"{ho.p_from_shuffles(above, [u for _, u in by_kmers]):.4f}",
            "kmers_shared": len(shared),
            "kmers_dropped_for_no_variance": constant,
            "kmers_up_after_in_training": int((signs[rows[:size]] > 0).sum()),
        }
    )


def write(name, rows_out):
    with open(f"{prefix}.{name}.tsv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)


write("cross", summary)
write("cross_chance", [{"trained_on": train_gse, "scored": scored_gse, **r} for r in chance])
write(
    "cross_samples",
    [
        {
            "trained_on": train_gse,
            "scored": scored_gse,
            "sample": s,
            "person": people[i],
            "group": "post" if post[i] else "pre",
            **{f"score_{size}": f"{scores[size][i]:.4f}" for size in SIZES},
        }
        for i, s in enumerate(samples)
    ],
)
