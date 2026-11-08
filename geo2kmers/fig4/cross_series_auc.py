"""Score one series' samples with k-mers chosen on another series on the same array.

Usage: cross_series_auc.py <train table> <train design> <scored table> <scored design> <series folder>
       <train dataset> <scored dataset> <scored comparison> <shuffles> <out prefix>
Tables and designs are kamrat_table.py's, or the tables are hp_value_table.py's for one array. The rules are in
cross_series_plan.md, and for HP k-mers also in hp_tests_plan.md. Writes <prefix>.cross.tsv, one row
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


train_kmers, _, train_values, train_post, train_weights, probe_sets = ho.read_table(train_table, train_design)
scored_kmers, samples, scored_values, post, _, _ = ho.read_table(scored_table, scored_design)
where = {k: i for i, k in enumerate(scored_kmers)}
shared = [(i, where[k]) for i, k in enumerate(train_kmers) if k in where]
train_values = train_values[[i for i, _ in shared]]
scored_values = scored_values[[j for _, j in shared]]
# A k-mer with the same value in every sample of either series cannot be tested or standardised
varies = (train_values.std(axis=1) > 0) & (scored_values.std(axis=1) > 0)
constant = int((~varies).sum())
train_values, scored_values = train_values[varies], scored_values[varies]
if train_weights is None:
    weights = probe_sets = None
else:
    keep = np.array([i for i, _ in shared])[varies]
    weights, probe_sets = train_weights[keep], [probe_sets[i] for i in keep]
fields = sample_fields(series, scored_gse)
people = np.array([person_of(scored_gse, fields[s]) for s in samples])

p, difference = ho.welch_p(train_values, train_values * train_values, ~train_post, train_post)
rows = np.lexsort((np.arange(len(p)), p))[: max(SIZES)]
signs = np.sign(difference)
everyone = np.ones(len(post), dtype=bool)
scores = ho.score_samples(scored_values, rows, signs[rows], everyone, everyone, SIZES, weights)
# An HP row stands for weights[row] k-mers; with plain k-mers each row is one
per_row = np.ones(len(p), dtype=np.int64) if weights is None else weights

rng = np.random.default_rng(1)
swaps = ho.arrangements(post, people, shuffles, rng)
up, down = np.flatnonzero(signs > 0), np.flatnonzero(signs <= 0)


def chosen_up(size):
    """How many of the first size chosen k-mers went up after exercise in training."""
    used = ho.first_kmers(per_row[rows], size)
    return int(used[signs[rows] > 0].sum())


def random_set(size):
    """Random k-mers with as many going up in training as among the chosen ones, so a shift of the whole scored
    series favours them as much as the chosen k-mers."""
    n_up = int((signs[rows[:size]] > 0).sum())
    picked = np.concatenate([rng.choice(up, n_up, replace=False), rng.choice(down, size - n_up, replace=False)])
    return rng.permutation(picked)


def draw(pool, n):
    """n k-mers drawn without replacement from the HP rows in pool, each k-mer equally likely, as one weight per
    row of the table."""
    ends = np.cumsum(per_row[pool])
    picked = np.searchsorted(ends, rng.choice(int(ends[-1]), n, replace=False), side="right")
    counts = np.zeros(len(p), dtype=np.int64)
    np.add.at(counts, pool[picked], 1)
    return counts


def random_hp_score(size):
    """As random_set, for HP rows that each stand for several k-mers."""
    size = min(size, int(per_row.sum()))
    n_up = chosen_up(size)
    counts = draw(up, n_up) + draw(down, size - n_up)
    picked = np.flatnonzero(counts)
    return ho.score_samples(scored_values, picked, signs[picked], everyone, everyone, (size,), counts)[size]


if weights is None:
    random_sets = [{size: random_set(min(size, len(p))) for size in SIZES} for _ in range(shuffles)]
    random_scores = [
        {
            size: ho.score_samples(scored_values, r[size], signs[r[size]], everyone, everyone, (size,))[size]
            for size in SIZES
        }
        for r in random_sets
    ]
else:
    random_scores = [{size: random_hp_score(size) for size in SIZES} for _ in range(shuffles)]

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
            "kmers_up_after_in_training": chosen_up(size),
        }
    )
    if weights is not None:
        used = ho.first_kmers(weights[rows], size)
        summary[-1].update(
            kmers_shared=int(train_weights[[i for i, _ in shared]].sum()),
            kmers_dropped_for_no_variance=int(train_weights[[i for i, _ in shared]].sum() - weights.sum()),
            probe_sets_behind=len(
                {n for r, u in zip(rows, used, strict=True) if u > 0 for n in probe_sets[r].split(",")}
            ),
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
