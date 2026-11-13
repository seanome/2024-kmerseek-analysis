"""Score each person's samples with k-mers chosen without that person, against labels swapped within people.

Usage: held_out_auc.py <table.tsv> <design.tsv> <series folder> <dataset> <comparison> <cpus> <shuffles> <out prefix>
table.tsv and design.tsv are kamrat_table.py's, or table.tsv is hp_value_table.py's. The rules are in
held_out_plan.md, and for HP k-mers also in hp_tests_plan.md. Writes <prefix>.held_out.tsv,
one row per k-mer count, and .held_out_samples.tsv, .held_out_shuffles.tsv and .held_out_kmers.tsv.
"""

import csv
import multiprocessing
import sys

import numpy as np

from geo2kmers import held_out as ho
from geo2kmers.people import person_of, sample_fields

SIZES = (100, 1000, 10000)
HEADLINE = 1000
table, design, series, dataset, comparison, cpus, shuffles, prefix = sys.argv[1:]
cpus, shuffles = int(cpus), int(shuffles)

kmers, samples, VALUES, POST, WEIGHTS, PROBE_SETS = ho.read_table(table, design)
SQUARES = VALUES * VALUES
fields = sample_fields(series, dataset)
PEOPLE = np.array([person_of(dataset, fields[s]) for s in samples])


def one_arrangement(post):
    """The AUC and the people count at each k-mer count, under one way of placing the labels."""
    scores = ho.held_out_scores(VALUES, SQUARES, post, PEOPLE, SIZES, WEIGHTS)[0]
    return {s: (ho.auc_of(scores[s], post), ho.people_post_above_pre(scores[s], post, PEOPLE)[0]) for s in SIZES}


scores, chosen = ho.held_out_scores(VALUES, SQUARES, POST, PEOPLE, SIZES, WEIGHTS)
everyone = np.ones(len(POST), dtype=bool)
rows, signs = ho.choose(VALUES, SQUARES, ~POST, POST, max(SIZES))
same_people = ho.score_samples(VALUES, rows, signs, everyone, everyone, SIZES, WEIGHTS)
all_p = ho.welch_p(VALUES, SQUARES, ~POST, POST)[0]
# An HP row is tested once but stands for WEIGHTS k-mers, each one test
below = ho.count_below_fdr(all_p if WEIGHTS is None else np.repeat(all_p, WEIGHTS))
other = ho.arrangements(POST, PEOPLE, shuffles, np.random.default_rng(1))
with multiprocessing.get_context("fork").Pool(cpus) as pool:
    swapped = pool.map(one_arrangement, other, chunksize=1)


def write(name, rows_out):
    with open(f"{prefix}.{name}.tsv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)


def probe_sets_behind(rows, size):
    """The probe sets whose proteins hold the first size k-mers of these rows."""
    used = ho.first_kmers(WEIGHTS[rows], size)
    return {name for row, n in zip(rows, used, strict=True) if n > 0 for name in PROBE_SETS[row].split(",")}


summary = []
for size in SIZES:
    auc = ho.auc_of(scores[size], POST)
    above, people = ho.people_post_above_pre(scores[size], POST, PEOPLE)
    null_auc = [s[size][0] for s in swapped]
    null_above = [s[size][1] for s in swapped]
    summary.append(
        {
            "dataset": dataset,
            "comparison": comparison,
            "kmers": size,
            "headline": size == HEADLINE,
            "held_out_auc": f"{auc:.3f}",
            "p": f"{ho.p_from_shuffles(auc, null_auc):.4f}",
            "swaps": len(null_auc),
            "median_auc_swapped": f"{np.median(null_auc):.3f}",
            "highest_auc_swapped": f"{max(null_auc):.3f}",
            "people_post_above_pre": above,
            "people": people,
            "people_p": f"{ho.p_from_shuffles(above, null_above):.4f}",
            "auc_kmers_chosen_from_everyone": f"{ho.auc_of(same_people[size], POST):.3f}",
            "kmers_tested": len(kmers),
            "kmers_below_fdr_all_samples": below,
        }
    )
    if WEIGHTS is not None:
        # An HP row stands for several k-mers; count k-mers, and the probe sets the chosen ones come from
        behind = [len(probe_sets_behind(person_rows, size)) for person_rows, _ in chosen.values()]
        summary[-1].update(
            kmers_tested=int(WEIGHTS.sum()), fewest_probe_sets_behind=min(behind), most_probe_sets_behind=max(behind)
        )
write("held_out", summary)
write(
    "held_out_samples",
    [
        {
            "dataset": dataset,
            "comparison": comparison,
            "sample": s,
            "person": PEOPLE[i],
            "group": "post" if POST[i] else "pre",
            **{f"score_{size}": f"{scores[size][i]:.4f}" for size in SIZES},
        }
        for i, s in enumerate(samples)
    ],
)
write(
    "held_out_shuffles",
    [
        {
            "dataset": dataset,
            "comparison": comparison,
            "swap": i,
            "kmers": size,
            "auc": f"{s[size][0]:.3f}",
            "people_post_above_pre": s[size][1],
        }
        for i, s in enumerate(swapped)
        for size in SIZES
    ],
)
folds = {}
for person_rows, person_signs in chosen.values():
    if WEIGHTS is not None:
        keep = ho.first_kmers(WEIGHTS[person_rows], HEADLINE) > 0
        person_rows, person_signs = person_rows[keep], person_signs[keep]
    for row, sign in zip(person_rows[:HEADLINE], person_signs[:HEADLINE], strict=True):
        count, direction = folds.get(row, (0, 0))
        folds[row] = (count + 1, direction + sign)
write(
    "held_out_kmers",
    [
        {
            "kmer": kmers[row],
            "people_left_out_when_chosen": count,
            "sum_of_directions": int(direction),
            **({} if WEIGHTS is None else {"kmers": int(WEIGHTS[row]), "probe_sets": PROBE_SETS[row]}),
        }
        for row, (count, direction) in sorted(folds.items(), key=lambda item: -item[1][0])
    ],
)
