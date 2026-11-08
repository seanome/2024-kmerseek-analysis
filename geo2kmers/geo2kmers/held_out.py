"""Choose k-mers without one person, and score that person's samples with them."""

import gzip
import itertools
import math

import numpy as np
from scipy.special import stdtr

from geo2kmers.geo import Geo2KmersError
from geo2kmers.scoring import auc_and_p


def group_sums(values, squares, member):
    """Each row's sum and sum of squares over the chosen columns."""
    weights = member.astype(float)
    return values @ weights, squares @ weights


def welch_p(values, squares, pre, post):
    """Each row's two-sided Welch t-test p, as KaMRaT computes it, and post mean minus pre mean.

    values is log2(intensity + 1) and squares holds the same values squared, so no column is copied. A row with no
    variance in either group gets p = 1, as in KaMRaT.
    """
    n1, n2 = pre.sum(), post.sum()
    if n1 < 2 or n2 < 2:
        raise Geo2KmersError(f"a Welch test needs at least 2 samples in each group, got {n1} pre and {n2} post")
    sum1, square1 = group_sums(values, squares, pre)
    sum2, square2 = group_sums(values, squares, post)
    mean1, mean2 = sum1 / n1, sum2 / n2
    t1 = np.maximum(square1 - sum1 * sum1 / n1, 0) / (n1 - 1) / n1
    t2 = np.maximum(square2 - sum2 * sum2 / n2, 0) / (n2 - 1) / n2
    both = t1 + t2
    usable = both > 0
    p = np.ones(len(values))
    df = both[usable] ** 2 / (t1[usable] ** 2 / (n1 - 1) + t2[usable] ** 2 / (n2 - 1))
    p[usable] = 2 * stdtr(df, -np.abs((mean1[usable] - mean2[usable]) / np.sqrt(both[usable])))
    return p, mean2 - mean1


def count_below_fdr(p, fdr=0.05):
    """How many rows pass Benjamini-Hochberg at this false discovery rate."""
    ordered = np.sort(p)
    q = np.minimum.accumulate((ordered * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return int((q < fdr).sum())


def choose(values, squares, pre, post, size):
    """The rows with the lowest p, best first, and +1 or -1 for up or down after exercise."""
    p, difference = welch_p(values, squares, pre, post)
    rows = np.lexsort((np.arange(len(p)), p))[:size]  # ties in p go to the lower row, on any numpy version
    return rows, np.sign(difference[rows])


def first_kmers(weights, size):
    """How many of each row to use so that the rows, in order, give size k-mers: each row is weights[i] k-mers with
    the same values, and the row that reaches size is cut short."""
    used = np.asarray(weights, dtype=float).copy()
    total = np.cumsum(used)
    if total[-1] > size:
        last = int(np.searchsorted(total, size))
        used[last] -= total[last] - size
        used[last + 1 :] = 0
    return used


def score_samples(values, rows, signs, reference, targets, sizes, weights=None):
    """Mean signed standardised value of targets, one array per size, standardised on the reference columns.

    weights, if given, is how many k-mers each row stands for, and a size counts k-mers rather than rows.
    """
    reference_values = values[rows][:, reference]
    standardised = (values[rows][:, targets] - reference_values.mean(axis=1, keepdims=True)) / reference_values.std(
        axis=1, ddof=1, keepdims=True
    )
    signed = signs[:, None] * standardised
    if weights is None:
        return {size: signed[:size].mean(axis=0) for size in sizes}
    out = {}
    for size in sizes:
        used = first_kmers(weights[rows], size)
        out[size] = used @ signed / used.sum()
    return out


def held_out_scores(values, squares, post, people, sizes, weights=None):
    """Each sample's score from k-mers chosen and standardised without that sample's person.

    Returns a score array per size, and each person's chosen rows at the largest size. weights is as in
    score_samples.
    """
    scores = {size: np.full(len(post), np.nan) for size in sizes}
    chosen = {}
    for person in np.unique(people):
        held = people == person
        kept = ~held
        rows, signs = choose(values, squares, kept & ~post, kept & post, max(sizes))
        for size, value in score_samples(values, rows, signs, kept, held, sizes, weights).items():
            scores[size][held] = value
        chosen[person] = (rows, signs)
    return scores, chosen


def auc_of(scores, post):
    """The chance a post sample scores above a pre sample, ties counting a half."""
    return auc_and_p(scores[~post], scores[post])[0]


def people_post_above_pre(scores, post, people):
    """How many people have a mean post score above their mean pre score, among people with both."""
    above = total = 0
    for person in np.unique(people):
        before, after = scores[(people == person) & ~post], scores[(people == person) & post]
        if len(before) and len(after):
            total += 1
            above += after.mean() > before.mean()
    return above, total


def arrangements(post, people, count, rng):
    """Other ways to place the labels within each person: all of them if at most count exist, else count at random."""
    groups = [np.flatnonzero(people == person) for person in np.unique(people)]
    posts = [int(post[g].sum()) for g in groups]
    total = math.prod(math.comb(len(g), n) for g, n in zip(groups, posts, strict=True))
    if total - 1 > count:
        drawn = []
        for _ in range(count):
            label = np.zeros(len(post), dtype=bool)
            for g, n in zip(groups, posts, strict=True):
                label[rng.choice(g, n, replace=False)] = True
            drawn.append(label)
        return drawn
    every = []
    per_person = [list(itertools.combinations(g, n)) for g, n in zip(groups, posts, strict=True)]
    for choice in itertools.product(*per_person):
        label = np.zeros(len(post), dtype=bool)
        label[[i for part in choice for i in part]] = True
        if not np.array_equal(label, post):
            every.append(label)
    return every


def p_from_shuffles(observed, shuffled):
    """One-sided p: the share of shuffles at or above the true value, counting the true value itself."""
    shuffled = np.asarray(shuffled)
    return (1 + int((shuffled >= observed).sum())) / (1 + len(shuffled))


def read_table(table, design):
    """A k-mer table and its design: names, samples, values, post labels, and for an HP table the k-mers and probe
    sets each row stands for.

    A KaMRaT table (kamrat_table.py) holds intensities, which are logged as log2(intensity + 1); weights and probe
    sets are then None. An HP table (hp_value_table.py) has `kmers` and `probe_sets` as its second and third columns
    and holds RMA values, already log2, used as they are."""
    with open(design) as f:
        labels = dict(line.split() for line in f)
    names, lines, weights, probe_sets = [], [], [], []
    opener = gzip.open if str(table).endswith(".gz") else open
    with opener(table, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        hp = header[1:3] == ["kmers", "probe_sets"]
        samples = header[3:] if hp else header[1:]
        for line in f:
            if hp:
                name, kmers, sets, rest = line.split("\t", 3)
                weights.append(int(kmers))
                probe_sets.append(sets)
            else:
                name, rest = line.split("\t", 1)
            names.append(name)
            lines.append(np.fromstring(rest, sep="\t", dtype=np.float32))
    values = np.array(lines, dtype=np.float64).reshape(len(lines), len(samples))
    # A series matrix can hold samples outside the comparison; keep the design's
    keep = [i for i, s in enumerate(samples) if s in labels]
    samples, values = [samples[i] for i in keep], values[:, keep]
    post = np.array([labels[s] == "post" for s in samples])
    if not hp:
        return names, samples, np.log2(values + 1), post, None, None
    return names, samples, values, post, np.array(weights), probe_sets
