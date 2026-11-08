"""Choose k-mers without one person, and score that person's samples with them."""

import itertools
import math

import numpy as np
from scipy.special import stdtr

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


def score_samples(values, rows, signs, reference, targets, sizes):
    """Mean signed standardised value of targets, one array per size, standardised on the reference columns."""
    reference_values = values[rows][:, reference]
    standardised = (values[rows][:, targets] - reference_values.mean(axis=1, keepdims=True)) / reference_values.std(
        axis=1, ddof=1, keepdims=True
    )
    signed = signs[:, None] * standardised
    return {size: signed[:size].mean(axis=0) for size in sizes}


def held_out_scores(values, squares, post, people, sizes):
    """Each sample's score from k-mers chosen and standardised without that sample's person.

    Returns a score array per size, and each person's chosen rows at the largest size.
    """
    scores = {size: np.full(len(post), np.nan) for size in sizes}
    chosen = {}
    for person in np.unique(people):
        held = people == person
        kept = ~held
        rows, signs = choose(values, squares, kept & ~post, kept & post, max(sizes))
        for size, value in score_samples(values, rows, signs, kept, held, sizes).items():
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
