"""The Fig. 4 score: standardise each protein across a series, then join M1 and M2 proteins."""

import numpy as np
from scipy.stats import mannwhitneyu

RULE_DISTANCE = 0.15


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
    """(value - (threshold + 0.5)) / (3 x sd), or None if the values do not vary."""
    sd = values.std(ddof=1)
    return None if sd == 0 else (values - (stepminer_threshold(values) + 0.5)) / (3 * sd)


def scores(z, cluster_of, m1_only):
    """Return the planned score and the -1, 1, 2 score. z maps a protein to its standardised values."""
    size = len(next(iter(z.values())))
    zero = np.zeros(size)
    cluster = {c: [p for p in z if c in cluster_of[p]] for c in (13, 14, 3)}
    m2 = [p for p in z if cluster_of[p] & {14, 3}]
    mean_of = lambda proteins: np.mean([z[p] for p in proteins], axis=0) if proteins else zero
    sum_of = lambda proteins: np.sum([z[p] for p in proteins], axis=0) if proteins else zero
    if m1_only:
        return -mean_of(cluster[13]), -sum_of(cluster[13])
    return -mean_of(cluster[13]) + mean_of(m2), -sum_of(cluster[13]) + sum_of(cluster[14]) + 2 * sum_of(cluster[3])


def auc_and_p(pre, post):
    """The chance a post sample scores above a pre sample, ties counting a half, and Mann-Whitney's two-sided p."""
    u, p = mannwhitneyu(post, pre, alternative="two-sided")
    return u / (len(pre) * len(post)), p


def passes(auc, published):
    """The rule fixed before running: on the same side of 0.5 as the published AUC, and within 0.15 of it."""
    return round(abs(auc - published), 10) <= RULE_DISTANCE and (auc > 0.5) == (published > 0.5)
