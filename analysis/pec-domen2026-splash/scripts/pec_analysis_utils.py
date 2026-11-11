"""Small statistical helpers shared by the PEC analysis scripts (notebook 361)."""

import numpy as np


def bh_over_fitted_anchors(pvalues):
    """BH q-values, retaining fitted rows with missing p-values in the family size.

    Missing p-values remain missing in the output. Their rows count toward the
    declared all-fitted-anchor multiplicity correction, as if their p-values were 1.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or np.any(np.isinf(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be between 0 and 1 or missing")
    valid = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if not valid.any():
        return q
    order = np.argsort(p[valid], kind="stable")
    ranked = p[valid][order] * len(p) / np.arange(1, valid.sum() + 1)
    adjusted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1)
    restored = np.empty(valid.sum())
    restored[order] = adjusted
    q[valid] = restored
    return q


def poscounts_size_factors(counts, as_pydeseq2=False):
    """DESeq2's `poscounts` size factor of each sample; counts is genes (rows) by samples (columns).

    A gene's reference is the sum of log(count) over the samples where it has a read, divided by the
    number of all samples. A sample's factor is the median, over its genes with a read, of
    log(count) minus the reference, taken back from the log. Factors are then divided by their
    geometric mean. Plan: design/sample_scaling/README.md.

    This follows R DESeq2's estimateSizeFactorsForMatrix with poscounts geometric means. PyDESeq2
    0.5.4 also leaves out every gene whose reference is 0, which is a gene whose only counts are
    single reads; as_pydeseq2=True does the same, and is there only to check this function against it.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2 or np.isnan(c).any() or (c < 0).any():
        raise ValueError("counts must be a genes x samples table of numbers that are zero or more")
    positive = c > 0
    if not positive.any(axis=0).all():
        raise ValueError("a sample has no reads, so it has no size factor")
    logc = np.where(positive, np.log(np.where(positive, c, 1.0)), 0.0)
    reference = logc.sum(axis=1) / c.shape[1]
    use = positive & (reference > 0)[:, None] if as_pydeseq2 else positive
    factors = np.array([np.exp(np.median((logc[:, j] - reference)[use[:, j]])) for j in range(c.shape[1])])
    return factors / np.exp(np.log(factors).mean())


def scaled_per_million(counts):
    """Counts divided by each sample's poscounts factor, times one constant for the whole table that
    makes the mean sample sum to one million. Only the scaling between samples differs from CPM."""
    c = np.asarray(counts, dtype=float)
    scaled = c / poscounts_size_factors(c)
    return scaled * (1e6 / scaled.sum(axis=0).mean())
