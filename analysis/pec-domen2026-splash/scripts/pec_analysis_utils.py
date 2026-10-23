"""Small statistical checks shared by the PEC anchor-count analysis."""

import numpy as np


def bh_over_fitted_anchors(pvalues):
    """BH q-values, retaining fitted rows with missing p-values in the family size.

    Missing p-values remain missing in the output. Their rows count toward the
    declared all-fitted-anchor multiplicity correction, as if their p-values were1.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or np.any(np.isinf(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be between0 and1 or missing")
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
