#!/usr/bin/env python3
"""Each sample's total and poscounts size factor per contrast, checked against PyDESeq2 (notebook 361).

Usage: sample_scaling_factors.py <all.counts.tsv> <design_dir> <out.tsv>

For each Botryllus contrast (design_dir/contrast_<name>.design.tsv), over that contrast's samples:
reads placed in a gene, the poscounts size factor from pec_analysis_utils, and the same factor from
PyDESeq2 (size_factors_fit_type="poscounts"). Stops if the two differ by more than 1 part in a
million. total_over_factor is the sample's total divided by its factor, each first divided by its
geometric mean over the contrast: above 1 means CPM scales this sample's genes down more than the
size factor does. Plan: design/sample_scaling/README.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pec_analysis_utils import poscounts_size_factors  # noqa: E402

counts_p, design_dir, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
counts = pd.read_csv(counts_p, sep="\t", index_col=0)
rows = []
for c in CONTRASTS:
    d = pd.read_csv(design_dir / f"contrast_{c}.design.tsv", sep="\t", dtype=str).set_index("sample")
    t = counts[list(d.index)]
    t = t[t.sum(axis=1) > 0]
    mine = poscounts_size_factors(t.to_numpy())
    dds = DeseqDataSet(
        counts=t.T.round().astype(int),
        metadata=d[["condition"]],
        design="~condition",
        size_factors_fit_type="poscounts",
        quiet=True,
    )
    dds.fit_size_factors()
    theirs = np.asarray(dds.obs["size_factors"] if "size_factors" in dds.obs else dds.obsm["size_factors"], float)
    assert np.allclose(mine, theirs, rtol=1e-6), (c, np.abs(mine / theirs - 1).max())
    total = t.sum().to_numpy(float)
    rel = total / np.exp(np.log(total).mean()) / mine
    for s, cond, n, f, p, r in zip(d.index, d.condition, total, mine, theirs, rel):
        rows.append(
            dict(
                contrast=c,
                sample=s,
                condition=cond,
                reads_in_a_gene=int(n),
                size_factor=f,
                size_factor_pydeseq2=p,
                total_over_factor=r,
            )
        )
    print(c, len(d), "samples, largest difference from PyDESeq2:", f"{np.abs(mine / theirs - 1).max():.2e}")
pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
