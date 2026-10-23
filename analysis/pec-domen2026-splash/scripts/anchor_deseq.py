#!/usr/bin/env python3
"""PyDESeq2 on anchor counts (the abundance channel), as fixed in design/ANALYSIS_PLAN.md.

Usage: anchor_deseq.py <anchor_counts.tsv> <samples.txt> <design.tsv> <out_dir> [n_cpus]

anchor_counts.tsv: no header; columns anchor, n_targets, then one count column per sample in
the order of samples.txt (output of `pec-tools anchor-counts`).
design.tsv: header with sample, condition (control|PEC), pair, genotype.

Fits three designs: ~ pair + condition (primary), ~ condition, ~ genotype + condition (skipped
when genotype is constant or confounded). For each: Wald test PEC vs control, BH over all fitted
anchors. Writes one parquet per design and a JSON summary with counts at the plan's thresholds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pec_analysis_utils import bh_over_fitted_anchors
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

counts_path, samples_path, design_path, out_dir = sys.argv[1:5]
n_cpus = int(sys.argv[5]) if len(sys.argv) > 5 else 8
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
samples = open(samples_path).read().split()
if len(set(samples)) != len(samples) or not samples:
    raise ValueError("sample names must be unique and nonempty")
design = pd.read_csv(design_path, sep="\t", dtype=str).set_index("sample").loc[samples]
counts = pl.read_csv(
    counts_path,
    separator="\t",
    has_header=False,
    new_columns=["anchor", "n_targets", *samples],
)
if counts["anchor"].n_unique() != counts.height:
    raise ValueError("anchor rows must be unique across SATC bins")
if set(design["condition"]) != {"control", "PEC"}:
    raise ValueError("both control and PEC conditions are required")
if design.groupby("pair")["genotype"].nunique().max() > 1:
    raise ValueError("each pair must belong to one genotype")
n_tested = counts.height
present = counts.select(
    (pl.concat_list(samples).list.eval(pl.element() > 0).list.sum()).alias("n_present")
)
keep = present["n_present"] >= int(np.ceil(len(samples) / 2))
counts = counts.filter(keep)
X = counts.select(samples).to_numpy().T.astype(np.int64)
anchors = counts["anchor"].to_list()
counts_df = pd.DataFrame(X, index=samples, columns=anchors)
summary = dict(
    anchors_tested_by_splash=n_tested,
    anchors_fitted=len(anchors),
    filter="count > 0 in at least half the samples",
    samples=samples,
    designs={},
)

designs = {"pair_condition": ["pair", "condition"], "condition": ["condition"]}
if (
    design["genotype"].nunique() > 1
    and design["genotype"].nunique() < design["pair"].nunique()
):
    designs["genotype_condition"] = ["genotype", "condition"]
elif design["genotype"].nunique() == design["pair"].nunique():
    summary["genotype_condition_note"] = (
        "genotype identical to pair here; same fit as pair_condition"
    )
else:
    summary["genotype_condition_note"] = "one genotype only; not estimable"

inference = DefaultInference(n_cpus=n_cpus)
for name, factors in designs.items():
    meta = design[factors].copy()
    formula = "~ " + " + ".join(factors)
    # a design is fitted only if its model matrix has full column rank and leaves residual df
    mm = pd.get_dummies(meta, drop_first=True).astype(float)
    mm.insert(0, "intercept", 1.0)
    rank = np.linalg.matrix_rank(mm.to_numpy())
    if rank < mm.shape[1] or rank >= len(samples):
        summary["designs"][name] = dict(formula=formula, fitted=False,
                                        reason=f"model matrix rank {rank}, columns {mm.shape[1]}, samples {len(samples)}")
        print(name, "not estimable", summary["designs"][name]["reason"])
        continue
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=meta,
        design=formula,
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", "PEC", "control"], inference=inference)
    st.summary()
    res = st.results_df.copy()
    res.index.name = "anchor"
    # BH over all fitted anchors; pydeseq2's padj applies independent filtering, so recompute plainly too
    p = res["pvalue"].to_numpy()
    q = bh_over_fitted_anchors(p)
    res["q_bh_all"] = q
    res = res.reset_index()
    pl.from_pandas(res).write_parquet(out / f"deseq_{name}.parquet")
    lfc = res["log2FoldChange"].abs() > 1
    summary["designs"][name] = dict(
        formula=formula,
        q_le_005_lfc_gt1=int(((res["q_bh_all"] <= 0.05) & lfc).sum()),
        q_le_005_lfc_gt1_up=int(
            ((res["q_bh_all"] <= 0.05) & (res["log2FoldChange"] > 1)).sum()
        ),
        q_le_005_lfc_gt1_down=int(
            ((res["q_bh_all"] <= 0.05) & (res["log2FoldChange"] < -1)).sum()
        ),
        nominal_p_le_005_lfc_gt1=int(((res["pvalue"] <= 0.05) & lfc).sum()),
        nominal_up=int(((res["pvalue"] <= 0.05) & (res["log2FoldChange"] > 1)).sum()),
        nominal_down=int(
            ((res["pvalue"] <= 0.05) & (res["log2FoldChange"] < -1)).sum()
        ),
        pvalue_nan=int(np.isnan(p).sum()),
        size_factors=dict(
            zip(
                samples,
                map(
                    float,
                    np.asarray(
                        dds.obs["size_factors"]
                        if "size_factors" in dds.obs
                        else dds.obsm["size_factors"]
                    ),
                ),
            )
        ),
    )
    print(name, json.dumps(summary["designs"][name]))
(out / "deseq_summary.json").write_text(json.dumps(summary, indent=1))
