#!/usr/bin/env python3
"""Compare this run's counts with the paper's, through one counter run on both gene sets.

Usage: compare_counts.py <tables_dir> <samples_paper.csv> <out_dir>

tables_dir holds the five count tables and the gene match listed in README.md, under these names:
paper_counts.tsv, salmon.merged.gene_counts.tsv, rsem.merged.gene_counts.tsv, genefree_old.tsv
(no-genome counts on the paper's genes), genefree_new.tsv (on this run's genes), g_vs_fun.m8.
Plan and reading rule, written before this ran: README.md.

Writes counts_per_library.tsv (one row per pair of tables and library), change_per_comparison.tsv
(one row per pair, comparison and kind of gene pairing) and summary.tsv (medians and the rule).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_paper_genes import gene_map, strong_match  # noqa: E402

tables, samples_csv, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
MIN_COUNT_RHO, MIN_CHANGE_RHO, MIN_READS = 0.8, 0.6, 10
COMPARISONS = {  # the paper's four comparisons: windows of samples_paper.csv pooled as the paper did
    "2 h": ["young_2h"],
    "24 h": ["young_24h"],
    "13-49 d": ["young_13_17d", "young_49_52d"],
    "132 d": ["young_104_132d"],
}

design = pd.read_csv(samples_csv)
srr_to_vial = dict(zip(design["sample"], design.vial))


def read(name, **kw):
    t = pd.read_csv(tables / name, sep="\t", index_col=0, **kw)
    return t.drop(columns=[c for c in ("gene_name",) if c in t.columns])


T = {
    "paper": read("paper_counts.tsv").loc[lambda t: t.index.astype(str).str.match(r"g\d+$")],  # see not_genes
    "salmon": read("salmon.merged.gene_counts.tsv").rename(columns=srr_to_vial),
    "rsem": read("rsem.merged.gene_counts.tsv"),
    "no-genome, paper's genes": read("genefree_old.tsv"),
    "no-genome, this run's genes": read("genefree_new.tsv"),
}
# The paper's table also holds HTSeq's summary rows (__no_feature, __not_aligned, ...) and a row Sum,
# which is the sum over genes. They are not genes and are left out above. Found 2026-11-11, after
# the plan: they show that the paper counted with HTSeq.
raw = read("paper_counts.tsv")
not_genes = raw[~raw.index.astype(str).str.match(r"g\d+$")]
libraries = [v for v in design.vial if all(v in t.columns for t in T.values())]
design = design[design.vial.isin(libraries)]
totals = pd.DataFrame(
    {
        "library": libraries,
        "comparison": [next(c for c, w in COMPARISONS.items() if x in w) for x in design.window],
        "treatment": design.treatment.to_numpy(),
        "trimmed_read_pairs": design.trimmed_pairs.to_numpy(),
        **{f"reads_in_genes: {k}": t[libraries].sum().to_numpy() for k, t in T.items()},
        **{f"paper_row {i}": not_genes.loc[i, libraries].to_numpy() for i in not_genes.index if isinstance(i, str)},
    }
)
totals["paper_over_salmon"] = totals["reads_in_genes: paper"] / totals["reads_in_genes: salmon"]
totals.to_csv(out / "reads_in_genes_per_library.tsv", sep="\t", index=False)
print(totals.groupby("comparison", sort=False).paper_over_salmon.describe().round(2))
print(len(libraries), "of the paper's", len(pd.read_csv(samples_csv)), "libraries are in every table")
T = {k: t[libraries] for k, t in T.items()}

match = (
    gene_map(str(tables / "g_vs_fun.m8"))
    .filter(strong_match())
    .to_pandas()
    .dropna(subset=["fun_gene"])
    .set_index("paper_gene")
    .fun_gene.astype(str)
)
per_fun = match.groupby(match).size()
kind = match.map(lambda f: "one g gene to one FUN gene" if per_fun[f] == 1 else "several g genes to one FUN gene")
print(len(match), "g genes matched to", match.nunique(), "FUN genes;", (per_fun == 1).sum(), "one to one")


def through_match(t):
    """A table on g genes, re-named to FUN genes: counts of g genes matched to one FUN gene are added."""
    t = t[t.index.isin(match.index)]
    return t.groupby(match.reindex(t.index).to_numpy()).sum()


fun_kind = kind.groupby(match).first()
PAIRS = {
    "A. paper against no-genome, on the paper's genes": (T["paper"], T["no-genome, paper's genes"], None),
    "B. salmon against no-genome, on this run's genes": (T["salmon"], T["no-genome, this run's genes"], None),
    "B2. salmon against rsem": (T["salmon"], T["rsem"], None),
    "C. no-genome on the paper's genes against no-genome on this run's genes": (
        through_match(T["no-genome, paper's genes"]),
        T["no-genome, this run's genes"],
        fun_kind,
    ),
    "D. paper against salmon": (through_match(T["paper"]), T["salmon"], fun_kind),
}


def change(t, treated, control):
    cpm = t / t.sum() * 1e6
    return np.log2((cpm[treated].mean(axis=1) + 1) / (cpm[control].mean(axis=1) + 1))


lib_rows, change_rows = [], []
for pair, (a, b, kinds) in PAIRS.items():
    genes = a.index.union(b.index) if kinds is None else a.index.intersection(b.index)
    a, b = a.reindex(genes).fillna(0), b.reindex(genes).fillna(0)
    for lib in libraries:
        seen = (a[lib] > 0) | (b[lib] > 0)
        lib_rows.append(
            dict(
                pair=pair,
                library=lib,
                genes=int(seen.sum()),
                rank_correlation=spearmanr(a[lib][seen], b[lib][seen]).statistic,
            )
        )
    for comparison, windows in COMPARISONS.items():
        d = design[design.window.isin(windows)]
        treated, control = list(d.vial[d.treatment == "treated"]), list(d.vial[d.treatment == "control"])
        libs = treated + control
        enough = (a[libs].sum(axis=1) >= MIN_READS) & (b[libs].sum(axis=1) >= MIN_READS)
        ca, cb = change(a, treated, control), change(b, treated, control)
        subsets = {"all genes": enough}
        if kinds is not None:
            for k in sorted(set(kinds)):
                subsets[k] = enough & (kinds.reindex(genes) == k).to_numpy()
        for subset, keep in subsets.items():
            change_rows.append(
                dict(
                    pair=pair,
                    comparison=comparison,
                    genes_compared=subset,
                    treated=len(treated),
                    control=len(control),
                    genes=int(keep.sum()),
                    rank_correlation_of_change=spearmanr(ca[keep], cb[keep]).statistic,
                    same_sign=float((np.sign(ca[keep]) == np.sign(cb[keep]))[(ca[keep] != 0) | (cb[keep] != 0)].mean()),
                )
            )
libs = pd.DataFrame(lib_rows)
changes = pd.DataFrame(change_rows)
libs.to_csv(out / "counts_per_library.tsv", sep="\t", index=False)
changes.to_csv(out / "change_per_comparison.tsv", sep="\t", index=False)

summary = (
    changes.groupby(["pair", "genes_compared"], sort=False)
    .agg(
        median_change_correlation=("rank_correlation_of_change", "median"),
        lowest=("rank_correlation_of_change", "min"),
        highest=("rank_correlation_of_change", "max"),
    )
    .reset_index()
)
summary["median_count_correlation"] = summary.pair.map(libs.groupby("pair").rank_correlation.median())
summary["agree"] = (summary.median_count_correlation >= MIN_COUNT_RHO) & (
    summary.median_change_correlation >= MIN_CHANGE_RHO
)
summary.to_csv(out / "summary.tsv", sep="\t", index=False)
with pd.option_context("display.width", 250, "display.max_colwidth", 80):
    print(summary.round(3).to_string(index=False))
    print(changes.round(3).to_string(index=False))
