#!/usr/bin/env python3
"""Is the genome version, or the read counter, why this run misses the paper's genes?

Reads inputs/count_split_listed.tsv: one row per count table and per entry of the paper's
782 listed genes (Domen et al. 2026, Dataset S1, sheet "annotated DE all groups young").
Every table was fitted the same way, as the paper did: each of its four young-colony
comparisons alone, ~ treatment, DESeq2's default size factors, its 38 libraries
(PR 223's count_split.R, Sherlock job 51959161). A listed gene is "found" when the fit
gives p < 0.05 with the fold change in the paper's direction.

Tables compared here (strand ignored, as the paper's libraries are unstranded):
  paper_table                     the paper's own HTSeq counts, from Dataset S1
  paper_genome__featurecounts_s0  this run's reads on the paper's genome (botznik-chr.fa) and g genes,
                                  counted as HTSeq counts (read pairs on one gene only)
  paper_genome__salmon            the same alignments, counted by Salmon
  this_genome__featurecounts_s0   this run's reads on the April 2024 genome and FUN genes,
                                  counted as HTSeq counts; matched to g genes by protein
  this_genome__salmon             the same alignments, counted by Salmon

Writes genome_version_summary.tsv, genome_version_split.tsv and the figure
figures/genome_version_domen2026_dataset_s1.{svg,png}.
Usage: genome_version.py   (run from this folder)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

TABLES = {
    "paper_table": ("The paper's own table", "paper"),
    "paper_genome__featurecounts_s0": ("Paper's genome (botznik), g genes", "htseq"),
    "paper_genome__salmon": ("Paper's genome (botznik), g genes", "salmon"),
    "this_genome__featurecounts_s0": ("April 2024 genome, FUN genes", "htseq"),
    "this_genome__salmon": ("April 2024 genome, FUN genes", "salmon"),
}
OLD, NEW = "paper_genome__featurecounts_s0", "this_genome__featurecounts_s0"

x = pl.read_csv(
    HERE / "inputs/count_split_listed.tsv",
    separator="\t",
    infer_schema_length=None,
    null_values="NA",
)
x = x.filter(pl.col("table").is_in(list(TABLES)))
key = ["comparison", "paper_gene", "direction"]
n_entries = x.filter(pl.col("table") == "paper_table").height
assert n_entries == 782, n_entries
assert x.group_by("table").len()["len"].to_list() == [782] * len(TABLES)

# 1. Listed genes found on each table, and the paper's genes tested there.
summary = (
    x.group_by("table")
    .agg(found=pl.col("found").sum(), tested=pl.col("tested").sum())
    .with_columns(listed=pl.lit(782))
)
summary = pl.DataFrame({"table": list(TABLES)}).join(
    summary, on="table", maintain_order="left"
)
print(summary)
f = dict(zip(summary["table"], summary["found"]))
gap = f["paper_table"] - f[OLD]
print(
    f"paper table {f['paper_table']}; HTSeq-style: old genome {f[OLD]}, new genome {f[NEW]}"
)
print(
    f"genome version (old - new, HTSeq-style): {f[OLD] - f[NEW]}; "
    f"(old - new, Salmon): {f['paper_genome__salmon'] - f['this_genome__salmon']}"
)
print(
    f"counter on the old genome (HTSeq-style - Salmon): {f[OLD] - f['paper_genome__salmon']}; "
    f"on the new genome: {f[NEW] - f['this_genome__salmon']}"
)
print(f"left between the paper's table and the old genome, HTSeq-style: {gap}")

# 2. The paper's genes found on its own table, split by whether each genome finds them.
w = x.pivot(on="table", index=key, values="found")
paper_found = w.filter(pl.col("paper_table"))
split = (
    paper_found.with_columns(
        status=pl.when(pl.col(OLD) & pl.col(NEW))
        .then(pl.lit("both genomes"))
        .when(pl.col(OLD))
        .then(pl.lit("paper genome only"))
        .when(pl.col(NEW))
        .then(pl.lit("April 2024 genome only"))
        .otherwise(pl.lit("neither genome"))
    )
    .group_by("status")
    .len()
)
order = [
    "both genomes",
    "paper genome only",
    "April 2024 genome only",
    "neither genome",
]
split = (
    pl.DataFrame({"status": order})
    .join(split, on="status", how="left", maintain_order="left")
    .fill_null(0)
)
assert split["len"].sum() == paper_found.height
print(split)

# 3. How closely each genome's fit follows the paper's own fit, gene by gene: rank
#    correlation of signed -log10 p over the listed genes tested on both tables.
p = x.with_columns(s=pl.col("log2FoldChange").sign() * -pl.col("pvalue").log10()).pivot(
    on="table", index=key, values="s"
)
rho = {}
for t in TABLES:
    if t == "paper_table":
        continue
    both = p.select("paper_table", t).drop_nulls()
    r = stats.spearmanr(both["paper_table"], both[t]).statistic
    rho[t] = (r, both.height)
    print(f"{t}: Spearman with the paper's fit {r:.2f} over {both.height} listed genes")

summary = summary.with_columns(
    spearman_with_paper_fit=pl.col("table").replace_strict(
        {t: round(v[0], 3) for t, v in rho.items()}, default=None
    ),
    genes_correlated=pl.col("table").replace_strict(
        {t: v[1] for t, v in rho.items()}, default=None
    ),
)
summary.write_csv(HERE / "genome_version_summary.tsv", separator="\t")
split.rename({"len": "paper_found_entries"}).write_csv(
    HERE / "genome_version_split.tsv", separator="\t"
)

# Figure.
pf.use_style()
fig, (a, b) = pf.figure(
    pf.TWO_COLUMN_MM, 62, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]}
)
COL = {
    "paper": pf.GREY,
    "htseq": pf.OKABE_ITO["blue"],
    "salmon": pf.OKABE_ITO["vermillion"],
}
LAB = {
    "paper": "The paper's own counts (HTSeq)",
    "htseq": "This run, counted as HTSeq does",
    "salmon": "This run, counted by Salmon",
}
xpos = {
    "paper_table": 0,
    "paper_genome__featurecounts_s0": 1.6,
    "paper_genome__salmon": 2.4,
    "this_genome__featurecounts_s0": 3.9,
    "this_genome__salmon": 4.7,
}
for t, (_, counter) in TABLES.items():
    a.bar(xpos[t], f[t], width=0.75, color=COL[counter], label=LAB[counter])
    a.text(xpos[t], f[t] + 8, f"{f[t]}", ha="center", va="bottom")
a.set_xticks(
    [0, 2.0, 4.3],
    [
        "The paper's\nown table",
        "Paper's genome, botznik\n(g genes)",
        "April 2024 genome\n(FUN genes)",
    ],
)
a.set_ylabel("Paper's listed genes found (of 782)")
a.set_ylim(0, 700)
a.set_xlim(-0.6, 5.3)

ys = np.arange(len(order))[::-1]
b.barh(ys, split["len"], color=COL["htseq"], height=0.6)
for y, n in zip(ys, split["len"]):
    b.text(n + 6, y, f"{n}", va="center")
b.set_yticks(
    ys,
    [
        "Found on both genomes",
        "Only on the paper's genome",
        "Only on the April 2024 genome",
        "Found on neither",
    ],
)
b.set_xlabel(
    f"Genes found on the paper's own table (of {paper_found.height}),\n"
    "status in this run, counted as HTSeq does"
)
b.set_xlim(0, max(split["len"]) * 1.18)

pf.shared_legend(fig, ncol=3)
pf.panel_label(a, "a")
pf.panel_label(b, "b", dx_pt=-90)
pf.save(fig, str(HERE / "figures/genome_version_domen2026_dataset_s1"))
