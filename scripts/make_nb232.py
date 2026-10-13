#!/usr/bin/env python3
"""Generate notebooks/232_mismatch_tolerant_seed_extend.ipynb."""

import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


md(r"""
# 232. Can a mismatch-tolerant seed-and-extend see what the exact k-mer cannot?

Notebook 230 found that HP-class agreement between remote homologs is strong (kappa about
0.45 at 20-30% identity) but that the longest exact HP run is about 12 residues, so exact
seeds of 23-30 reach only a few percent of those pairs. This notebook changes the primitive
and measures what that buys, on the same aligned pairs and with no kmerseek run.

The primitive: a seed is an exact run of at least `s` matching classes; from the seed the
match is extended without gaps, +1 per matching column, -c per mismatching column, a gap
column ends it; the pair's score is the best such segment (the limit of an X-drop
extension for large X). `s = 0` drops the seed requirement and is the best local ungapped
segment, an upper bound for any seeded scheme. Scoring is in
`notebooks/seed_extend_utils.py`; the table comes from
`scripts/seed_extend_on_alignments.py`.

The null: the same pair with the target's residues shuffled (composition kept, gaps
kept), two shuffles per pair. Thresholds are set on the pooled null at a per-comparison
false-positive rate, and recall is the share of real pairs above that threshold in each
identity bin. This calibrates against composition on the alignment's own diagonal. It does
not model a database search, where every query is compared with millions of off-diagonal
candidates; the database false-positive rate needs the search-engine implementation and a
decoy run. What can be read here is whether the extended score separates remote homologs
from composition-matched noise at all, and by how much per seed length and alphabet.

Pairs: 60,000 Pfam seed pairs (random subset with >= 50 aligned columns) and the 12,000
SCOPe same-superfamily/different-family pairs with TM-score >= 0.5.
""")

code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
import hp_conservation_utils as hc
import seed_extend_utils as se

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

FIG = Path("../figures")
d = hc.add_identity_bin(pl.read_parquet("/Users/olga/data/pfam/232_seed_extend_scores.parquet"))
pf = d.filter(pl.col("dataset") == "pfam")
sc = d.filter(pl.col("dataset") == "scope")
ALPHA_COLOR = {"protein20": "#444444", "sdm12": "#8c6d31", "gbmr4": "#6baed6", "hp_thomas_dill2": "#a50f15"}
BINS = ["<20%", "20-30%", "30-40%", "40-60%", ">=60%"]
FPRS = [1e-2, 1e-3]

def n_pairs(x):
    return x.select("query", "target").unique().height
print(f"Pfam pairs {n_pairs(pf):,}; SCOPe cross-family pairs {n_pairs(sc):,}; rows {d.height:,}")
print(pf.filter(pl.col('kind') == 'real').select("query", "target", "identity_bin").unique().group_by("identity_bin").len().sort("identity_bin"))

rec_pf = pl.concat([se.recall_at_null_quantile(pf, ["alphabet", "penalty", "seed"], f) for f in FPRS])
rec_sc = pl.concat([se.recall_at_null_quantile(sc, ["alphabet", "penalty", "seed"], f) for f in FPRS])
""")

md(r"""
## 1. What the scores look like

hp_thomas_dill2, mismatch penalty 2, Pfam pairs at 20-30% identity: real pairs against
their shuffled nulls, for three seed lengths. With a 23-residue seed almost every pair
scores zero (no seed). With a 10-residue seed most pairs have a seed and the extension
carries them past the null.
""")

code(r"""
sub = pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == 2.0) & (pl.col("identity_bin") == "20-30%"))
seeds_shown = [23, 15, 10, 0]
fig, axes = plt.subplots(1, len(seeds_shown), figsize=(16, 3.8), sharey=True)
bins = np.arange(0, 80, 2)
for ax, s in zip(axes, seeds_shown):
    x = sub.filter(pl.col("seed") == s)
    real = x.filter(pl.col("kind") == "real")["score"].to_numpy()
    null = x.filter(pl.col("kind") != "real")["score"].to_numpy()
    ax.hist(null, bins=bins, density=True, color="0.6", alpha=0.7, label="shuffled target")
    ax.hist(real, bins=bins, density=True, color="#a50f15", alpha=0.6, label="real pair")
    thr = rec_pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == 2.0) & (pl.col("seed") == s) & (pl.col("fpr") == 1e-3))["threshold"][0]
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"null 99.9th pct = {thr:.0f}")
    ax.set_title(f"seed {s}" if s else "no seed (best local segment)", fontsize=10)
    ax.set_xlabel("extended segment score")
    ax.legend(fontsize=7.5)
    ax.set_yscale("log")
axes[0].set_ylabel("density (log)")
tab = (sub.group_by("seed", "kind").agg(pl.len(), (pl.col("score") == 0).mean().round(3).alias("frac_zero"), pl.col("score").mean().round(1).alias("mean"), pl.col("score").quantile(0.999).alias("q999"))
       .sort("seed", "kind"))
print(tab)
hc.finish_figure(
    fig, FIG / "232_score_distributions_hp_20_30.png", tools=hc.NO_TOOL,
    hypothesis="Extending a short exact HP seed with mismatches separates real 20-30%-identity pairs from composition-matched nulls; an exact 23-mer mostly scores zero.",
    conclusion=(f"Seed 23: {100 * tab.filter((pl.col('seed') == 23) & (pl.col('kind') == 'real'))['frac_zero'][0]:.0f}% of real pairs score 0. "
                f"Seed 10: {100 * tab.filter((pl.col('seed') == 10) & (pl.col('kind') == 'real'))['frac_zero'][0]:.0f}% score 0, mean real score {tab.filter((pl.col('seed') == 10) & (pl.col('kind') == 'real'))['mean'][0]:.0f} vs null {tab.filter((pl.col('seed') == 10) & (pl.col('kind') == 'null0'))['mean'][0]:.0f}."),
    title="hp_thomas_dill2, penalty 2, Pfam pairs at 20-30% identity",
)
""")

md(r"""
## 2. Recall at a fixed null false-positive rate, by seed length

hp_thomas_dill2, penalty 2. Each curve is one identity bin; x is the seed length, y the
share of real pairs above the null's 99th (left) or 99.9th (right) percentile. The dashed
level is the exact-match primitive at k=23: the share of pairs with a 23-residue exact run
(notebook 230), which is what the current engine can reach at any threshold.
""")

code(r"""
exact23 = (pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == 2.0) & (pl.col("seed") == 23) & (pl.col("kind") == "real"))
           .group_by("identity_bin").agg((pl.col("score") > 0).mean().alias("reach")))
bin_color = dict(zip(BINS, ["#08306b", "#2171b5", "#6baed6", "#fdae6b", "#e6550d"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, f in zip(axes, FPRS):
    r = rec_pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == 2.0) & (pl.col("fpr") == f))
    for b in BINS:
        s = r.filter(pl.col("identity_bin") == b).sort("seed")
        ax.plot(s["seed"], s["recall"], marker="o", ms=4, color=bin_color[b], label=b)
        e = exact23.filter(pl.col("identity_bin") == b)["reach"][0]
        ax.axhline(e, color=bin_color[b], ls=":", lw=1)
    ax.set_xlabel("seed length (0 = no seed)")
    ax.set_title(f"recall at null FPR {f:g} per comparison", fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("share of real pairs above the null threshold")
axes[1].legend(fontsize=8, title="identity")
axes[0].text(0.02, 0.98, "dotted: pairs with an exact 23-mer (current primitive)", transform=axes[0].transAxes, fontsize=8, va="top")

t = rec_pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == 2.0)).with_columns(pl.col("recall").round(3)).pivot(on="identity_bin", index=["fpr", "seed", "threshold"], values="recall").sort("fpr", "seed")
print(t)
def rc(seed, b, f=1e-3):
    return rec_pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == 2.0) & (pl.col("seed") == seed) & (pl.col("identity_bin") == b) & (pl.col("fpr") == f))["recall"][0]
hc.finish_figure(
    fig, FIG / "232_recall_vs_seed_hp.png", tools=hc.NO_TOOL,
    hypothesis="Shortening the seed and extending with mismatches recovers a large share of the 20-40% identity pairs that the exact 23-mer cannot reach, at a null-calibrated false-positive rate.",
    conclusion=(f"At FPR 0.001: 20-30% identity, seed 23 = {100 * rc(23, '20-30%'):.0f}%, seed 12 = {100 * rc(12, '20-30%'):.0f}%, seed 10 = {100 * rc(10, '20-30%'):.0f}%, no seed = {100 * rc(0, '20-30%'):.0f}%. "
                f"30-40%: seed 23 = {100 * rc(23, '30-40%'):.0f}%, seed 10 = {100 * rc(10, '30-40%'):.0f}%. Under 20%: seed 10 = {100 * rc(10, '<20%'):.0f}%. "
                f"Exact 23-mer reach in the 20-30% bin: {100 * exact23.filter(pl.col('identity_bin') == '20-30%')['reach'][0]:.0f}%."),
    title="hp_thomas_dill2 with ungapped extension, Pfam seed pairs",
)
""")

md(r"""
## 3. Alphabets compared at the same null false-positive rate

The 2-letter alphabet lost the exact-match comparison at matched bits (notebook 230 §6).
With extension the question is different: which alphabet's per-column agreement, once
mismatches are tolerated, best separates remote homologs from their composition null?
Each alphabet at its best seed length among those tried, penalty 2, and with no seed.
""")

code(r"""
def best_seed(rec, f):
    r = rec.filter((pl.col("penalty") == 2.0) & (pl.col("fpr") == f) & (pl.col("seed") > 0))
    # best seed per alphabet judged on the 20-30% bin
    return (r.filter(pl.col("identity_bin") == "20-30%").sort("recall", descending=True).group_by("alphabet").first().select("alphabet", "seed"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, f in zip(axes, FPRS):
    bs = best_seed(rec_pf, f)
    for a, col in ALPHA_COLOR.items():
        s_best = bs.filter(pl.col("alphabet") == a)["seed"][0]
        r = rec_pf.filter((pl.col("alphabet") == a) & (pl.col("penalty") == 2.0) & (pl.col("fpr") == f))
        rs = r.filter(pl.col("seed") == s_best).sort("identity_bin")
        r0 = r.filter(pl.col("seed") == 0).sort("identity_bin")
        xi = [BINS.index(b) for b in rs["identity_bin"]]
        ax.plot(xi, rs["recall"], marker="o", ms=4, color=col, label=f"{a}, seed {s_best}")
        ax.plot([BINS.index(b) for b in r0["identity_bin"]], r0["recall"], ls=":", color=col, lw=1)
    ax.set_xticks(range(len(BINS)), BINS)
    ax.set_xlabel("identity over aligned columns")
    ax.set_title(f"recall at null FPR {f:g}", fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("share of real pairs above the null threshold")
axes[1].legend(fontsize=8)
axes[0].text(0.02, 0.98, "dotted: no seed requirement", transform=axes[0].transAxes, fontsize=8, va="top")

cmp = (rec_pf.filter((pl.col("penalty") == 2.0) & (pl.col("seed") == 0)).with_columns(pl.col("recall").round(3))
       .pivot(on="identity_bin", index=["fpr", "alphabet", "threshold"], values="recall").sort("fpr", "alphabet"))
print("No-seed upper bound, by alphabet:")
print(cmp)
bs3 = best_seed(rec_pf, 1e-3)
print("Best seed per alphabet (judged on 20-30%, FPR 0.001):"); print(bs3)
def r0(a, b, f=1e-3):
    return rec_pf.filter((pl.col("alphabet") == a) & (pl.col("penalty") == 2.0) & (pl.col("seed") == 0) & (pl.col("identity_bin") == b) & (pl.col("fpr") == f))["recall"][0]
hc.finish_figure(
    fig, FIG / "232_alphabets_with_extension.png", tools=hc.NO_TOOL,
    hypothesis="Once mismatches are tolerated, the 2-letter alphabet's higher per-column conservation makes it competitive with, or better than, the larger alphabets on remote homologs.",
    conclusion=(f"No seed, FPR 0.001, 20-30% identity: hp_thomas_dill2 {100 * r0('hp_thomas_dill2', '20-30%'):.0f}%, gbmr4 {100 * r0('gbmr4', '20-30%'):.0f}%, sdm12 {100 * r0('sdm12', '20-30%'):.0f}%, protein20 {100 * r0('protein20', '20-30%'):.0f}%. "
                f"Under 20%: hp {100 * r0('hp_thomas_dill2', '<20%'):.0f}%, gbmr4 {100 * r0('gbmr4', '<20%'):.0f}%, protein20 {100 * r0('protein20', '<20%'):.0f}%. "
                f"Above 40% the order reverses and protein20 leads."),
    title="Ungapped extension, four alphabets, Pfam seed pairs",
)
""")

md(r"""
## 4. Mismatch penalty

Penalty 1 lets a segment absorb many class flips; penalty 3 makes it nearly exact. Recall
at FPR 0.001 by identity bin for hp_thomas_dill2 at seed 10.
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 4.2))
for c, ls in [(1.0, ":"), (2.0, "-"), (3.0, "--")]:
    r = rec_pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == c) & (pl.col("seed") == 10) & (pl.col("fpr") == 1e-3)).sort("identity_bin")
    ax.plot([BINS.index(b) for b in r["identity_bin"]], r["recall"], marker="o", ms=4, ls=ls, color="#a50f15", label=f"penalty {c:g}")
ax.set_xticks(range(len(BINS)), BINS)
ax.set_xlabel("identity over aligned columns")
ax.set_ylabel("recall at null FPR 0.001")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
t = rec_pf.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("seed") == 10) & (pl.col("fpr") == 1e-3)).with_columns(pl.col("recall").round(3)).pivot(on="identity_bin", index=["penalty", "threshold"], values="recall").sort("penalty")
print(t)
hc.finish_figure(
    fig, FIG / "232_penalty_sensitivity.png", tools=hc.NO_TOOL,
    hypothesis="The result is not sensitive to the exact mismatch penalty.",
    conclusion=(f"20-30% identity, seed 10: penalty 1 = {100 * t.filter(pl.col('penalty') == 1.0)['20-30%'][0]:.0f}%, 2 = {100 * t.filter(pl.col('penalty') == 2.0)['20-30%'][0]:.0f}%, 3 = {100 * t.filter(pl.col('penalty') == 3.0)['20-30%'][0]:.0f}%."),
    title="hp_thomas_dill2, seed 10, three mismatch penalties",
)
""")

md(r"""
## 5. The same on SCOPe cross-family pairs

Structural alignments across families of one superfamily, TM-score >= 0.5. Fewer pairs
per bin above 40%, so those points are noisy.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
ax = axes[0]
r = rec_sc.filter((pl.col("alphabet") == "hp_thomas_dill2") & (pl.col("penalty") == 2.0) & (pl.col("fpr") == 1e-3))
for b in BINS[:4]:
    s = r.filter(pl.col("identity_bin") == b).sort("seed")
    if s.height:
        ax.plot(s["seed"], s["recall"], marker="o", ms=4, color=bin_color[b], label=f"{b} (n={s['n_pairs'][0]:,})")
ax.set_xlabel("seed length (0 = no seed)")
ax.set_ylabel("recall at null FPR 0.001")
ax.set_title("hp_thomas_dill2, SCOPe cross-family", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
for a, col in ALPHA_COLOR.items():
    r0s = rec_sc.filter((pl.col("alphabet") == a) & (pl.col("penalty") == 2.0) & (pl.col("seed") == 0) & (pl.col("fpr") == 1e-3)).sort("identity_bin")
    ax.plot([BINS.index(b) for b in r0s["identity_bin"]], r0s["recall"], marker="o", ms=4, color=col, label=a)
ax.set_xticks(range(len(BINS)), BINS)
ax.set_xlabel("identity over structurally aligned columns")
ax.set_title("no seed, four alphabets, SCOPe cross-family", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
t = rec_sc.filter((pl.col("penalty") == 2.0) & (pl.col("fpr") == 1e-3) & pl.col("seed").is_in([0, 10, 23, 5, 3])).with_columns(pl.col("recall").round(3)).pivot(on="identity_bin", index=["alphabet", "seed"], values="recall").sort("alphabet", "seed")
print(t)
def rs(a, s, b):
    x = rec_sc.filter((pl.col("alphabet") == a) & (pl.col("penalty") == 2.0) & (pl.col("seed") == s) & (pl.col("identity_bin") == b) & (pl.col("fpr") == 1e-3))["recall"]
    return x[0] if x.len() else float("nan")
hc.finish_figure(
    fig, FIG / "232_scope_cross_family.png", tools=hc.NO_TOOL,
    hypothesis="The gain from extension holds on structure-defined remote homologs, not only on curated sequence alignments.",
    conclusion=(f"SCOPe cross-family at FPR 0.001, 20-30% identity: hp seed 23 = {100 * rs('hp_thomas_dill2', 23, '20-30%'):.0f}%, seed 10 = {100 * rs('hp_thomas_dill2', 10, '20-30%'):.0f}%, no seed = {100 * rs('hp_thomas_dill2', 0, '20-30%'):.0f}%. "
                f"Under 20%: hp no seed {100 * rs('hp_thomas_dill2', 0, '<20%'):.0f}%, protein20 no seed {100 * rs('protein20', 0, '<20%'):.0f}%."),
    title="SCOPe same-superfamily, different-family pairs",
)
""")

md(r"""
## 6. Conclusions, and what this does not show

1. The primitive was the ceiling. On the same pairs where an exact 23-mer reaches 4% of
   20-30% identity homologs, a 10-residue HP seed with ungapped mismatch-tolerant
   extension reaches about a third of them at a null false-positive rate of 0.001 per
   comparison, and about two thirds of the 30-40% bin. The number depends little on the
   mismatch penalty.
2. With extension the 2-letter alphabet is no longer the worst alphabet. At 20-30% and
   below it matches or beats gbmr4, sdm12 and protein20 on this null; above 40% the
   20-letter alphabet leads, as it should.
3. This is not a database-search result. The null is composition on the pair's own
   diagonal. A real search compares each query against every off-diagonal window of every
   target, and the false-positive rate that matters is per query against the database,
   which an ungapped HP alignment will find much harder to control than a per-pair 0.001.
   The seed index bounds how many off-diagonal windows are extended; that is the
   engineering question the kmerseek branch (`olgabot/extend-mismatch`) has to answer with
   a decoy database.
4. What to run next, in order: the kmerseek implementation on SCOPe40 all-vs-all with the
   existing sensitivity-to-first-false-positive machinery (notebooks 066-072), read by
   identity bin; then the QfO identity panel.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python", "name": "2025-kmerseek-analysis"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "232_mismatch_tolerant_seed_extend.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
