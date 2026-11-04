#!/usr/bin/env python3
"""Real against swapped labels on the human study lists (notebook 361).

Usage: plot_human_label_swap.py <human_compare.tsv> <swap_compare.tsv> <human_genes_summary.tsv> <swap_genes_summary.tsv> <label_swap/contrasts.tsv> <out_stem>

One row per study list, one column per route. x: share of the list the route reaches. Filled
triangle: real labels. Open triangles: the three swaps. Grey bar: the middle 95 % of 1,000 random
gene sets matched to the list's baseMean deciles, for the real labels' reached genes; the tick is
their median.

Writes <out_stem>.tsv, one row per list and route with the real and swapped shares, and
<out_stem>_summary.tsv, one row per route, and prints both.
  lists_beat_random        lists where 5 % or fewer of the random sets did as well
  real_above_all_swaps     lists where the real share is above all three swaps'
  swap_lists_beat_random   the same test for the swaps, of 36 (3 swaps x 12 lists)
  median_swap_over_real    median of mean swap share / real share, over lists
"""
import sys
from pathlib import Path

import polars as pl
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

real_p, swap_p, rs_p, ss_p, swaps_p, out_stem = sys.argv[1:]
ROUTES = {"splash_compactors": "SPLASH anchors\ncorrected p ≤ 0.05",
          "kamrat_ttest_corrected": "KaMRaT, t-test\ncorrected p ≤ 0.05",
          "kamrat_ttest_p05": "KaMRaT, t-test\np < 0.05",
          "kamrat_deseq2_corrected": "KaMRaT, DESeq2\ncorrected p ≤ 0.05",
          "kamrat_deseq2_p05": "KaMRaT, DESeq2\np < 0.05"}
COL = {r: (pf.OKABE_ITO["blue"] if r.startswith("splash") else pf.OKABE_ITO["reddish_purple"]) for r in ROUTES}
SHORT = {"gse99963_sed_post_vs_pre": "high-fat diet: after vs before",
         "gse99963_ex_post_vs_pre": "diet + exercise: after vs before",
         "gse144304_female_frail_vs_fit": "old women: frail vs fit",
         "gse144304_male_frail_vs_fit": "old men: frail vs fit",
         "gse144304_female_old_vs_young": "women: old vs young",
         "gse144304_male_old_vs_young": "men: old vs young"}
MARK = {"up": "^", "down": "v"}
real = pl.read_csv(real_p, separator="\t").filter(pl.col("contrast").is_in(list(SHORT)))
swaps = pl.read_csv(swaps_p, separator="\t", columns=["contrast", "base_contrast", "swap_name"])
sw = pl.read_csv(swap_p, separator="\t").join(swaps, on="contrast")
BEAT = 0.05

keys = ["base_contrast", "route", "direction"]
wide = (sw.group_by(keys).agg(pl.len().alias("swaps"), pl.col("share").mean().alias("swap_share_mean"),
                              pl.col("share").max().alias("swap_share_max"),
                              pl.col("reached").min().alias("swap_reached_min"),
                              pl.col("reached").max().alias("swap_reached_max"),
                              (pl.col("share_random_at_or_above") <= BEAT).sum().alias("swaps_beat_random"))
        .rename({"base_contrast": "contrast"}))
t = (real.select("series", "contrast", "route", "direction", "list_genes", "reached", "share", "null_median",
                 "share_random_at_or_above")
     .join(wide, on=["contrast", "route", "direction"], how="left", validate="1:1")
     .with_columns((pl.col("share") > pl.col("swap_share_max")).alias("real_above_all_swaps"),
                   (pl.col("swap_share_mean") / pl.col("share")).alias("swap_over_real")))
assert (t["swaps"] == 3).all(), t.filter(pl.col("swaps") != 3)
t.write_csv(f"{out_stem}.tsv", separator="\t")
summary = (t.group_by("route", maintain_order=True)
           .agg(pl.len().alias("lists"), (pl.col("share_random_at_or_above") <= BEAT).sum().alias("lists_beat_random"),
                pl.col("real_above_all_swaps").sum(), pl.col("swaps_beat_random").sum().alias("swap_lists_beat_random"),
                pl.col("swaps").sum().alias("swap_lists"), pl.col("reached").sum().alias("real_reached"),
                pl.col("swap_over_real").median().round(3).alias("median_swap_over_real")))
kept = pl.concat([pl.read_csv(p, separator="\t").select("contrast", "route", "sequences_kept", "genes") for p in (rs_p, ss_p)])
summary.write_csv(f"{out_stem}_summary.tsv", separator="\t")
with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=220):
    print(summary)
    print(t.select("contrast", "route", "direction", "reached", "swap_reached_min", "swap_reached_max", "share",
                   "swap_share_mean", "swap_share_max", "real_above_all_swaps", "swaps_beat_random"))
    print(kept.join(swaps.rename({"contrast": "c"}), left_on="contrast", right_on="c", how="left")
          .with_columns(pl.coalesce("base_contrast", "contrast").alias("real_contrast"),
                        pl.col("swap_name").fill_null("real"))
          .group_by("real_contrast", "route", "swap_name").agg(pl.col("sequences_kept").sum(), pl.col("genes").sum())
          .sort("real_contrast", "route", "swap_name"))

pf.use_style()
rows = [(c, d) for c in SHORT for d in ("up", "down")]
y = {k: i for i, k in enumerate(rows)}
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 95, ncols=len(ROUTES), sharey=True, sharex=True, squeeze=False)
key = [Line2D([], [], linestyle="none", marker="o", color="black", markersize=3.5, label="real labels"),
       Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", color="black", markersize=3.5, markeredgewidth=0.6,
              label="swapped labels"),
       Line2D([], [], color=pf.GREY, linewidth=3, solid_capstyle="butt", label="middle 95 % of random gene sets"),
       Line2D([], [], linestyle="none", marker="|", color="black", markersize=5, label="median of random sets"),
       Line2D([], [], linestyle="none", marker="^", color="black", markersize=3.5, label="study's up list"),
       Line2D([], [], linestyle="none", marker="v", color="black", markersize=3.5, label="study's down list")]
fig.legend(handles=key, loc="outside upper center", ncol=3, frameon=False)
for j, r in enumerate(ROUTES):
    ax = axes[0][j]
    for x in real.filter(pl.col("route") == r).iter_rows(named=True):
        k = (x["contrast"], x["direction"])
        ax.plot([x["null_lo"], x["null_hi"]], [y[k]] * 2, color=pf.GREY, linewidth=2.5, solid_capstyle="butt", zorder=1)
        ax.plot([x["null_median"]], [y[k]], marker="|", color="black", markersize=4, zorder=2)
        ax.plot([x["share"]], [y[k]], marker=MARK[x["direction"]], color=COL[r], markersize=3.5, linestyle="none",
                markeredgewidth=0, zorder=4)
    for x in sw.filter(pl.col("route") == r).iter_rows(named=True):
        k = (x["base_contrast"], x["direction"])
        ax.plot([x["share"]], [y[k]], marker=MARK[x["direction"]], markerfacecolor="white", color=COL[r], markersize=3,
                linestyle="none", markeredgewidth=0.5, zorder=3)
    ax.set_title(ROUTES[r])
    ax.set_xlabel("share of the study's\nlist reached")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)
axes[0][0].set_yticks(range(len(rows)))
axes[0][0].set_yticklabels([f"{SHORT[c]}, {d}" for c, d in rows])
axes[0][0].set_ylim(len(rows) - 0.5, -0.5)
pf.save(fig, out_stem)
print("wrote", out_stem)
