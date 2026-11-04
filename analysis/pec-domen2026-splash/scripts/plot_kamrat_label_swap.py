#!/usr/bin/env python3
"""Real against swapped labels for KaMRaT + DESeq2 on the paper's lists (notebook 361).

Usage: plot_kamrat_label_swap.py <kamrat_swap_recovery.tsv> <kamrat_swap_kept.tsv> <out_stem>

One row per paper list. x: share of the list reached. Filled triangle: real labels. Open triangles:
swaps. Grey bar: the middle 95 % of 1,000 random gene sets for the real labels' reached genes; the
tick is their median. Same marks as exports/human/label_swap/human_label_swap.png.

Writes <out_stem>.tsv, one row per list, <out_stem>_summary.tsv, one row per contrast and a total,
and prints both.
  lists_beat_random       lists where 5 % or fewer of the random sets did as well
  real_above_all_swaps    lists where the real share is above every swap's
  swap_lists_beat_random  the same test for the swaps
  median_swap_over_real   median of mean swap share / real share, over lists with a nonzero share
  ..._either_way          the same, also counting list genes reached by contigs going the other way
"""
import sys
from pathlib import Path

import polars as pl
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

rec_p, kept_p, out_stem = sys.argv[1:]
SHORT = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h", "young_13_49d": "young 13-49 d",
         "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
MARK = {"up": "^", "down": "v"}
COL = pf.OKABE_ITO["reddish_purple"]
BEAT = 0.05
keys = ["contrast", "list"]
d = pl.read_csv(rec_p, separator="\t").with_columns(
    (pl.col("reached_either_way") / pl.col("list_genes_with_protein")).alias("fraction_either_way"))
real, sw = d.filter(pl.col("run") == "real"), d.filter(pl.col("run") != "real")
wide = sw.group_by(keys).agg(pl.len().alias("swaps"), pl.col("fraction_reached").mean().alias("swap_mean"),
                             pl.col("fraction_reached").max().alias("swap_max"),
                             pl.col("fraction_either_way").mean().alias("swap_either_way_mean"),
                             pl.col("reached").min().alias("swap_reached_min"), pl.col("reached").max().alias("swap_reached_max"),
                             (pl.col("share_null_at_or_above") <= BEAT).sum().alias("swaps_beat_random"))
t = (real.select(*keys, "list_genes_with_protein", "reached", "fraction_reached", "fraction_either_way", "null_median",
                 "null_2_5", "null_97_5", "share_null_at_or_above")
     .join(wide, on=keys, how="left", validate="1:1")
     .with_columns((pl.col("fraction_reached") > pl.col("swap_max")).alias("real_above_all_swaps"),
                   (pl.col("swap_mean") / pl.col("fraction_reached")).alias("swap_over_real"),
                   (pl.col("swap_either_way_mean") / pl.col("fraction_either_way")).alias("swap_over_real_either_way")))
assert t["swaps"].null_count() == 0
t.write_csv(f"{out_stem}.tsv", separator="\t")


def summarise(g):
    return g.agg(pl.len().alias("lists"), (pl.col("share_null_at_or_above") <= BEAT).sum().alias("lists_beat_random"),
                 pl.col("real_above_all_swaps").sum(), pl.col("swaps_beat_random").sum().alias("swap_lists_beat_random"),
                 pl.col("swaps").sum().alias("swap_lists"), pl.col("reached").sum().alias("real_reached"),
                 pl.col("swap_over_real").median().round(3).alias("median_swap_over_real"),
                 pl.col("swap_over_real_either_way").median().round(3).alias("median_swap_over_real_either_way"))


summary = pl.concat([summarise(t.group_by("contrast", maintain_order=True)),
                     summarise(t.with_columns(pl.lit("all").alias("contrast")).group_by("contrast"))])
summary.write_csv(f"{out_stem}_summary.tsv", separator="\t")
kept = pl.read_csv(kept_p, separator="\t")
with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=220):
    print(summary)
    print(t.select(*keys, "reached", "swap_reached_min", "swap_reached_max", "fraction_reached", "swap_mean", "swap_max",
                   "real_above_all_swaps", "swaps_beat_random"))
    print(kept)

pf.use_style()
rows = [(c, x) for c in SHORT for x in ("up", "down")]
y = {k: i for i, k in enumerate(rows)}
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 80, layout="constrained")
key = [Line2D([], [], linestyle="none", marker="^", color=COL, markersize=3.5, markeredgewidth=0, label="real labels"),
       Line2D([], [], linestyle="none", marker="^", markerfacecolor="white", color=COL, markersize=3, markeredgewidth=0.5,
              label="swapped labels"),
       Line2D([], [], color=pf.GREY, linewidth=2.5, solid_capstyle="butt", label="middle 95 % of random gene sets"),
       Line2D([], [], linestyle="none", marker="|", color="black", markersize=4, label="median of random sets")]
fig.legend(handles=key, loc="outside upper center", ncol=2, frameon=False,
           title="KaMRaT, DESeq2, p < 0.05.   up triangle: paper's up list; down triangle: down list")
for x in real.iter_rows(named=True):
    k = (x["contrast"], x["list"])
    ax.plot([x["null_2_5"], x["null_97_5"]], [y[k]] * 2, color=pf.GREY, linewidth=2.5, solid_capstyle="butt", zorder=1)
    ax.plot([x["null_median"]], [y[k]], marker="|", color="black", markersize=4, zorder=2)
    ax.plot([x["fraction_reached"]], [y[k]], marker=MARK[x["list"]], color=COL, markersize=3.5, linestyle="none",
            markeredgewidth=0, zorder=4)
for x in sw.iter_rows(named=True):
    k = (x["contrast"], x["list"])
    ax.plot([x["fraction_reached"]], [y[k]], marker=MARK[x["list"]], markerfacecolor="white", color=COL, markersize=3,
            linestyle="none", markeredgewidth=0.5, zorder=3)
ax.set_xlabel("share of the paper's list reached")
ax.set_xlim(0, 1)
ax.grid(axis="x", color="#dddddd", linewidth=0.4)
ax.set_yticks(range(len(rows)), [f"{SHORT[c]}, {x}" for c, x in rows])
ax.set_ylim(len(rows) - 0.5, -0.5)
pf.save(fig, out_stem)
print("wrote", out_stem)
