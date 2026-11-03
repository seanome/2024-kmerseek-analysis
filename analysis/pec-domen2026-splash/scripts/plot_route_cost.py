#!/usr/bin/env python3
"""Accuracy against cost for each route on the simulated control (notebook 361).

Usage: plot_route_cost.py <control_truth_summary_p05.json> <control_route_cost.tsv> <out_stem>

The control has 3,000 genes: 400 with a planted change (150 raised 4x, 150 lowered 4x, 100 with
isoforms switched 9x) and 2,600 unchanged. A gene is found when a kept sequence lies in that gene
only (control_truth.py). Per route:
  sensitivity = planted genes found / 400
  specificity = unchanged genes not found / 2,600
  precision   = planted found / all found
  F1          = 2 x precision x sensitivity / (precision + sensitivity)
  accuracy    = (planted found + unchanged not found) / 3,000
Cost is the detection steps only, up to the kept sequences, on the control's 12 samples; the
protein search after them is left out because the control places sequences on genes directly.
Rows: the five measures. Columns: CPU hours, wall time, peak memory, each axis reversed so that
less cost is to the right: up and to the right is better.
Writes <out_stem>.tsv with every number drawn, and prints it.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

summary_json, cost_tsv, out_stem = sys.argv[1:]
C = pf.OKABE_ITO
# channel in the summary: (label, cost row, colour, marker, filled = corrected cutoff)
ROUTES = {
    "splash": ("SPLASH anchors, corrected p ≤ 0.05", "splash", C["blue"], "o", True),
    "abundance": ("anchor counts, DESeq2, corrected p ≤ 0.05, 2-fold", "anchor_deseq2", C["vermillion"], "s", True),
    "dekupl_limma": ("DE-kupl, limma, corrected p ≤ 0.05, 2-fold", "dekupl_limma", C["bluish_green"], "o", True),
    "dekupl_limma_p05": ("DE-kupl, limma, p < 0.05", "dekupl_limma", C["bluish_green"], "o", False),
    "dekupl_DESeq2": ("DE-kupl, DESeq2, corrected p ≤ 0.05, 2-fold", "dekupl_deseq2", C["orange"], "s", True),
    "dekupl_DESeq2_p05": ("DE-kupl, DESeq2, p < 0.05", "dekupl_deseq2", C["orange"], "s", False),
    "kamrat_ttest": ("KaMRaT, t-test, corrected p ≤ 0.05", "kamrat_ttest", C["reddish_purple"], "o", True),
    "kamrat_ttest_p05": ("KaMRaT, t-test, p < 0.05", "kamrat_ttest", C["reddish_purple"], "o", False),
    "kamrat_deseq2_q05": ("KaMRaT, DESeq2, corrected p ≤ 0.05", "kamrat_deseq2", C["reddish_purple"], "s", True),
    "kamrat_deseq2_p05": ("KaMRaT, DESeq2, p < 0.05", "kamrat_deseq2", C["reddish_purple"], "s", False),
}
PLANTED = ["abundance_up", "abundance_down", "isoform_switch"]

ch = json.load(open(summary_json))["channels"]
cost = pl.read_csv(cost_tsv, separator="\t")
rows = []
for key, (label, cost_row, *_rest) in ROUTES.items():
    g, n = ch[key]["genes_found"], ch[key]["genes_in_class"]
    tp, pos = sum(g[c] for c in PLANTED), sum(n[c] for c in PLANTED)
    fp, neg = g["unchanged"], n["unchanged"]
    sens, spec = tp / pos, 1 - fp / neg
    prec = tp / (tp + fp) if tp + fp else float("nan")
    c = cost.filter(pl.col("route") == cost_row).row(0, named=True)
    rows.append(dict(channel=key, route=label, planted_found=tp, planted=pos, unchanged_found=fp, unchanged=neg,
                     sensitivity=round(sens, 4), specificity=round(spec, 4), precision=round(prec, 4),
                     F1=round(2 * prec * sens / (prec + sens), 4), accuracy=round((tp + neg - fp) / (pos + neg), 4),
                     cpu_hours=round(c["cpu_seconds"] / 3600, 3), wall_minutes=round(c["wall_seconds"] / 60, 1),
                     peak_gb=c["peak_gb"]))
t = pl.DataFrame(rows)
t.write_csv(f"{out_stem}.tsv", separator="\t")
with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250):
    print(t)

METRICS = ["sensitivity", "specificity", "precision", "F1", "accuracy"]
COSTS = [("cpu_hours", "CPU hours"), ("wall_minutes", "wall time (minutes)"), ("peak_gb", "peak memory (GB)")]  # all lower is better
TICKS = [[0.2, 0.5, 1, 2, 5], [1, 2, 5, 10, 20, 50], [0.5, 1, 2, 5, 10, 20]]
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 165, nrows=len(METRICS), ncols=len(COSTS), sharex="col", sharey="row",
                     layout="constrained")
for r in t.iter_rows(named=True):
    _, _, col, mk, filled = ROUTES[r["channel"]]
    for i, m in enumerate(METRICS):
        for j, (cc, _) in enumerate(COSTS):
            axs[i, j].plot(r[cc], r[m], marker=mk, markersize=4, linestyle="none", color=col,
                           markerfacecolor=col if filled else "white", markeredgewidth=0.8,
                           label=r["route"] if (i, j) == (0, 0) else None)
for i, m in enumerate(METRICS):
    axs[i, 0].set_ylabel(m)
    for j in range(len(COSTS)):
        axs[i, j].set_ylim(0, 1.05)
for j, (_, lab) in enumerate(COSTS):
    axs[-1, j].set_xscale("log")
    axs[-1, j].set_xticks(TICKS[j], [f"{v:g}" for v in TICKS[j]])
    axs[-1, j].minorticks_off()
    axs[-1, j].invert_xaxis()
    axs[-1, j].set_xlabel(f"{lab}\n← more    less →")
for a, l in zip(axs[:, 0], "abcde"):
    pf.panel_label(a, l)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False,
           title="up and to the right is better.   colour: tool and test; square: DESeq2 test; filled: corrected cutoff, open: p < 0.05")
pf.save(fig, out_stem)
print("wrote", out_stem)
