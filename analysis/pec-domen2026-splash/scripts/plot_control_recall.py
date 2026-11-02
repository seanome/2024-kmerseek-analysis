#!/usr/bin/env python3
"""Figure: share of the planted genes each method finds in the simulated control.

Usage: plot_control_recall.py <out_stem> <control_truth_summary.json> [<more summary.json> ...]

Channels are taken from the summaries in order; a later file adds the channels the earlier ones
lack (KaMRaT was scored in a later run, which did not score DE-kupl DESeq2). A channel present in
two files must have the same genes found in both.

The control has 3,000 genes: 150 raised 4x, 150 lowered 4x, 100 with two isoforms switched 9x,
and 2,600 unchanged. One panel per gene class, one bar per method. A gene is found when one of
its sequences that sits in that gene only is kept by the method (control_truth.py). In the
unchanged panel every found gene is a false find.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

out_stem, jsons = sys.argv[1], sys.argv[2:]
ch = {}
for j in jsons:
    for m, v in json.load(open(j))["channels"].items():
        if m in ch:
            assert ch[m]["genes_found"] == v["genes_found"], (m, j)
        else:
            ch[m] = v
METHODS = [("splash", "SPLASH anchors"),
           ("abundance", "anchor counts, DESeq2"),
           ("dekupl_limma", "DE-kupl k-mers, limma"),
           ("dekupl_DESeq2", "DE-kupl k-mers, DESeq2"),
           ("kamrat_ttest", "KaMRaT k-mers, t-test")]
CLASSES = [("abundance_up", "raised 4x"), ("abundance_down", "lowered 4x"),
           ("isoform_switch", "isoforms switched 9x"), ("unchanged", "unchanged (false finds)")]
# SPLASH blue and anchor counts vermillion, as in plot_paper_recovery.py
COLOUR = [pf.OKABE_ITO["blue"], pf.OKABE_ITO["vermillion"], pf.OKABE_ITO["bluish_green"], pf.OKABE_ITO["orange"],
          pf.OKABE_ITO["reddish_purple"]]

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 52, ncols=len(CLASSES), sharey=True)
for ax, (cls, title) in zip(axs, CLASSES):
    for y, ((m, _), colour) in enumerate(zip(METHODS, COLOUR)):
        found, total = ch[m]["genes_found"][cls], ch[m]["genes_in_class"][cls]
        ax.barh(y, found / total, color=colour, height=0.7)
        ax.text(found / total + 0.02, y, f"{found}/{total}", va="center")
    ax.set_title(title)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("share of genes found")
axs[0].set_yticks(range(len(METHODS)), [name for _, name in METHODS])
axs[0].invert_yaxis()
pf.save(fig, out_stem)
print("wrote", out_stem)
