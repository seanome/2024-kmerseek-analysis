#!/usr/bin/env python3
"""Figure for call_types.py: what kind of sequence difference SPLASH calls are (notebook 361).

Usage: plot_call_types.py <dir with call_types.py TSVs> <out_stem>

One column per contrast (the simulated control, young 2 h, young 24 h); rows:
a. Share of anchors in each bin of Hamming distance between the anchor's two most abundant
   targets: SPLASH's kept anchors, the anchors kept by DESeq2 on anchor counts, and a random
   300,000 of all tested anchors with two or more targets.
b. Where the reached peptides fall along the protein (0 = start, 1 = end), as density per
   twentieth of the protein; 1 is what an even spread would give.
c. Genes reached: by at least one gene-unique peptide (solid) or only by peptides that also
   match other genes (hatched). The control has no anchor-count route.
d. Share of SPLASH anchors whose second target's share moves the same way (PEC against control)
   in every colony pair counted, by Hamming bin; the dotted line is the share expected if each
   pair's direction were a coin toss (target_share_by_pair.py).
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

d, out_stem = Path(sys.argv[1]), sys.argv[2]
CONTRASTS = ["control", "young_2h", "young_24h"]
NAME = {"control": "simulated control", "young_2h": "young 2 h", "young_24h": "young 24 h"}
SETS = {"splash": "SPLASH", "anchor_nominal": "anchor counts, DESeq2", "all_tested_sample": "all tested anchors (sample)"}
COLOUR = {"splash": pf.OKABE_ITO["blue"], "anchor_nominal": pf.OKABE_ITO["vermillion"], "all_tested_sample": pf.GREY}
BINS = ["1", "2-3", "4-10", ">10"]

dist = pl.read_csv(d / "target_distance_bins.tsv", separator="\t", schema_overrides={"hamming_bin": pl.Utf8})
pos = pl.read_csv(d / "protein_position_bins.tsv", separator="\t")
uniq = pl.read_csv(d / "gene_unique_reach.tsv", separator="\t")
share = pl.read_csv(d / "target_share_by_pair.tsv", separator="\t")

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 160, nrows=4, ncols=3)
for j, c in enumerate(CONTRASTS):
    ax = axs[0, j]
    present = [s for s in SETS if dist.filter((pl.col("contrast") == c) & (pl.col("anchor_set") == s)).height]
    w = 0.8 / len(present)
    for k, s in enumerate(present):
        t = dist.filter((pl.col("contrast") == c) & (pl.col("anchor_set") == s))
        frac = [t.filter(pl.col("hamming_bin") == b)["fraction"].sum() for b in BINS]
        ax.bar([i - 0.4 + w * (k + 0.5) for i in range(len(BINS))], frac, width=w, color=COLOUR[s],
               label=SETS[s], linewidth=0)
    ax.set_xticks(range(len(BINS)), BINS)
    ax.set_xlabel("bases that differ between the two top targets")
    ax.set_title(NAME[c])
    ax.set_ylim(0, 1)

    ax = axs[1, j]
    for route in ["splash", "anchor_nominal"]:
        t = pos.filter((pl.col("contrast") == c) & (pl.col("route") == route)).sort("bin_start")
        if not t.height:
            continue
        x = t["bin_start"].to_list() + [t["bin_end"][-1]]
        y = [f * len(t) for f in t["fraction"].to_list()]
        ax.stairs(y, x, color=COLOUR[route], linewidth=1)
    ax.axhline(1, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0\nstart (N)", "0.5", "1\nend (C)"])
    ax.set_xlabel("position of peptide along protein")

    ax = axs[2, j]
    t = uniq.filter(pl.col("contrast") == c)
    for k, route in enumerate(["splash", "anchor_nominal"]):
        r = t.filter(pl.col("route") == route)
        if not r.height:
            ax.text(k, 0.02, "not run\nin control", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5)
            continue
        r = r.row(0, named=True)
        u, sh = r["genes_gene_unique_peptides"], r["genes_only_shared_peptides"]
        ax.bar(k, u, color=COLOUR[route], linewidth=0, width=0.6)
        ax.bar(k, sh, bottom=u, facecolor="white", edgecolor=COLOUR[route], hatch="//////", linewidth=0.6, width=0.6)
        ax.annotate(f"{u + sh:,}\n{sh / (u + sh):.0%} only shared", (k, u + sh), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=5)
    ax.set_xticks([0, 1], ["SPLASH", "anchor counts"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, max(t["genes_all_peptides"]) * 1.35)

    ax = axs[3, j]
    t = share.filter(pl.col("contrast") == c)
    frac = [t.filter(pl.col("hamming_bin") == b)["frac_same_direction"].sum() for b in BINS]
    chance = [t.filter(pl.col("hamming_bin") == b)["same_direction_by_chance"].sum() for b in BINS]
    ax.bar(range(len(BINS)), frac, width=0.6, color=COLOUR["splash"], linewidth=0)
    for i, ch in enumerate(chance):
        ax.plot([i - 0.4, i + 0.4], [ch, ch], color="black", linewidth=0.6, linestyle=":")
    n = t.filter(pl.col("hamming_bin") == "all")["anchors"].item()
    ax.text(0.03, 0.97, f"{n:,} SPLASH anchors with 4+ pairs counted", transform=ax.transAxes, va="top", fontsize=5)
    ax.set_xticks(range(len(BINS)), BINS)
    ax.set_xlabel("bases that differ between the two top targets")
    ax.set_ylim(0, 1.15)
for i in (0, 1, 3):
    lo, hi = min(a.get_ylim()[0] for a in axs[i]), max(a.get_ylim()[1] for a in axs[i])
    for a in axs[i]:
        a.set_ylim(lo, hi)
    for a in axs[i, 1:]:
        a.tick_params(labelleft=False)
axs[0, 0].set_ylabel("share of anchors")
axs[1, 0].set_ylabel("peptides, density\n(1 = even spread)")
axs[2, 0].set_ylabel("genes reached")
axs[3, 0].set_ylabel("share moving the same\nway in every pair")
for i, l in enumerate("abcd"):
    pf.panel_label(axs[i, 0], l)
handles = [pf.plt.Rectangle((0, 0), 1, 1, color=COLOUR[s], label=SETS[s]) for s in SETS]
handles += [pf.plt.Rectangle((0, 0), 1, 1, facecolor="black", label="c: reached by a gene-unique peptide"),
            pf.plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//////",
                             label="c: reached only by peptides shared with other genes"),
            pf.plt.Line2D([], [], color="black", linewidth=0.6, linestyle=":", label="b: even spread; d: expected by chance")]
fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False)
pf.save(fig, out_stem)
print("wrote", out_stem)
