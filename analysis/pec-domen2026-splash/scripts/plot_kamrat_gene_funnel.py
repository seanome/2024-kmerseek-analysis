#!/usr/bin/env python3
"""From list genes to genes reached, for KaMRaT + DESeq2 on the three small contrasts (notebook 361).

Usage: plot_kamrat_gene_funnel.py <reads_dir> <out_stem>

reads_dir holds kmers_in_genes.tsv and reads_vs_contigs.tsv. Writes <out_stem>.tsv and prints it. A
swap's average is taken within each contrast and then summed, as in plot_kamrat_reads.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

reads_dir, out_stem = Path(sys.argv[1]), sys.argv[2]
k = pd.read_csv(reads_dir / "kmers_in_genes.tsv", sep="\t").query("group == 'list' and route == 'kamrat_deseq2_p05'")
t = pd.read_csv(reads_dir / "reads_vs_contigs.tsv", sep="\t").set_index("channel")
real = k[k.run == "real"]
swaps = k[k.run != "real"].groupby(["contrast", "run"]).sum(numeric_only=True).groupby("contrast").mean().sum()
rows = pd.DataFrame([("list genes with a protein", real.genes.sum(), None),
                     ("a k-mer in KaMRaT's table", real.in_table.sum(), None),
                     ("a k-mer past the test, either way", real.kept_either_way.sum(), swaps.kept_either_way),
                     ("a k-mer past the test, same way as the list", real.kept_same_way.sum(), swaps.kept_same_way),
                     ("reached by contigs, same way", t.reached["kamrat_deseq2_p05"], t.swap_mean["kamrat_deseq2_p05"]),
                     ("reached by reads, same way", t.reached["kamrat_deseq2_p05_reads"], t.swap_mean["kamrat_deseq2_p05_reads"])],
                    columns=["step", "real_labels", "swapped_labels_average"])
rows.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(rows.to_string(index=False))

pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM * 1.4, 52, layout="constrained")
col = pf.OKABE_ITO["reddish_purple"]
ys = list(range(len(rows)))[::-1]
for y, r in zip(ys, rows.itertuples()):
    ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
    ax.plot(r.real_labels, y, "o", color=col, markersize=4, zorder=3)
    if r.swapped_labels_average == r.swapped_labels_average:
        ax.plot(r.swapped_labels_average, y, "o", markerfacecolor="white", markeredgecolor=col, markeredgewidth=0.7, markersize=4, zorder=2)
    ax.annotate(f"{r.real_labels:.0f}", (r.real_labels, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5, color=col)
ax.set_yticks(ys, rows.step)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlim(0, 200)
ax.set_xlabel("list genes, of 188")
h = [pf.plt.Line2D([], [], marker="o", color=col, linestyle="none", markersize=4, label="real labels"),
     pf.plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=col, linestyle="none", markersize=4, label="swapped labels, average")]
fig.legend(handles=h, loc="outside upper center", ncols=2, frameon=False, fontsize=5)
pf.save(fig, out_stem)
print("wrote", out_stem)
