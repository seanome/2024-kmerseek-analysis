"""Draw how long KaMRaT's merged contigs are in each comparison, one panel per comparison.

Usage: plot_contig_lengths.py <contig_lengths.tsv> <path to the repository's notebooks folder>
Each dot is one contig length and how many contigs had it. The dashed line is the 25-letter k-mer, the shortest
possible contig. SPLASH gives no contigs here, because it ran without its compactor step. Writes contig_lengths.png
and .svg next to the table.
"""

import csv
import sys
from collections import defaultdict

table, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

ARRAY = {
    "GSE104079": "Agilent, 60 nt probes",
    "GSE111555": "HTA 2.0",
    "GSE27536": "U133 Plus 2",
    "GSE28998": "U133 Plus 2",
    "GSE83578": "U219",
}
counts = defaultdict(dict)
with open(table) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        counts[r["comparison"]][int(r["contig_length"])] = int(r["contigs"])
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 120, nrows=len(counts), sharex=True)
axs[0].plot([], [], "o", color=pf.OKABE_ITO["blue"], label="Contigs of that length")
axs[0].axvline(25, color=pf.GREY, linestyle="--", label="25 nt, one k-mer: the shortest contig")
for ax, (name, by_length) in zip(axs, sorted(counts.items()), strict=True):
    ax.axvline(25, color=pf.GREY, linestyle="--")
    ax.plot(list(by_length), list(by_length.values()), "o", color=pf.OKABE_ITO["blue"], markersize=2.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 20000)
    ax.set_yticks([1, 100, 10000], ["1", "100", "10000"])
    ax.minorticks_off()
    total = sum(by_length.values())
    single = by_length.get(25, 0)
    gse = name.split(".")[0]
    ax.text(
        1.0,
        0.95,
        f"{name.replace('.main', '').replace('.', ', ')} ({ARRAY[gse]}): {total} contigs, "
        f"{single} of them one k-mer; longest {max(by_length)} nt",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
axs[len(axs) // 2].set_ylabel("Contigs (each step is 100 times the last)")
axs[-1].set_xlabel("Contig length (nucleotides), from the 10000 top k-mers of each comparison")
axs[-1].set_xlim(20, 90)
pf.shared_legend(fig, ncol=2)
pf.save(fig, table.rsplit("/", 1)[0] + "/contig_lengths")
