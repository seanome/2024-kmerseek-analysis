"""Draw each series' probe lengths next to the k-mer lengths the analysis uses.

Usage: plot_probe_lengths.py <probe_lengths.tsv> <path to the repository's notebooks folder>
Each dot is one probe length in one series, labelled with its probe count. Vertical lines
mark the k-mer lengths: KaMRaT's 25 letters, SPLASH's 15-letter anchor, and 21 letters (7 amino acids) for
landing probes on proteins. Writes probe_lengths.png and .svg next to the table.
"""

import csv
import sys

table, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

PLATFORM = {"GSE104079": "Agilent", "GSE111555": "HTA 2.0", "GSE27536": "U133 Plus 2", "GSE28998": "U133 Plus 2",
            "GSE83578": "U219"}
with open(table) as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
series = sorted({r["gse"] for r in rows})[::-1]
pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
marks = [(15, pf.OKABE_ITO["orange"], "--", "SPLASH anchor, 15 nt"),
         (21, pf.OKABE_ITO["vermillion"], ":", "Landing on proteins, 7 aa = 21 nt"),
         (25, pf.OKABE_ITO["blue"], "-", "KaMRaT k-mer, 25 nt")]
for x, colour, style, label in marks:
    ax.axvline(x, color=colour, linestyle=style, label=label, zorder=1)
ax.plot([], [], "o", color="black", label="Probe length in one series")
for y, gse in enumerate(series):
    for r in (r for r in rows if r["gse"] == gse):
        length, n = int(r["probe_length"]), int(r["probes"])
        ax.plot(length, y, "o", color="black", zorder=3)
        each = int(r["kmers_per_probe"])
        text = f"{n:,} probes,\n{each} k-mer{'s' if each > 1 else ''} each"
        right = length < 40
        ax.text(length + (2 if right else -2), y, text, va="center", ha="left" if right else "right")
ax.set_yticks(range(len(series)), [f"{g} ({PLATFORM[g]})" for g in series])
ax.set_ylim(-0.6, len(series) - 0.4)
ax.set_xlim(10, 64)
ax.set_xticks([15, 25, 40, 60])
ax.set_xlabel("Length (nucleotides)")
pf.shared_legend(fig, ncol=2)
pf.save(fig, table.rsplit("/", 1)[0] + "/probe_lengths")
