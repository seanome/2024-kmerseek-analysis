"""Draw how many probe sets hold each 30-letter HP k-mer, for each array and HP alphabet.

Usage: plot_hp_sharing.py <hp folder> <path to the repository's notebooks folder>
Reads <array>.<alphabet>.sharing.tsv and .summary.tsv. One panel per array; each dot is a number of probe sets and how
many k-mers are held by exactly that many. Both axes step by a factor of 10. Writes hp_sharing.png and .svg there.
"""

import csv
import sys
from pathlib import Path

folder, notebooks = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

ARRAYS = ["U133_Plus_2", "HTA_2.0"]
ALPHABETS = {"hp_lehninger2": pf.OKABE_ITO["blue"], "hp_thomas_dill2": pf.OKABE_ITO["vermillion"]}
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 60, ncols=2, sharey=True)
for alphabet, colour in ALPHABETS.items():
    axs[0].plot([], [], "o", color=colour, label=alphabet)
for ax, array in zip(axs, ARRAYS, strict=True):
    lines = []
    for alphabet, colour in ALPHABETS.items():
        with open(folder / f"{array}.{alphabet}.sharing.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        ax.plot(
            [int(r["probe_sets_sharing_a_kmer"]) for r in rows],
            [int(r["kmers"]) for r in rows],
            "o",
            color=colour,
            markersize=2,
            alpha=0.8,
        )
        with open(folder / f"{array}.{alphabet}.summary.tsv") as f:
            s = next(csv.DictReader(f, delimiter="\t"))
        share = int(s["kmers_in_one_probe_set"]) / int(s["distinct_kmers"])
        lines.append(
            f"{alphabet}: {int(s['distinct_kmers']):,} k-mers from {int(s['probe_sets_with_a_kmer']):,} "
            f"probe sets;\n{share:.1%} in one probe set, most shared in {s['most_probe_sets']}"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [1, 10, 100, 1000, 10000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks([1, 100, 10000, 1000000, 100000000], ["1", "100", "10000", "1000000", "100000000"])
    ax.minorticks_off()
    ax.set_xlabel("Probe sets holding the k-mer")
    ax.set_title(array.replace("_", " ") + f", {s['probe_sets_with_a_kmer']} probe sets with a protein")
axs[0].set_ylabel("30-letter HP k-mers")
pf.shared_legend(fig, ncol=2)
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, str(folder / "hp_sharing"))
