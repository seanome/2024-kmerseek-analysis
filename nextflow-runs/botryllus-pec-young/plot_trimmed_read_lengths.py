"""Figure: read lengths after Trim Galore, with a 62-base minimum (PR 203 reads) and a 20-base one.

Inputs (pulled from Sherlock on 2026-11-12, the 38 libraries of the paper-genome runs):
  inputs/trimmed_read_lengths_fastqc.tsv  FastQC "Sequence Length Distribution" of each trimmed
      FASTQ. 62: /scratch/users/olgabot/botryllus-shazam-PEC-20261023/nfcore/results/fastqc/trim
      20: /scratch/users/olgabot/botryllus-pec-young/rnaseq-paper-genome-length20/fastqc/trim
  inputs/trimgalore_pairs_removed.tsv     "Number of sequence pairs removed because at least one
      read was shorter than the length cutoff", from each library's read-2 trimming report
      in the trimgalore/ folder next to the fastqc/ folder above.
Writes figures/trimmed_read_lengths_trimgalore.{pdf,svg,png}.
Run from this folder: python plot_trimmed_read_lengths.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import numpy as np  # noqa: E402
import pubfig as pf  # noqa: E402

COLOUR = {"62": pf.OKABE_ITO["blue"], "20": pf.OKABE_ITO["vermillion"]}
LABEL = {
    "62": "minimum 62 bases (PR 203 reads, used by SPLASH)",
    "20": "minimum 20 bases (Trim Galore default)",
}


def bin_start(text):
    return int(text.split("-")[0])


def bin_width(text):
    lo, _, hi = text.partition("-")
    return int(hi or lo) - int(lo) + 1


def main():
    # Pooled over libraries and both reads, as reads at each length. FastQC picks its bin
    # widths per file (1, 2 or 5 bases), so each bin's reads are spread evenly over its bases
    # before the files are added up.
    pooled = defaultdict(lambda: defaultdict(float))
    with open(HERE / "inputs/trimmed_read_lengths_fastqc.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            start, width = bin_start(row["length_bin"]), bin_width(row["length_bin"])
            for length in range(start, start + width):
                pooled[row["min_length"]][length] += float(row["reads"]) / width

    removed = defaultdict(dict)
    with open(HERE / "inputs/trimgalore_pairs_removed.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            removed[row["min_length"]][row["library"]] = float(row["pairs_removed_pct"])

    pf.use_style()
    fig, (ax_a, ax_b) = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)

    # a: length distribution
    for key in ("62", "20"):
        starts = np.array(sorted(pooled[key]))
        reads = np.array([pooled[key][s] for s in starts])
        pct = 100 * reads / sum(pooled[key].values())
        ax_a.step(starts, pct, where="post", color=COLOUR[key], label=LABEL[key])
        print("min {}: {:.0f} M reads kept, shortest {} bases".format(
            key, sum(pooled[key].values()) / 1e6, starts.min()))
    for cut in (20, 62):
        ax_a.axvline(cut, color=pf.GREY, linestyle=":", linewidth=0.6)
        ax_a.text(cut + 1.5, ax_a.get_ylim()[1] * 0.97, "{} bases".format(cut), va="top", fontsize=6)
    ax_a.set_xlabel("Read length after trimming (bases)")
    ax_a.set_ylabel("Reads per base of length (% of reads kept)")
    ax_a.set_xlim(0, None)

    # b: pairs removed per library, sorted by the 62-base loss
    libs = sorted(removed["62"], key=lambda lib: removed["62"][lib])
    x = np.arange(len(libs))
    for i, lib in enumerate(libs):
        ax_b.plot([i, i], [removed["20"][lib], removed["62"][lib]], color=pf.GREY, linewidth=0.5,
                  zorder=1)
    for key in ("62", "20"):
        ax_b.scatter(x, [removed[key][lib] for lib in libs], color=COLOUR[key], s=8, zorder=2)
        vals = np.array([removed[key][lib] for lib in libs])
        print("min {}: pairs removed per library median {:.1f}%, range {:.1f}-{:.1f}%".format(
            key, np.median(vals), vals.min(), vals.max()))
    ax_b.set_xticks([])
    ax_b.set_xlabel("Library (38, sorted by pairs removed at 62 bases)")
    ax_b.set_ylabel("Read pairs removed by trimming (%)")
    ax_b.set_ylim(0, 100)

    pf.shared_legend(fig)
    for ax, letter in zip((ax_a, ax_b), "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, str(HERE / "figures/trimmed_read_lengths_trimgalore"), formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    main()
