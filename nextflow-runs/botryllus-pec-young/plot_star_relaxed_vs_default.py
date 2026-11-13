"""Figure: which STAR length rule gives per-library counts closest to the paper's?

Inputs:
  inputs/count_split_libraries_length20.tsv  read pairs counted for a gene in each library by the
      full rnaseq-paper-genome-length20 run (STAR's default 66% rule, featureCounts pairs, strand
      ignored), next to the paper's own HTSeq count (count-split run, 2026-11-12)
  star_relaxed_vs_default_subsample.tsv      the same count on the first 500,000 pairs of each
      library with STAR's default rule and with a 30-base minimum instead
      (star_relaxed_vs_default.py, Sherlock job 52189672, 2026-11-13)
The 30-base estimate for a whole library is its full default count times relaxed/default on
its subsample.
Writes figures/star_relaxed_vs_default_vs_paper.{pdf,svg,png}.
Run from this folder: python plot_star_relaxed_vs_default.py
"""

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import numpy as np  # noqa: E402
import pubfig as pf  # noqa: E402

COLOUR = {"default": pf.OKABE_ITO["blue"], "relaxed": pf.OKABE_ITO["vermillion"]}
LABEL = {
    "default": "STAR default: match covers 66% of the pair (whole library)",
    "relaxed": "30 matching bases are enough (estimated from 500,000 pairs)",
}


def main():
    vial_to_library = {r["vial"]: r["sample"] for r in csv.DictReader(open(HERE / "samples_paper.csv"))}
    full = {}
    for r in csv.DictReader(open(HERE / "inputs/count_split_libraries_length20.tsv"), delimiter="\t"):
        full[vial_to_library[r["vial"]]] = (float(r["counted"]), float(r["paper_counted"]))
    sub = {}
    for r in csv.DictReader(open(HERE / "star_relaxed_vs_default_subsample.tsv"), delimiter="\t"):
        sub.setdefault(r["library"], {})[r["mode"]] = int(r["pairs_counted_for_a_gene"])

    ratio = {"default": {}, "relaxed": {}}
    for lib, (counted, paper) in full.items():
        gain = sub[lib]["relaxed"] / sub[lib]["default"]
        ratio["default"][lib] = counted / paper
        ratio["relaxed"][lib] = counted * gain / paper
    for key, vals in ratio.items():
        v = list(vals.values())
        print("{}: counted / paper median {:.3f}, range {:.3f}-{:.3f}, within 10% {} of {}".format(
            key, statistics.median(v), min(v), max(v), sum(abs(x - 1) <= 0.1 for x in v), len(v)))

    libs = sorted(full, key=lambda lib: ratio["relaxed"][lib])
    x = np.arange(len(libs))
    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 62)
    ax.axhspan(0.9, 1.1, color="#eeeeee", zorder=0, label="within 10% of the paper")
    ax.axhline(1, color=pf.GREY, linewidth=0.5, zorder=1)
    for key in ("default", "relaxed"):
        ax.scatter(x, [ratio[key][lib] for lib in libs], s=8, color=COLOUR[key], label=LABEL[key],
                   zorder=2)
    ax.set_xticks([])
    ax.set_xlabel("Library (38, sorted by the 30-base estimate)")
    ax.set_ylabel("Read pairs counted for a gene,\nthis run / the paper")
    ax.set_ylim(0.8, 1.8)
    pf.shared_legend(fig, ncol=1)
    pf.save(fig, str(HERE / "figures/star_relaxed_vs_default_vs_paper"), formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    main()
