"""Figure: for pairs STAR calls "too short", does one read align, or neither?

Reads star_too_short_test.tsv (written by star_too_short_test.py, Sherlock job 52078147,
2026-11-12: first 500,000 trimmed pairs of each library, 20-base trimming, botznik-chr.fa).
Writes figures/star_too_short_reads_alone.{pdf,svg,png}.
Run from this folder: python plot_star_too_short.py
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

GROUPS = [
    ("pairs: both reads align alone", "both reads align alone", pf.OKABE_ITO["blue"]),
    ("pairs: only read 1 aligns alone", "only read 1 aligns alone", pf.OKABE_ITO["sky_blue"]),
    ("pairs: only read 2 aligns alone", "only read 2 aligns alone", pf.OKABE_ITO["orange"]),
    ("pairs: neither read aligns alone", "neither read aligns alone", pf.GREY),
]


def main():
    values, paired_short = {}, {}
    with open(HERE / "star_too_short_test.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["mode"] == "mate1+mate2":
                values[(row["sample"], row["measure"])] = float(row["value"].split("(")[1].rstrip("%)"))
            if row["mode"] == "paired" and row["measure"] == "% of reads unmapped: too short":
                paired_short[row["sample"]] = row["value"]
    samples = sorted(paired_short, key=lambda s: float(paired_short[s].rstrip("%")))

    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 40)
    for y, sample in enumerate(samples):
        left = 0.0
        for key, label, colour in GROUPS:
            width = values[(sample, key)]
            ax.barh(y, width, left=left, color=colour, label=label if y == 0 else None, height=0.6)
            left += width
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels(["{}\n{} of pairs too short".format(s, paired_short[s]) for s in samples])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Read pairs, by which read aligns on its own (%)")
    pf.shared_legend(fig, ncol=2)
    pf.save(fig, str(HERE / "figures/star_too_short_reads_alone"), formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    main()
