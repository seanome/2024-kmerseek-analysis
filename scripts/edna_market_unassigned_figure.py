"""Share of eDNA sequences not named at genus, five marine studies (2023-2025), one row per marker.

Reads docs/edna_market/unassigned_at_genus_studies.tsv and writes
figures/edna_market_unassigned_at_genus_marine_studies.{pdf,png}.
The pass rule (fixed before the search): below 10% in every study means the gap
is too small to sell.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
import pubfig as pf  # noqa: E402

KILL_PCT = 10
COLOUR = {
    "yes": pf.OKABE_ITO["vermillion"],  # COI: a protein-coding gene
    "no": pf.OKABE_ITO["blue"],  # 12S rRNA: no protein
    "mixed": pf.GREY,  # COI, 18S and 16S reported together
}


def main():
    rows = list(csv.DictReader(open(REPO / "docs/edna_market/unassigned_at_genus_studies.tsv"), delimiter="\t"))
    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 72)
    ax.axvspan(0, KILL_PCT, color="#eeeeee", zorder=0)
    ax.text(KILL_PCT / 2, len(rows) - 0.35, "below 10%:\ngap too small,\nanswer No", ha="center", va="top", fontsize=5)

    labels = []
    for i, r in enumerate(reversed(rows)):
        y = i
        c = COLOUR[r["marker_codes_protein"]]
        ax.axhline(y, color="#dddddd", lw=0.3, zorder=0)
        share = r["share_not_at_genus_pct"]
        if " to " in share:
            lo, hi = (float(v) for v in share.split(" to "))
            ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="butt", zorder=2)
            ax.annotate("", xy=(hi + 8, y), xytext=(hi, y), arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
            ax.text(hi + 9, y, f"{lo:g}-{hi:g}% (at least; by marker)", va="center", fontsize=5.5)
        elif r["bound"] == "at_least":
            v = float(share)
            ax.plot(v, y, marker=">", ms=5, mfc="white", mec=c, mew=1, ls="none", zorder=3)
            ax.annotate("", xy=(v + 8, y), xytext=(v + 1.5, y), arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
            ax.text(v + 9, y, f"{v:g}% (at least)", va="center", fontsize=5.5)
        else:
            v = float(share)
            ax.plot(v, y, marker="o", ms=5, color=c, ls="none", zorder=3)
            ax.text(v + 2.5, y, f"{v:g}%", va="center", fontsize=5.5)
        labels.append(f"{r['study']} {r['year']}\n{r['marker']}; {r['unit']}, n = {r['n_units']}")

    ax.set_yticks(range(len(rows)), labels, fontsize=5.5)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Sequences (or taxa) not named at genus (%)")
    ax.grid(False)

    handles = [
        Patch(color="#eeeeee", label="Rule: below 10% in every study means No"),
        Line2D([], [], marker="o", color="k", ls="none", ms=5, label="Share reported at genus"),
        Line2D(
            [],
            [],
            marker=">",
            mfc="white",
            mec="k",
            ls="none",
            ms=5,
            label="Share unnamed at a higher rank (genus share is at least this)",
        ),
        Line2D([], [], marker="s", color=COLOUR["yes"], ls="none", ms=5, label="COI: protein-coding gene"),
        Line2D([], [], marker="s", color=COLOUR["no"], ls="none", ms=5, label="rRNA gene (12S, 16S, 18S): codes no protein"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=3,
        fontsize=5.5,
        frameon=False,
        borderaxespad=0,
    )
    ax.set_title(
        "Every marine eDNA study found leaves more than 10% unnamed at genus; the largest gaps are in COI, the one marker that codes a protein",
        fontsize=6.5,
        loc="left",
        pad=26,
    )
    pf.save(fig, REPO / "figures/edna_market_unassigned_at_genus_marine_studies", formats=("pdf", "png"))


if __name__ == "__main__":
    main()
