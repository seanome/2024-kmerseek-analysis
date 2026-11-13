"""Plot which DESeq2 design gives back the paper's p-values (paper_design.out).

Usage: plot_paper_design.py <paper_design.out> <figure path without extension>
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

LABELS = {1: "2 h", 2: "24 h", 4: "13-49 d", 5: "132 d"}


def main(out, fig_path):
    rows = re.findall(
        r"group (\d) (~ [a-z +]+?)\s+genes (\d+) spearman_p ([\d.]+)", open(out).read()
    )
    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
    order = [1, 2, 4, 5]
    styles = {
        "~ treat": ("treatment only", pf.OKABE_ITO["blue"], "o"),
        "~ colony + treat": ("colony + treatment", pf.OKABE_ITO["vermillion"], "s"),
    }
    for design, (label, colour, marker) in styles.items():
        pts = {int(g): float(r) for g, d, _, r in rows if d.strip() == design}
        ax.scatter(
            range(len(order)), [pts[g] for g in order], color=colour, marker=marker,
            label=label, zorder=3,
        )
    ax.set_xticks(range(len(order)), [LABELS[g] for g in order])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Paper's young-colony comparison")
    ax.set_ylabel("Rank correlation with\nthe paper's p-values")
    ax.legend(
        title="DESeq2 design, refitted on the paper's own counts", loc="lower left",
        bbox_to_anchor=(0, 1.02), ncol=2, alignment="left",
    )
    pf.save(fig, fig_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
