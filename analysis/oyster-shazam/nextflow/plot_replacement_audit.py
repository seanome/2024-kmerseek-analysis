"""Plot measured corrected-sample hashes and the separate internal-X fixture."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "notebooks"))
import pubfig as pf
from matplotlib.lines import Line2D
from PIL import Image


def main():
    evidence = Path(__file__).resolve().parents[1] / "results/scheduler-20261025"
    inputs = evidence.parent / "migration-20261025/job49520265/summaries"
    build = json.loads((evidence / "query-build-completed.json").read_text())
    sample = {"collections": []}
    for i in range(6):
        rows = [chunk["validation"]["collections"][i] for chunk in build["chunks"]]
        sample["collections"].append(
            dict(
                file=rows[0]["file"],
                origins=sum(r["origins"] for r in rows),
                empty=sum(r["empty"] for r in rows),
            )
        )
    probe = json.loads((inputs / "protocol--protocol-audit.json").read_text())
    pf.use_style()
    fig, axes = pf.figure(
        pf.TWO_COLUMN_MM,
        70,
        ncols=2,
        layout=None,
        gridspec_kw={"width_ratios": [1.4, 1]},
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.77, wspace=0.45)
    colors = {10: pf.OKABE_ITO["blue"], 100: pf.OKABE_ITO["vermillion"]}
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=colors[10],
            label="Audited query set: scaled10",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color=colors[100],
            markerfacecolor="white",
            label="Audited query set: scaled100",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            color=pf.OKABE_ITO["bluish_green"],
            label="X fixture: scaled1",
        ),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3)
    fig.suptitle(
        "Branchwater 0.9.15: corrected stretches preserve X; every original ID is retained",
        y=0.99,
    )
    mols = ["protein", "dayhoff", "hp"]
    labels = ["Protein k7", "Dayhoff k12", "HP k30"]
    for y, mol in enumerate(mols):
        for scale, delta, marker in ((10, -0.13, "o"), (100, 0.13, "s")):
            row = next(
                r for r in sample["collections"] if r["file"] == f"{mol}-{scale}.sig.gz"
            )
            count = row["origins"] - row["empty"]
            fraction = count / row["origins"] * 100
            axes[0].plot(
                fraction,
                y + delta,
                marker,
                color=colors[scale],
                markerfacecolor=colors[scale] if scale == 10 else "white",
            )
            axes[0].annotate(
                f"{count:,}/{row['origins']:,}",
                (fraction, y + delta),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
            )
        row = next(r for r in probe["x_hash_probes"] if r["moltype"] == mol)
        retained = row["x_spanning_hashes_retained"]
        total = row["x_spanning_distinct_hashes"]
        axes[1].plot(retained, y, "D", color=pf.OKABE_ITO["bluish_green"])
        axes[1].annotate(
            f"{retained}/{total}",
            (retained, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
        )
    for ax in axes:
        ax.set_yticks(range(3), labels)
        ax.set_ylim(2.6, -0.6)
        ax.grid(axis="y")
    axes[0].set_xlim(0, 20)
    axes[0].set_xticks([0, 5, 10, 15, 20])
    axes[0].set_xlabel("Original queries with a nonempty sketch (%)")
    axes[1].set_xlim(0, 39)
    axes[1].set_xticks([0, 10, 20, 30])
    axes[1].set_xlabel("Distinct X-spanning hashes retained (n)")
    axes[0].set_title("17,232 SPLASH + 363,635 DE-kupl queries")
    axes[1].set_title("One 121-aa peptide containing X")
    pf.panel_label(axes[0], "a")
    pf.panel_label(axes[1], "b")
    fig.text(
        0.5,
        0.025,
        "Keep ≥30-aa stretches from six frames; split only at stops. Retained real-query segments contain no X.",
        ha="center",
    )
    # Checklist: three meanings/colours, legend first, category guides, visible data labels;
    # all six real counts and three fixture counts are read directly from committed evidence.
    pf.save(fig, evidence / "oyster_corrected_query_rebuild")
    path = evidence / "oyster_corrected_query_rebuild.png"
    with Image.open(path) as image:
        image.convert("RGB").quantize(colors=256).save(path)


if __name__ == "__main__":
    main()
