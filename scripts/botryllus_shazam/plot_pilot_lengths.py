"""Plot checked pilot sequence lengths at publication size."""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf


def main(pilot, output):
    with (pilot / "results/compactors.tsv").open() as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    parents = {r["parent_id"] for r in records}
    dna = [len(r["compactor"]) for r in records]
    terminal = [len(r["compactor"]) for r in records if r["id"] not in parents]
    peptides = [
        len(r.seq) for r in SeqIO.parse(pilot / "results/peptides.fasta", "fasta")
    ]
    pf.use_style()
    fig, axes = pf.figure(pf.TWO_COLUMN_MM, 65, ncols=2)

    def curve(ax, values, color, label, **style):
        x, counts = np.unique(values, return_counts=True)
        y = np.cumsum(counts) / len(values) * 100
        assert int(counts.sum()) == len(values) and y[-1] == 100
        ax.step(
            np.r_[0, x], np.r_[0, y], where="post", color=color, label=label, **style
        )

    curve(
        axes[0],
        dna,
        pf.OKABE_ITO["blue"],
        f"All emitted DNA records (n={len(dna)})",
        linewidth=3,
        alpha=0.45,
    )
    curve(
        axes[0],
        terminal,
        pf.OKABE_ITO["vermillion"],
        f"Terminal DNA records (n={len(terminal)})",
        linestyle="--",
    )
    curve(
        axes[1],
        peptides,
        pf.OKABE_ITO["bluish_green"],
        f"Unique peptide fragments (n={len(peptides)})",
    )
    axes[0].set(
        xlabel="DNA length (nt)",
        ylabel="Records at or below length (%)",
        ylim=(0, 102),
        xlim=(0, max(dna) * 1.03),
    )
    axes[1].set(
        xlabel="Peptide length (aa)",
        ylabel="Sequences at or below length (%)",
        ylim=(0, 102),
        xlim=(0, max(peptides) * 1.03),
    )
    for ax, letter in zip(axes, "ab"):
        pf.panel_label(ax, letter, dx_pt=-10)
    pf.shared_legend(fig, ncol=1)
    # Checklist: units, three distinct meanings, legend first, all lengths retained.
    # This is a cumulative distribution; the step at each observed length is exact.
    pf.save(fig, output)
    svg = output.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    with Image.open(output.with_suffix(".png")) as image:
        image.convert("RGB").quantize(colors=256).save(output.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pilot", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.pilot, a.output)
