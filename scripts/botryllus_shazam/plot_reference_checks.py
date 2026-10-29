"""Plot the recorded genome/GTF reconstruction checks without changing their inputs."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import pubfig as pf
from PIL import Image

folder = ROOT / "reports/botryllus-shazam/production-launch"
data = json.loads((folder / "reference-verified-summary.json").read_text())
audit = data["coordinate_audit"]
pf.use_style()
fig, axes = pf.figure(183, 65, ncols=2)
for ax, kind, label in zip(
    axes, ["exons", "CDS-plus-stop"], ["Full exons", "Coding sequence plus stop codon"]
):
    counts = [audit["checks"][f"{kind}:{offset}"]["exact"] for offset in [-1, 0, 1]]
    values = [100 * n / audit["transcripts"] for n in counts]
    ax.scatter(
        values,
        [2, 1, 0],
        color=pf.OKABE_ITO["blue"],
        label="Exact sequence reconstruction",
    )
    for y, value, count in zip([2, 1, 0], values, counts):
        ax.annotate(
            f'{count:,} / {audit["transcripts"]:,}',
            (value, y),
            xytext=(-4 if value > 70 else 4, 6),
            textcoords="offset points",
            ha="right" if value > 70 else "left",
        )
    ax.set_yticks([2, 1, 0], ["1 base earlier", "Original GTF", "1 base later"])
    ax.set_ylim(-0.4, 2.5)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Reference sequences reconstructed (%)")
    ax.set_title(label)
    ax.grid(axis="y")
pf.shared_legend(fig)
# Checklist: six observed points, one colour/shape, legend before marks, all labels in axes.
# Denominator is the complete reference, with no missing values dropped.
assert sum(len(ax.collections[0].get_offsets()) for ax in axes) == 6
pf.save(fig, folder / "genome-gtf-reference-reconstruction")
svg = folder / "genome-gtf-reference-reconstruction.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
png = folder / "genome-gtf-reference-reconstruction.png"
with Image.open(png) as im:
    im.convert("RGB").quantize(colors=256).save(png)
