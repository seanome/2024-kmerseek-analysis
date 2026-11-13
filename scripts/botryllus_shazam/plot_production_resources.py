"""Show observed command memory and proposed first production requests."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import pubfig as pf
from PIL import Image

p = ROOT / "reports/botryllus-shazam/production-launch/followup-20261025"
rows = json.loads((p / "measured-resources.json").read_text())
pf.use_style()
fig, ax = pf.figure(183, 55)
y = list(range(len(rows)))
ax.scatter(
    [r["observed_gib"] for r in rows],
    y,
    color=pf.OKABE_ITO["blue"],
    label="Observed command memory",
)
ax.scatter(
    [r["requested_gb"] for r in rows],
    y,
    marker="s",
    facecolors="none",
    edgecolors=pf.OKABE_ITO["vermillion"],
    label="First production request",
)
for i, r in enumerate(rows):
    for key, dy in [("observed_gib", 6), ("requested_gb", -10)]:
        ax.annotate(
            f"{r[key]:.1f}", (r[key], i), xytext=(4, dy), textcoords="offset points"
        )
ax.set_yticks(y, [r["stage"] for r in rows])
ax.invert_yaxis()
ax.set_ylim(2.5, -0.5)
ax.set_xlim(0, 100)
ax.set_xlabel("Memory (GiB; Nextflow GB requests use binary units)")
ax.grid(axis="y")
pf.shared_legend(fig)
# Six marks: observed memory and request for each of three measured operations.
assert sum(len(c.get_offsets()) for c in ax.collections) == 6
pf.save(fig, p / "largest-library-memory", formats=("pdf", "png"))
with Image.open(p / "largest-library-memory.png") as im:
    im.convert("RGB").quantize(colors=256).save(p / "largest-library-memory.png")
