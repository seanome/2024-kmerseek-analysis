"""Plot measured retained pairs for every recovered library."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import pubfig as pf
import matplotlib.pyplot as plt

out = ROOT / "reports/botryllus-shazam/nextflow-migration"
rows = json.loads((out / "trimmed-inputs.validated.json").read_text())
pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 75)
x = [r["input_pairs"] / 1e6 for r in rows]
y = [r["pairs"] / 1e6 for r in rows]
ax.plot([0, max(x)], [0, max(x)], color="#999999", label="All input pairs retained")
ax.scatter(x, y, color="#0072B2", label="One validated library (n=64)", zorder=3)
ax.set(
    xlabel="Raw pairs (millions)",
    ylabel="Validated trimmed pairs (millions)",
    xlim=(0, max(x) * 1.06),
    ylim=(0, max(x) * 1.06),
)
ax.ticklabel_format(style="plain", axis="both")
fig.legend(loc="upper left", bbox_to_anchor=(0.16, 0.99), frameon=False)
fig.set_layout_engine(None)
fig.subplots_adjust(left=0.19, bottom=0.19, top=0.87, right=0.97)
# Position compares counts on shared linear axes; colour identifies libraries and reference.
# Legend precedes marks; all 64 libraries, including very small libraries, are retained.
pf.save(fig, out / "trimmed-pairs-PRJNA1219309")
plt.close(fig)
