"""Draw subset_null_summary.csv: the C2H2 ratio against 200 random 32-protein sets of each kind.

Run: python subset_null_figure.py   (reads subset_null_summary.csv beside this file)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import shrink_png  # noqa: E402

s = pl.read_csv(HERE / "subset_null_summary.csv")
obs = dict(
    s.filter(pl.col("cut").str.starts_with("observed"))
    .select("cut", "typical_ratio")
    .iter_rows()
)
ROWS = [
    ("any", "Any 32 query proteins"),
    ("short", "32 proteins with a 16-30 aa\ndomain, not C2H2"),
]

pubfig.use_style()
fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=45)
rng = np.random.default_rng(0)
for i, (key, _) in enumerate(ROWS):
    vals = s.filter(pl.col("cut").str.starts_with(f"random_{key}_"))[
        "typical_ratio"
    ].to_numpy()
    y = len(ROWS) - 1 - i + rng.uniform(-0.15, 0.15, len(vals))
    ax.scatter(
        vals,
        y,
        s=4,
        color=pubfig.GREY,
        alpha=0.6,
        linewidths=0,
        label="Random set (200 of each kind)" if i == 0 else None,
    )
ax.axvline(
    obs["observed_c2h2"],
    color=pubfig.OKABE_ITO["vermillion"],
    lw=1,
    label=f"C2H2 zinc fingers ({obs['observed_c2h2']:.2f})",
)
ax.axvline(1, color="black", lw=0.5)
ax.set_yticks(range(len(ROWS)), [r[1] for r in reversed(ROWS)])
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlim(0, 3)
ax.set_xlabel("kmerseek / MMseqs2 Fmax, typical setting\n(above 1 = kmerseek better)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
fig.savefig(HERE / "subset_null.png", dpi=300)
shrink_png.process(HERE / "subset_null.png", check=False)
fig.savefig(HERE / "subset_null.pdf")
