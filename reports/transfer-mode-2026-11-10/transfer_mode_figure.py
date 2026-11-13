"""Draw where transfer_domains puts six zinc fingers on the query, in each mode.

The example is the one in tests/test_transfer_mode.py: an alignment of query [50, 190)
to target [195, 335) over six 23-residue C2H2 fingers starting at target 200.
Run from the repo root: python reports/transfer-mode-2026-11-10/transfer_mode_figure.py
"""

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin"))
import pubfig as pf  # noqa: E402
import evaluate_domain_calls as edc  # noqa: E402

FINGER, STARTS = 23, [200 + 23 * i for i in range(6)]
QS, QE, TS, TE = 50, 190, 195, 335

regions = pl.LazyFrame({"query_acc": ["Q"], "target_acc": ["T"], "qstart": [QS],
                        "qend": [QE], "tstart": [TS], "tend": [TE], "score": [1.0]})
dmap = pl.LazyFrame({"accession": ["T"] * 6, "pfam_id": ["PF00096"] * 6,
                     "domain_start": STARTS, "domain_end": [s + FINGER for s in STARTS]})
calls = {m: edc.transfer_domains(regions, dmap, 0.5, m).collect().sort("qstart")
         for m in edc.TRANSFER_MODES}
# The query's true fingers: same positions as the projected ones, for the drawing.
truth = [(s + QS - TS, s + QS - TS + FINGER) for s in STARTS]

pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 42)
ALN, CALL, TRUE = pf.OKABE_ITO["blue"], pf.OKABE_ITO["vermillion"], pf.GREY
rows = [("true fingers on the query", 3), ("aligned stretch", 2),
        ("calls, span", 1), ("calls, projected", 0)]
ax.plot([0, 230], [3, 3], color="black", lw=1)
for a, b in truth:
    ax.add_patch(pf.plt.Rectangle((a, 2.75), b - a, 0.5, fc=TRUE, ec="black", lw=0.5))
ax.add_patch(pf.plt.Rectangle((QS, 1.85), QE - QS, 0.3, fc=ALN, ec="none"))
n = calls["span"].height
for i, (a, b) in enumerate(zip(calls["span"]["qstart"], calls["span"]["qend"])):
    y = 0.65 + 0.07 * i
    ax.plot([a, b], [y, y], color=CALL, lw=1.2)
for a, b in zip(calls["projected"]["qstart"], calls["projected"]["qend"]):
    ax.add_patch(pf.plt.Rectangle((a + 0.6, -0.2), b - a - 1.2, 0.4, fc=CALL, ec="none"))
ax.set_yticks([r[1] for r in rows], [r[0] for r in rows])
ax.set_xlim(0, 230)
ax.set_ylim(-0.6, 3.6)
ax.set_xlabel("Query position (residue)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.text(QE + 3, 0.65 + 0.07 * (n - 1) / 2, f"{n} calls, each {QE - QS} aa:\nIoU with one finger 0.16",
        va="center", fontsize=5.5)
ax.text(QE + 3, 0, "6 calls, 20-23 aa", va="center", fontsize=5.5)
pf.save(fig, str(Path(__file__).with_name("transfer_mode_tandem_example")))
