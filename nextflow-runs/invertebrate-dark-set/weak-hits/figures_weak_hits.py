"""The two weak-hit figures, drawn from summarize.py's table. Test half only, so the
kmerseek setting picked on the tune half is scored on proteins it was not picked on, and
every tool is scored on the same proteins."""
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D

import pubfig as pf

SPECIES = ["human", "worm", "yeast", "borrelia"]
SPECIES_LABEL = {"human": "Human", "worm": "C. elegans", "yeast": "S. cerevisiae",
                 "borrelia": "B. burgdorferi"}
ITALIC = {"human": "normal", "worm": "italic", "yeast": "italic", "borrelia": "italic"}
BANDS = ["le_0.001", "0.001_1", "1_10"]
BAND_TICK = ["≤ 0.001", "0.001–1", "1–10"]
TOOLS = ["phmmer", "jackhmmer", "mmseqs2", "kmerseek"]
TOOL_LABEL = {"phmmer": "phmmer", "jackhmmer": "jackhmmer, 3 rounds",
              "mmseqs2": "MMseqs2, 3 rounds", "kmerseek": "kmerseek, setting picked on tune half"}
COLOR = {"phmmer": pf.OKABE_ITO["blue"], "jackhmmer": pf.OKABE_ITO["orange"],
         "mmseqs2": pf.OKABE_ITO["bluish_green"], "kmerseek": pf.OKABE_ITO["vermillion"]}
MARKER = {"phmmer": "o", "jackhmmer": "s", "mmseqs2": "D", "kmerseek": "^"}
FEW = "#cccccc"  # fewer than 30 judged hits
MIN_N = 30


def _row(t, sp, src, qset, band):
    r = t.filter((pl.col("species") == sp) & (pl.col("source") == src)
                 & (pl.col("query_set") == qset) & (pl.col("split") == "test")
                 & (pl.col("band") == band))
    return r.row(0, named=True) if r.height else None


def _point(ax, x, y, r, tool, dy=0.0, horizontal=True):
    """One precision with its 95% interval; grey when fewer than 30 hits were judged."""
    n = r["n_judged"]
    p = 100 * r["right_strict"] / n
    lo, hi = 100 * r["strict_lo"], 100 * r["strict_hi"]
    c = COLOR[tool] if n >= MIN_N else FEW
    if horizontal:
        ax.plot([lo, hi], [y + dy, y + dy], color=c, lw=1, zorder=2)
        ax.plot(p, y + dy, MARKER[tool], color=c, ms=3.5, zorder=3)
    else:
        ax.plot([x + dy, x + dy], [lo, hi], color=c, lw=1, zorder=2)
        ax.plot(x + dy, p, MARKER[tool], color=c, ms=3.5, zorder=3)
    return p, hi


def draw(allp, chosen, outdir):
    pf.use_style()
    outdir = Path(outdir)
    saved = []

    # Figure 1: dark proteins, best hit at E 1-10, test half
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 66, ncols=4, sharey=True, sharex=True)
    xmax = 5.0
    for ax, sp in zip(axs, SPECIES):
        k = allp.filter((pl.col("species") == sp) & (pl.col("query_set") == "dark_1")
                        & (pl.col("split") == "test") & (pl.col("band") == "1_10")
                        & pl.col("source").str.starts_with("kmerseek:")
                        & (pl.col("n_judged") >= MIN_N))
        ys = list(range(len(TOOLS)))[::-1]
        for y in ys:
            ax.axhline(y, color="#e6e6e6", lw=0.3, zorder=0)
        # every other kmerseek setting with >= 30 judged hits, as small grey dots
        others = k.filter(pl.col("source") != chosen.get(sp, ""))
        ax.plot(100 * others["precision_strict"].to_numpy(), [ys[3] - 0.25] * others.height,
                "|", color=pf.GREY, ms=4, alpha=0.7, zorder=1)
        for tool, y in zip(TOOLS, ys):
            src = chosen.get(sp) if tool == "kmerseek" else tool
            r = _row(allp, sp, src, "dark_1", "1_10") if src else None
            if r is None or r["n_judged"] == 0:
                ax.text(0.5, y, "no judged hits" if src else "no setting with 30 hits",
                        va="center", fontsize=6, color=pf.GREY)
                continue
            p, hi = _point(ax, None, y, r, tool)
            xmax = max(xmax, hi)
            ax.annotate(f"{r['right_strict']}/{r['n_judged']}", (hi, y), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=5.5)
        if others.height:
            xmax = max(xmax, 100 * others["precision_strict"].max())
        ax.set_title(SPECIES_LABEL[sp], fontsize=7, fontstyle=ITALIC[sp])
        ax.set_xlabel("Dark proteins: best hits\nat E 1–10 that are right (%)")
        ax.set_yticks(ys)
        ax.set_yticklabels([TOOL_LABEL[t].split(",")[0] for t in TOOLS])
        ax.set_ylim(-0.7, len(TOOLS) - 0.4)
    for ax in axs:
        ax.set_xlim(0, xmax * 1.25)
    handles = [Line2D([], [], marker=MARKER[t], color=COLOR[t], lw=1, ms=3.5, label=TOOL_LABEL[t])
               for t in TOOLS]
    handles += [Line2D([], [], marker="|", color=pf.GREY, lw=0, ms=5,
                       label="other kmerseek settings (≥ 30 judged)"),
                Line2D([], [], marker="o", color=FEW, lw=1, ms=3.5, label="fewer than 30 judged"),
                Line2D([], [], color="black", lw=1, label="line: 95% interval; n = right/judged, test half")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    for ax, letter in zip(axs, "abcd"):
        pf.panel_label(ax, letter)
    stem = outdir / "weak_hits_dark_e1_10_test_half"
    pf.save(fig, stem)
    saved += [stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    plt.close(fig)

    # Figure 2: every band, dark and placed, test half
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 100, ncols=4, nrows=2, sharey=True)
    for row, (qset, qlab) in enumerate((("dark_1", "Dark proteins"), ("placed_1", "Placed proteins"))):
        for col, sp in enumerate(SPECIES):
            ax = axs[row, col]
            for x in range(len(BANDS)):
                ax.axvline(x, color="#e6e6e6", lw=0.3, zorder=0)
            for i, tool in enumerate(TOOLS):
                src = chosen.get(sp) if tool == "kmerseek" else tool
                if not src:
                    continue
                for x, band in enumerate(BANDS):
                    r = _row(allp, sp, src, qset, band)
                    if r is None or r["n_judged"] == 0:
                        continue
                    _point(ax, x, None, r, tool, dy=(i - 1.5) * 0.14, horizontal=False)
            ax.set_xticks(range(len(BANDS)))
            ax.set_xticklabels(BAND_TICK)
            ax.set_xlim(-0.5, len(BANDS) - 0.5)
            ax.set_ylim(0, 100)
            if row == 0:
                ax.set_title(SPECIES_LABEL[sp], fontsize=7, fontstyle=ITALIC[sp])
            if row == 1:
                ax.set_xlabel("E-value of the best hit")
            if col == 0:
                ax.set_ylabel(f"{qlab}\nbest hits right (%)")
    h2 = handles[:4] + [handles[5], Line2D([], [], color="black", lw=1,
                                           label="line: 95% interval; test half of the proteins")]
    fig.legend(handles=h2, loc="outside upper center", ncol=3)
    for ax, letter in zip(axs.flat, "abcdefgh"):
        pf.panel_label(ax, letter)
    stem = outdir / "weak_hits_by_band_test_half"
    pf.save(fig, stem)
    saved += [stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    plt.close(fig)
    return saved
