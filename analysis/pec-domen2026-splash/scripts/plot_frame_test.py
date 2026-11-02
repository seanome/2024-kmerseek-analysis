#!/usr/bin/env python3
"""Frame choice tested on Botryllus reads whose true frame is known.

Usage: plot_frame_test.py <frame_test_summary.tsv> <frame_test_hist.tsv> <setting> <chosen_cutoff> <out_stem>

a, b: reads by the share of the chosen frame's k-mers found in Swiss-Prot, for reads whose
chosen frame is right and wrong; simulated reads (a) and real reads that lie wholly in a CDS (b).
c: as the cutoff rises, the share of right-frame reads kept, and the share of kept reads with
the right frame. d: per frame-choice setting, the share of reads with the right frame and the
share with no Swiss-Prot match at all.
Writes <out_stem>.tsv with the numbers drawn in c and d.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

summ_p, hist_p, setting, chosen, out_stem = sys.argv[1:]
chosen = float(chosen)
summ = pl.read_csv(summ_p, separator="\t", schema_overrides={"cutoff": pl.Float64})
hist = pl.read_csv(
    hist_p, separator="\t", schema_overrides={"bin": pl.Float64}
).with_columns(pl.col("bin").round(2))
SIM, REAL = "simulated from CDS", "real: whole read coding"
SET_COL = {"paper_list": pf.OKABE_ITO["reddish_purple"], "random": "black"}
SETTING_SHADE = [
    "#000000",
    "#666666",
    "#999999",
    "#CCCCCC",
]  # one grey per frame-choice setting
SET_NAME = {"paper_list": "paper-list genes", "random": "random genes"}

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 125, nrows=2, ncols=2)
(a, b), (c, d) = axs

for ax, src in [(a, SIM), (b, REAL)]:
    h = hist.filter((pl.col("setting") == setting) & (pl.col("source") == src))
    tot = summ.filter(
        (pl.col("setting") == setting)
        & (pl.col("source") == src)
        & (pl.col("cutoff") == 0)
    )
    for st in ["paper_list", "random"]:
        n = tot.filter(pl.col("set") == st)["reads"].sum()
        for corr, ls in [(True, "-"), (False, "--")]:
            x = h.filter((pl.col("set") == st) & (pl.col("correct") == corr)).sort(
                "bin"
            )
            full = (
                pl.DataFrame({"bin": [round(i / 20, 2) for i in range(20)]})
                .join(x.select("bin", "reads"), on="bin", how="left")
                .fill_null(0)
            )
            ax.step(
                full["bin"] + 0.05,
                100 * full["reads"] / max(n, 1),
                where="pre",
                color=SET_COL[st],
                ls=ls,
                label=(
                    f"{SET_NAME[st]}, {'right' if corr else 'wrong'} frame"
                    if ax is a
                    else None
                ),
            )
    ax.axvline(0.5, color=pf.GREY, lw=0.8, ls=":")
    ax.axvline(chosen, color=pf.GREY, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of chosen frame's k-mers in Swiss-Prot")
    ax.set_ylabel("Reads (% of the gene set's reads)")
    ax.set_title(
        "Reads cut from CDS" if ax is a else "Real reads lying wholly in a CDS",
        loc="left",
    )

rows = []
for src, marker in [(SIM, "o"), (REAL, "s")]:
    for st in ["paper_list", "random"]:
        x = summ.filter(
            (pl.col("setting") == setting)
            & (pl.col("source") == src)
            & (pl.col("set") == st)
        ).sort("cutoff")
        if x.height == 0:
            continue
        correct_all = x["correct"][0]
        kept_right = 100 * x["kept_correct"] / max(correct_all, 1)
        right_of_kept = 100 * x["kept_correct"] / x["kept"].clip(1)
        c.plot(
            x["cutoff"], kept_right, color=SET_COL[st], marker=marker, ms=2.5, ls="-"
        )
        c.plot(
            x["cutoff"],
            right_of_kept,
            color=SET_COL[st],
            marker=marker,
            ms=2.5,
            ls="--",
        )
        for cut, k1, k2 in zip(x["cutoff"], kept_right, right_of_kept):
            rows.append(
                {
                    "panel": "c",
                    "setting": setting,
                    "source": src,
                    "set": st,
                    "cutoff": cut,
                    "pct_right_frame_reads_kept": round(k1, 2),
                    "pct_kept_reads_right_frame": round(k2, 2),
                }
            )
c.axvline(0.5, color=pf.GREY, lw=0.8, ls=":")
c.axvline(chosen, color=pf.GREY, lw=0.8)
c.set_xlim(0, 1)
c.set_ylim(0, 102)
c.set_xlabel("Cutoff on share of k-mers in Swiss-Prot")
c.set_ylabel("Reads (%)")
c.plot(
    [],
    [],
    color="black",
    ls="-",
    label="right-frame reads kept (% of right-frame reads)",
)
c.plot(
    [], [], color="black", ls="--", label="kept reads with the right frame (% of kept)"
)
c.plot([], [], color="black", marker="o", ls="", ms=2.5, label="reads cut from CDS")
c.plot(
    [], [], color="black", marker="s", ls="", ms=2.5, label="real reads wholly in a CDS"
)
c.plot([], [], color=pf.GREY, ls=":", label="cutoff 0.5 (Olga's)")
c.plot([], [], color=pf.GREY, ls="-", label=f"cutoff {chosen} (chosen from this test)")
c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5)

settings = sorted(s for s in summ["setting"].unique() if s.endswith("stopfree"))
z = (
    summ.filter(pl.col("cutoff") == 0)
    .group_by("setting", "source")
    .agg(pl.col("reads").sum(), pl.col("correct").sum(), pl.col("any_match").sum())
)
srcs = [SIM, REAL, "real: partly coding", "real: no protein hit"]
w = 0.18
for i, src in enumerate(srcs):
    for j, s in enumerate(settings):
        r = z.filter((pl.col("setting") == s) & (pl.col("source") == src))
        if r.height == 0:
            continue
        r = r.row(0, named=True)
        right = 100 * r["correct"] / r["reads"]
        nomatch = 100 * (1 - r["any_match"] / r["reads"])
        xpos = i + (j - (len(settings) - 1) / 2) * w
        d.bar(
            xpos,
            right,
            width=w * 0.9,
            color=SETTING_SHADE[j % 4],
            edgecolor="black",
            lw=0.3,
            label=s if i == 0 else None,
        )
        d.plot(
            [xpos - w * 0.45, xpos + w * 0.45],
            [right + nomatch] * 2,
            color="black",
            lw=0.6,
        )
        rows.append(
            {
                "panel": "d",
                "setting": s,
                "source": src,
                "set": "all",
                "reads": r["reads"],
                "pct_right_frame": round(right, 2),
                "pct_no_swissprot_match": round(nomatch, 2),
            }
        )
d.plot(
    [],
    [],
    color="black",
    lw=0.6,
    label="bar + reads with no stop-free frame\nor no Swiss-Prot match",
)
d.set_xticks(
    range(len(srcs)),
    [
        "cut from CDS",
        "real, whole\nread in CDS",
        "real, partly\nin CDS",
        "real, no\nprotein hit",
    ],
)
d.set_ylabel("Reads with the right frame (%)")
d.set_ylim(0, 105)
d.legend(loc="upper right", fontsize=5, ncol=1)

a.legend(loc="upper right", fontsize=5)
for ax, letter in zip([a, b, c, d], "abcd"):
    pf.panel_label(ax, letter)
pl.DataFrame(rows, infer_schema_length=None).write_csv(
    f"{out_stem}.tsv", separator="\t"
)
pf.save(fig, out_stem)
