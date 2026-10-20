"""Figures and tables for notebook 356: does a search call land on one exon's residues?

Reads the outputs of scripts/score_356_exon_landing.py (tables/356/ or tables/356/pilot/).
Every figure is drawn at printed size with pubfig and the Nature style.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pubfig as pf

#: Tool -> (label, colour, marker). Three hues plus black and grey: kmerseek blue,
#: structure tools vermillion, sequence aligners bluish green, the no-search scan black,
#: tools outside the rules grey. Within a hue the marker tells tools apart.
TOOLS = {
    "kmerseek_best_of_all": (
        "kmerseek, best of all settings",
        pf.OKABE_ITO["blue"],
        "o",
    ),
    "kmerseek_best_single": (
        "kmerseek, best single setting",
        pf.OKABE_ITO["sky_blue"],
        "o",
    ),
    "foldseek": ("Foldseek", pf.OKABE_ITO["vermillion"], "s"),
    "reseek": ("Reseek", pf.OKABE_ITO["vermillion"], "^"),
    "prostt5": ("ProstT5", pf.OKABE_ITO["vermillion"], "D"),
    "hmmer3_phmmer": ("phmmer", pf.OKABE_ITO["bluish_green"], "s"),
    "mmseqs2_seqseq": ("MMseqs2", pf.OKABE_ITO["bluish_green"], "^"),
    "mmseqs2_iterative": ("MMseqs2 iterative", pf.OKABE_ITO["bluish_green"], "D"),
    "kd_scan": ("Kyte-Doolittle scan (no search)", pf.OKABE_ITO["black"], "x"),
    "hmmer3_jackhmmer": ("jackhmmer (not in the rules)", pf.GREY, "v"),
    "hhblits": ("HHblits (not in the rules)", pf.GREY, "P"),
}
BIN_ORDER = ["<=20", "21-40", "41-60", "61-120", ">120"]
OUTCOMES = ["lands", "partial", "spills", "no call"]
#: Outcome colours use hues no tool uses, so a colour never means both a tool and an outcome.
OUTCOME_COLOURS = {
    "lands": pf.OKABE_ITO["orange"],
    "partial": pf.OKABE_ITO["yellow"],
    "spills": pf.OKABE_ITO["reddish_purple"],
    "no call": "#dddddd",
}
#: Best of all settings is drawn first, wide and pale, so the best single setting stays
#: visible on top of it where the two are equal (clear-figures rule 7b).
UNDERLAY = {"kmerseek_best_of_all": dict(lw=3, alpha=0.5)}


def load(tables: Path) -> dict:
    return {
        "exon_tool": pl.read_parquet(tables / "exon_tool.parquet"),
        "settings": pl.read_parquet(tables / "kmerseek_settings.parquet"),
        "by_length": pl.read_csv(tables / "summary_by_length.csv"),
        "by_disorder": pl.read_csv(tables / "summary_by_disorder.csv"),
        "null": pl.read_csv(tables / "null_kmerseek_best.csv"),
        "null_variants": pl.read_csv(tables / "null_kmerseek_variants.csv"),
        "verdict": json.loads((tables / "verdict.json").read_text()),
    }


def present(tools) -> list[str]:
    return [t for t in TOOLS if t in set(tools)]


def landing_by_length(d: dict, out: Path, tools_line: str) -> pl.DataFrame:
    """Figure 1: landing rate per exon-length bin, one line per tool; pass lines drawn in.

    Internal exons on pairs where both proteins have an AlphaFold model. x is the median
    exon length of the bin, so the spacing is the real length."""
    et = d["exon_tool"].filter(pl.col("internal") & pl.col("both_models"))
    tab = (
        et.group_by("tool", "length_bin")
        .agg(
            n=pl.len(),
            n_lands=pl.col("lands").sum(),
            median_len=pl.col("exon_len").median(),
        )
        .with_columns(rate=pl.col("n_lands") / pl.col("n"))
        .sort("tool", "median_len")
    )
    x_of_bin = et.group_by("length_bin").agg(x=pl.col("exon_len").median())
    tab = tab.drop("median_len").join(x_of_bin, on="length_bin").sort("tool", "x")
    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 80)
    ax.axhline(
        0.50,
        color=pf.OKABE_ITO["blue"],
        lw=0.5,
        ls="--",
        label="Rule 1: kmerseek must reach 50%",
    )
    ax.axhline(
        0.15,
        color=pf.OKABE_ITO["vermillion"],
        lw=0.5,
        ls="--",
        label="Rule 2: each structure tool must stay under 15%",
    )
    ax.axvspan(
        0, 60.5, color="#f2f2f2", zorder=0, label="Exons the rules use (<= 60 aa)"
    )
    # Tools at 0 in every bin overlap on the x axis; draw them first and name them in a
    # note, so the reader does not have to find hidden lines (clear-figures rule 7b).
    all_zero = [
        t
        for t in present(tab["tool"].unique())
        if tab.filter(pl.col("tool") == t)["rate"].max() == 0
    ]
    order = all_zero + [t for t in present(tab["tool"].unique()) if t not in all_zero]
    for t in order:
        lab, col, mk = TOOLS[t]
        s = tab.filter(pl.col("tool") == t)
        ax.plot(
            s["x"],
            s["rate"],
            color=col,
            marker=mk,
            ms=3,
            label=lab,
            **UNDERLAY.get(t, {}),
        )
    if all_zero:
        ax.text(
            0.99,
            0.97,
            "0 in every length bin: " + ", ".join(TOOLS[t][0] for t in all_zero),
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    ax.set_xscale("log")
    ticks = sorted(x_of_bin["x"].to_list())
    ax.set_xticks(ticks, [f"{int(v)}" for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel(
        "Exon length, median of the bin (aa); bins <=20, 21-40, 41-60, 61-120, >120"
    )
    ax.set_ylabel("Share of exons a call lands on")
    ax.set_ylim(0, 1)
    pf.shared_legend(fig, ncol=4)
    fig.text(0.0, -0.02, tools_line, fontsize=5, ha="left", va="top")
    pf.save(fig, out)
    return tab


def iou_by_length(d: dict, out: Path) -> pl.DataFrame:
    """SF1: median IoU of the landing call, per tool and bin (exons where the tool lands)."""
    et = d["exon_tool"].filter(
        pl.col("internal") & pl.col("both_models") & pl.col("lands")
    )
    tab = (
        et.group_by("tool", "length_bin")
        .agg(
            n=pl.len(),
            median_iou=pl.col("land_iou").median(),
            x=pl.col("exon_len").median(),
        )
        .sort("tool", "x")
    )
    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 65)
    for t in present(tab["tool"].unique()):
        lab, col, mk = TOOLS[t]
        s = tab.filter(pl.col("tool") == t)
        ax.plot(
            s["x"],
            s["median_iou"],
            color=col,
            marker=mk,
            ms=3,
            label=lab,
            **UNDERLAY.get(t, {}),
        )
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Exon length, median of the bin (aa)")
    ax.set_ylabel("Median IoU of the landing call")
    pf.shared_legend(fig, ncol=2)
    pf.save(fig, out)
    return tab


def outcome_shares(d: dict, out: Path) -> pl.DataFrame:
    """SF2: for the verdict exons, the share where each tool lands, partly overlaps,
    spills or makes no call. Stacked horizontal bars, one per tool."""
    v = d["exon_tool"].filter(pl.col("verdict_set"))
    tab = (
        v.group_by("tool", "outcome")
        .len()
        .pivot(on="outcome", index="tool", values="len")
        .fill_null(0)
    )
    for o in OUTCOMES:
        if o not in tab.columns:
            tab = tab.with_columns(pl.lit(0).alias(o))
    order = [t for t in present(tab["tool"])][::-1]
    tab = tab.with_columns(n=pl.sum_horizontal(OUTCOMES))
    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 75)
    for i, t in enumerate(order):
        r = tab.filter(pl.col("tool") == t).row(0, named=True)
        left = 0.0
        for o in OUTCOMES:
            w = r[o] / r["n"]
            ax.barh(
                i,
                w,
                left=left,
                color=OUTCOME_COLOURS[o],
                edgecolor="white",
                lw=0.3,
                label=o if i == 0 else None,
            )
            left += w
    ax.set_yticks(range(len(order)), [TOOLS[t][0] for t in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        f"Share of the {int(tab['n'][0])} internal exons <= 60 aa "
        "(both proteins with an AlphaFold model)"
    )
    pf.shared_legend(fig, ncol=4)
    pf.save(fig, out)
    return tab.select("tool", *OUTCOMES, "n")


def landing_by_disorder(d: dict, out: Path) -> pl.DataFrame:
    """SF3: landing rate on verdict exons, ordered (pLDDT >= 70) vs disordered (< 70)."""
    v = d["exon_tool"].filter(
        pl.col("verdict_set") & (pl.col("disorder") != "no model")
    )
    tab = (
        v.group_by("tool", "disorder")
        .agg(n=pl.len(), n_lands=pl.col("lands").sum())
        .with_columns(rate=pl.col("n_lands") / pl.col("n"))
        .sort("tool", "disorder")
    )
    tools = present(tab["tool"].unique())
    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 70)
    for j, (cls, mk_fill) in enumerate((("ordered", True), ("disordered", False))):
        s = tab.filter(pl.col("disorder") == cls)
        for i, t in enumerate(tools):
            r = s.filter(pl.col("tool") == t)
            if not r.height:
                continue
            lab, col, mk = TOOLS[t]
            ax.plot(
                i + (j - 0.5) * 0.25,
                r["rate"][0],
                marker="o",
                ms=4,
                color=col,
                mfc=col if mk_fill else "white",
                ls="none",
            )
    ax.plot([], [], "o", color="k", mfc="k", label="ordered exon (mean pLDDT >= 70)")
    ax.plot(
        [], [], "o", color="k", mfc="white", label="disordered exon (mean pLDDT < 70)"
    )
    ax.set_xticks(
        range(len(tools)), [TOOLS[t][0] for t in tools], rotation=30, ha="right"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of exons a call lands on")
    pf.shared_legend(fig, ncol=2)
    pf.save(fig, out)
    return tab


def settings_and_null(d: dict, out: Path) -> pl.DataFrame:
    """(a) every kmerseek setting's landing rate on the verdict exons, the best of all
    settings as a line; (b) each kmerseek number the verdict quotes (blue dot) against its
    random-protein null (grey: every draw as a tick, the 0.1-99.9 percentile range as a
    bar, the median as a white notch)."""
    st = d["settings"].sort("rate")
    nv = d["null_variants"]
    v = d["verdict"]
    extra = v["random_protein_null_added_after_stop0"]
    best_all = v["rule_1"]["kmerseek_best_of_all_lands"][2]
    single_col = [c for c in nv.columns if c.startswith("single_")][0]
    rows = [
        ("best of all settings", "best_of_all", best_all),
        (
            "best of all, p_place < 0.05",
            "best_of_all_p_place_lt_0.05",
            extra["best_of_all_p_place_lt_0.05"]["observed"],
        ),
        (
            f"best single setting ({single_col.removeprefix('single_')})",
            single_col,
            extra["best_single_setting"]["observed"],
        ),
    ]
    pf.use_style()
    fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 60, ncols=2)
    a.plot(
        np.arange(st.height),
        st["rate"],
        "o",
        ms=1.5,
        color=pf.GREY,
        label="one kmerseek setting (alphabet, k, mask)",
    )
    a.axhline(
        best_all,
        color=pf.OKABE_ITO["blue"],
        lw=1,
        label="best of all settings, real exons",
    )
    a.set_xlabel(f"kmerseek settings, sorted by landing rate (n = {st.height})")
    a.set_ylabel("Share of verdict exons a call lands on")
    a.set_ylim(0, 1)
    out_rows = []
    for i, (lab, col, obs) in enumerate(rows[::-1]):
        x = nv[col].to_numpy()
        lo, med, hi = np.quantile(x, [0.001, 0.5, 0.999])
        b.plot(x, np.full(len(x), i), "|", color=pf.GREY, ms=4, alpha=0.3)
        b.plot([lo, hi], [i, i], color=pf.GREY, lw=3, solid_capstyle="butt")
        b.plot(med, i, "|", color="white", ms=5, mew=1)
        b.plot(obs, i, "o", color=pf.OKABE_ITO["blue"], ms=4)
        out_rows.append((lab, obs, med, lo, hi))
    b.plot(
        [], [], "s", color=pf.GREY, label="null: exon moved to another pair's target"
    )
    b.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    b.set_xlim(0, 1)
    b.set_xlabel(f"Share of verdict exons landed (null: {len(nv)} draws)")
    pf.shared_legend(fig, ncol=3)
    pf.panel_label(a, "a")
    pf.panel_label(b, "b")
    pf.save(fig, out)
    return pl.DataFrame(
        out_rows[::-1],
        schema=[
            "kmerseek number",
            "real exons",
            "null median",
            "null 0.1%",
            "null 99.9%",
        ],
        orient="row",
    )


def verdict_figure(d: dict, out: Path) -> pl.DataFrame:
    """Each rule's landing rate on the verdict exons as a dot, with the rule's threshold
    as a vertical tick on the same row; filled dot = the rule passes, open = it fails.
    """
    v = d["verdict"]
    rows = [
        (
            "Rule 1: kmerseek, best of all settings",
            "kmerseek_best_of_all",
            v["rule_1"]["kmerseek_best_of_all_lands"][2],
            v["rule_1"]["threshold"],
            v["rule_1"]["pass"],
        )
    ]
    for t, x in v["rule_2"]["structure_tools_land"].items():
        rows.append(
            (
                f"Rule 2: {TOOLS[t][0]}",
                t,
                x[2],
                v["rule_2"]["threshold"],
                x[2] is not None and x[2] < v["rule_2"]["threshold"],
            )
        )
    km3 = v["rule_3"]["kmerseek_lands_with_p_place_lt_0.05"][2]
    kd = v["rule_3"]["kd_scan_lands"][2]
    rows.append(
        (
            "Rule 3: kmerseek, p_place < 0.05",
            "kmerseek_best_of_all",
            km3,
            None if kd is None else 2 * kd,
            v["rule_3"]["pass"],
        )
    )
    rows.append(("Rule 3: Kyte-Doolittle scan", "kd_scan", kd, None, None))
    tab = pl.DataFrame(
        rows, schema=["row", "tool", "rate", "threshold", "passes"], orient="row"
    )
    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 55)
    n = tab.height
    for i, r in enumerate(tab.iter_rows(named=True)):
        y = n - 1 - i
        ax.axhline(y, color="#e5e5e5", lw=0.3, zorder=0)
        col = TOOLS[r["tool"]][1]
        filled = r["passes"] is not False
        if r["rate"] is not None and r["passes"] is None:
            # Context row (the scan): its own tool marker, not a pass/fail circle.
            ax.plot(r["rate"], y, TOOLS[r["tool"]][2], ms=5, color=col)
        elif r["rate"] is not None:
            ax.plot(r["rate"], y, "o", ms=5, color=col, mfc=col if filled else "white")
        if r["threshold"] is not None:
            ax.plot(r["threshold"], y, "|", ms=10, color="k")
    ax.plot(
        [], [], "|", color="k", ms=8, label="rule's threshold (rule 3: twice the scan)"
    )
    ax.plot([], [], "o", color="k", mfc="k", label="passes")
    ax.plot([], [], "o", color="k", mfc="white", label="fails")
    ax.set_yticks(range(n), tab["row"].to_list()[::-1])
    ax.set_xlim(-0.02, 1)
    ax.set_xlabel("Share of verdict exons a call lands on")
    pf.shared_legend(fig, ncol=3)
    pf.save(fig, out)
    return tab
