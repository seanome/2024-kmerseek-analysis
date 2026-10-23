"""Recall with the Pfam index against the whole-protein and regions indexes, full run.

Reads results_full_2026_10_16/comparison/recall_by_kind.tsv (bin/compare_indexes.py on
every scored label of the full run, 2026-10-16) and plots, for each kmerseek setting and
for MMseqs2, the share of query features found at 5% decoy error with each of the three
indexes: headline landing rule (the aligned piece), decoys shuffled in 10-residue windows,
all evidence. Panel a: folded domains (907 features). Panel b: all 7_174 features.

The Pfam index searched with the whole-protein index's fit ("pfamshared") is left out: its
recall is within 1 percentage point of "pfam" for every setting (printed below).

Writes results_full_2026_10_16/pfam_index_recall.{pdf,svg,png}.

    python tools/pfam_index_figure.py
"""

import re
import sys
from pathlib import Path

import polars as pl
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

RESULTS = HERE / "results_full_2026_10_16"
INDEXES = {  # index label -> (legend text, colour)
    "whole": ("Whole proteins", pf.OKABE_ITO["blue"]),
    "regions": ("Annotated regions", pf.OKABE_ITO["orange"]),
    "pfam": ("Pfam domains", pf.OKABE_ITO["bluish_green"]),
}
KINDS = [("folded_domain", "Folded domains"), ("all", "All features")]


def main() -> None:
    r = pl.read_csv(RESULTS / "comparison/recall_by_kind.tsv", separator="\t").filter(
        (pl.col("landing_rule") == "aligned piece")
        & (pl.col("window") == "w10")
        & (pl.col("evidence") == "all")
        & (pl.col("gene_group") == "all")
    )
    shared = (
        r.filter(pl.col("index").is_in(["pfam", "pfamshared"]))
        .pivot(on="index", index=["setting", "feature_kind"], values="recall")
        .drop_nulls()
        .select((pl.col("pfam") - pl.col("pfamshared")).abs().max() * 100)
        .item()
    )
    print(f"largest |pfam - pfamshared| recall difference: {shared:.2f} points")

    r = r.filter(pl.col("index").is_in(list(INDEXES)))
    # Rows sorted by whole-protein recall on folded domains; a setting whose whole-protein
    # search got no fit (hp_kyte_doolittle2 k22 in this window) goes at the bottom.
    whole = r.filter(
        (pl.col("feature_kind") == "folded_domain") & (pl.col("index") == "whole")
    ).select("setting", pl.col("recall").alias("whole_recall"))
    order = (
        r.filter(pl.col("tool") == "kmerseek")
        .select("setting")
        .unique()
        .join(whole, on="setting", how="left")
        .sort("whole_recall", "setting", nulls_last=False)["setting"]
        .to_list()
    )
    rows = order + ["mmseqs2"]
    ypos = {s: i + (1 if s == "mmseqs2" else 0) for i, s in enumerate(rows)}
    labels = [re.sub(r"_k(\d+)_s1$", r" k\1", s) for s in order] + ["MMseqs2"]

    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 95, ncols=2, sharey=True)
    for ax, (kind, title) in zip(axs, KINDS):
        sub = r.filter(pl.col("feature_kind") == kind)
        n = sub["features"].unique().item()
        for y in ypos.values():
            ax.axhline(y, color="#dddddd", lw=0.3, zorder=0)
        for idx, (text, colour) in INDEXES.items():
            pts = sub.filter(pl.col("index") == idx).drop_nulls("recall")
            ax.scatter(
                pts["recall"] * 100,
                [ypos[s] for s in pts["setting"]],
                s=14,
                color=colour,
                label=text if ax is axs[0] else None,
                zorder=2,
            )
        ax.axhline(len(order) + 0.0, color="black", lw=0.5)
        ax.set_xlim(0, 80)
        ax.set_xlabel(f"{title} found, of {n:_} (%)".replace("_", ","))
    axs[0].set_yticks(list(ypos.values()), labels)
    axs[0].set_ylim(-0.7, max(ypos.values()) + 0.7)
    axs[0].set_ylabel("kmerseek alphabet and k-mer size")
    pf.shared_legend(fig, ncol=3)
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    out = RESULTS / "pfam_index_recall"
    pf.save(fig, str(out))
    # 8-bit palette PNG; this branch's pubfig.save predates its own conversion step.
    png = out.with_suffix(".png")
    with Image.open(png) as image:
        dpi = image.info.get("dpi")
        small = image.convert("RGB").quantize(colors=256, dither=Image.Dither.NONE)
    small.save(png, optimize=True, **({"dpi": dpi} if dpi else {}))


if __name__ == "__main__":
    main()
