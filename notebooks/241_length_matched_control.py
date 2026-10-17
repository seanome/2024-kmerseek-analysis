#!/usr/bin/env python3
"""Random-protein control: does a measure rank BCL2 (for Ced9) and CD47 (for P66) above
human proteins of the same length?

The size-scaled whole-protein enrichment of seanome/kmerseek PR 136 ranks both partners
better than any other measure, but it also favours short proteins, and both partners are
shorter than the mean human protein (BCL2 239 aa, CD47 323 aa, mean 575 aa). If length is
all it sees, a random protein of the partner's length ranks as well as the partner.

For each setting, query and measure: take every human protein within 10% of the
partner's length (the partner excluded), rank all proteins as the collector does, and
report the share of those length-matched proteins that rank strictly better than the
partner. Proteins with no shared k-mer have no value and count as tied for last.
0 means the partner beats every protein of its length; about 0.5 means it is an ordinary
protein of its length.

Inputs: the PR 136 searches 241_length_scaled_check.py writes
($NB241_CHECK_DIR/pr136/search/*.csv) and the human FASTA. Writes
notebooks/241_results/length_matched_control.csv and
figures/241_length_matched_control.{png,pdf}.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import polars as pl
from matplotlib.lines import Line2D

import pubfig as pf

HERE = Path(__file__).resolve().parent
CHECK = Path(os.environ["NB241_CHECK_DIR"])
HUMAN = Path(
    os.environ.get(
        "NB241_HUMAN",
        "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    )
)
PARTNER = {"Ced9": "BCL2", "P66": "CD47"}
WINDOW = 0.10  # length-matched: within 10% of the partner's length
OUT_CSV = HERE / "241_results" / "length_matched_control.csv"
OUT_FIG = HERE.parent / "figures" / "241_length_matched_control"
SHOWN = [
    "protein enrichment",
    "protein Poisson p-value",
    "E-value",
    "mean IDF",
    "tf-idf",
    "Poisson score",
]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def human_lengths() -> pl.DataFrame:
    rows = []
    for line in open(HUMAN):
        if line.startswith(">"):
            f = line[1:].strip().split("|")
            rows.append((line[1:].strip(), f[6], int(f[7])))
    return pl.DataFrame(rows, schema=["target_name", "gene", "length"], orient="row")


def main() -> None:
    col = load("nb241_collect", "241_alphabet_ranking_collect.py")
    proteins = human_lengths()
    rows = []
    for csv in sorted((CHECK / "pr136" / "search").glob("*.csv")):
        tag = csv.stem
        alphabet, k = tag.rsplit(".k", 1)
        df = col.read_search(tag, out=CHECK / "pr136")
        if df is None:
            continue
        for query, partner in PARTNER.items():
            sub = df.filter(pl.col("query_name") == query)
            p_len = proteins.filter(pl.col("gene") == partner)["length"][0]
            lo, hi = p_len * (1 - WINDOW), p_len * (1 + WINDOW)
            matched = proteins.filter(
                pl.col("length").is_between(lo, hi) & (pl.col("gene") != partner)
            )
            for column, lower, label in col.METRICS:
                if column not in sub.columns:
                    continue
                per = col.per_target(sub, column, lower)
                part = per.filter(pl.col("gene") == partner)
                if part.height == 0:
                    continue  # partner not hit, or no value under this measure
                pv = part["v"][0]
                m = matched.join(
                    per.select("target_name", "v"), on="target_name", how="left"
                )
                better = (m["v"] < pv) if lower else (m["v"] > pv)
                n_better = int(better.fill_null(False).sum())
                rows.append(
                    dict(
                        alphabet=alphabet,
                        ksize=int(k),
                        query=query,
                        partner=partner,
                        partner_length=p_len,
                        measure=label,
                        n_matched=m.height,
                        n_matched_hit=int(m["v"].is_not_null().sum()),
                        n_matched_better=n_better,
                        share_matched_better=n_better / m.height,
                    )
                )
    table = pl.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    table.write_csv(OUT_CSV)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(170)
    shown = table.filter(pl.col("measure").is_in(SHOWN))
    print(shown.sort("query", "measure", "share_matched_better"))
    print(
        shown.group_by("query", "measure")
        .agg(
            pl.len().alias("settings"),
            pl.col("share_matched_better").median().alias("median_share_better"),
            pl.col("share_matched_better").min().alias("best"),
        )
        .sort("query", "median_share_better")
    )

    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 78, ncols=2, sharey=True)
    c = pf.OKABE_ITO
    colour = {
        "protein enrichment": c["orange"],
        "protein Poisson p-value": c["blue"],
        "E-value": c["bluish_green"],
        "mean IDF": c["reddish_purple"],
        "tf-idf": c["sky_blue"],
        "Poisson score": "black",  # vermillion means kmerseek main in the other 241 figures
    }
    label = {
        "protein enrichment": "whole-protein\nenrichment",
        "protein Poisson p-value": "whole-protein\np-value",
        "E-value": "region\nE-value",
        "mean IDF": "region\nmean IDF",
        "tf-idf": "region\ntf-idf",
        "Poisson score": "region\nPoisson score",
    }
    for ax, (query, partner) in zip(axs, PARTNER.items()):
        q = shown.filter(pl.col("query") == query)
        ax.axhline(0.5, color=pf.GREY, ls="--", lw=0.8, zorder=0)
        for i, measure in enumerate(SHOWN):
            v = q.filter(pl.col("measure") == measure)["share_matched_better"].to_list()
            xs = [i + (j - (len(v) - 1) / 2) * 0.06 for j in range(len(v))]
            ax.plot(xs, v, "o", color=colour[measure], ms=3, alpha=0.9)
        n_matched = q["n_matched"][0] if q.height else 0
        ax.set_xticks(range(len(SHOWN)))
        ax.set_xticklabels([label[m] for m in SHOWN])
        ax.set_xlim(-0.6, len(SHOWN) - 0.4)
        ax.set_title(
            f"query {query}, partner {partner} ({q['partner_length'][0]} aa)\n"
            f"vs the {n_matched:,} human proteins within 10% of its length",
            loc="left",
        )
    axs[0].set_ylim(-0.02, 1.02)
    axs[0].set_ylabel(
        "Share of same-length proteins\nranked above the partner (0 = best)"
    )
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color="black",
            ms=3,
            label="one kmerseek alphabet and k (where the partner is hit)",
        ),
        Line2D(
            [],
            [],
            ls="--",
            lw=0.8,
            color=pf.GREY,
            label="0.5: where a random protein of the same length lands on average",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, OUT_FIG, formats=("pdf", "png"))
    print(f"wrote {OUT_CSV} and {OUT_FIG}.png/.pdf")


if __name__ == "__main__":
    main()
