#!/usr/bin/env python3
"""Does kmerseek's expected count of shared k-mers match the observed count at every
target length? Before and after seanome/kmerseek PR 136.

For one query, the count of k-mers it shares with each human protein is observed; the
whole-protein chance count μ is what kmerseek expects (the mean of the Poisson behind its
p-value, not an E-value). Sum both over all proteins in a length bin (proteins with no
shared k-mer add 0 observed). If μ is right, observed / μ is the same in every bin.
kmerseek main's μ is the same for every target, so the ratio climbs with length; PR 136's
μ scales with the target's k-mer count.

Inputs: the hp_lehninger2 k=17 searches 241_length_scaled_check.py writes
($NB241_CHECK_DIR/{main,pr136}/search/hp_lehninger2.k17.csv) and the human FASTA.
Proteins that share no k-mer with the query are not in the search output, so their PR 136
μ is read off the μ of hit proteins with the same k-mer count (np.interp), and their k-mer
count from their length by a straight-line fit on the hit proteins.

Writes notebooks/241_results/expected_vs_observed_by_length.csv and
figures/241_expected_vs_observed_by_length.{png,pdf}. With --plot-only, redraws the
figure from that CSV without the search outputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.lines import Line2D

import pubfig as pf

HERE = Path(__file__).resolve().parent
HUMAN = Path(
    os.environ.get(
        "NB241_HUMAN",
        "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    )
)
TAG = "hp_lehninger2.k17"
N_BINS = 8
OUT_CSV = HERE / "241_results" / "expected_vs_observed_by_length.csv"
OUT_FIG = HERE.parent / "figures" / "241_expected_vs_observed_by_length"


def per_target(build_dir: str) -> pl.DataFrame:
    check = Path(os.environ["NB241_CHECK_DIR"])
    d = pl.read_csv(check / build_dir / "search" / f"{TAG}.csv", infer_schema_length=0)
    cols = [
        "n_intersecting_hashes",
        "query_expected_shared_kmers",
        "containment_target_in_query",
    ]
    return (
        d.with_columns(pl.col("query_name").str.strip_chars())
        .group_by("query_name", "target_name")
        .agg(*[pl.col(c).first().cast(pl.Float64) for c in cols])
    )


def build_table() -> pl.DataFrame:
    headers = [line[1:].strip() for line in open(HUMAN) if line.startswith(">")]
    proteins = pl.DataFrame({"target_name": headers}).with_columns(
        pl.col("target_name").str.split("|").list.get(7).cast(pl.Int64).alias("length")
    )
    old, new = per_target("main"), per_target("pr136")
    rows = []
    for query in ["Ced9", "P66"]:
        e_old = old.filter(pl.col("query_name") == query)[
            "query_expected_shared_kmers"
        ][0]
        hit = (
            new.filter(pl.col("query_name") == query)
            .with_columns(
                (
                    pl.col("n_intersecting_hashes")
                    / pl.col("containment_target_in_query")
                ).alias("kmers")
            )
            .select(
                "target_name",
                "n_intersecting_hashes",
                "query_expected_shared_kmers",
                "kmers",
            )
        )
        t = proteins.join(hit, on="target_name", how="left")
        h = t.filter(pl.col("kmers").is_not_null())
        slope, intercept = np.polyfit(h["length"].to_numpy(), h["kmers"].to_numpy(), 1)
        by_kmers = h.sort("kmers")
        kmers = (
            t["kmers"]
            .fill_null(pl.Series(t["length"].to_numpy() * slope + intercept))
            .clip(1)
            .to_numpy()
        )
        e_new = np.where(
            t["query_expected_shared_kmers"].is_null().to_numpy(),
            np.interp(
                kmers,
                by_kmers["kmers"].to_numpy(),
                by_kmers["query_expected_shared_kmers"].to_numpy(),
            ),
            t["query_expected_shared_kmers"].fill_null(0).to_numpy(),
        )
        t = t.with_columns(
            pl.Series("e_new", e_new),
            pl.col("n_intersecting_hashes").fill_null(0).alias("observed"),
            pl.col("length")
            .qcut(N_BINS, labels=[str(i) for i in range(N_BINS)])
            .alias("bin"),
        )
        s = (
            t.group_by("bin")
            .agg(
                pl.len().alias("proteins"),
                pl.col("length").min().alias("length_min"),
                pl.col("length").max().alias("length_max"),
                pl.col("length").median().alias("length_median"),
                pl.col("observed").sum(),
                pl.col("e_new").sum().alias("expected_pr136"),
            )
            .with_columns(
                (pl.col("proteins") * e_old).alias("expected_main"),
                pl.lit(query).alias("query"),
                pl.lit(h.height).alias("proteins_hit"),
            )
            .with_columns(
                (pl.col("observed") / pl.col("expected_main")).alias("ratio_main"),
                (pl.col("observed") / pl.col("expected_pr136")).alias("ratio_pr136"),
            )
            .sort("length_min")
        )
        rows.append(s.drop("bin"))
        print(
            f"{query}: main's μ = {e_old:.3f} per protein; kmers ~ {slope:.3f} * length "
            f"+ {intercept:.1f} (fit on the {h.height:,} proteins hit)"
        )
    table = pl.concat(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    table.write_csv(OUT_CSV)
    return table


def plot(table: pl.DataFrame) -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(160)
    print(
        table.select(
            "query",
            "length_min",
            "length_max",
            "proteins",
            "observed",
            "expected_main",
            "expected_pr136",
            "ratio_main",
            "ratio_pr136",
        )
    )

    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 70)
    c = pf.OKABE_ITO
    style = {"Ced9": "o", "P66": "s"}
    ax.axhline(1, color=pf.GREY, lw=0.5, zorder=0)
    for query, marker in style.items():
        q = table.filter(pl.col("query") == query)
        for col, colour in (
            ("ratio_main", c["vermillion"]),
            ("ratio_pr136", c["blue"]),
        ):
            ax.plot(
                q["length_median"], q[col], marker=marker, ls="-", color=colour, ms=3
            )
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["100", "200", "500", "1,000", "2,000"])
    ax.minorticks_off()
    ax.set_xlabel("Median length of the human proteins in the bin (aa)")
    ax.set_ylabel("Observed / expected shared k-mers\n(1 = expectation right)")
    ax.set_ylim(0, 3)
    handles = [
        Line2D(
            [],
            [],
            color=c["vermillion"],
            label="kmerseek main: same μ for every target",
        ),
        Line2D([], [], color=c["blue"], label="PR 136: μ scaled by target size"),
        Line2D([], [], ls="", marker="o", color="black", ms=3, label="query Ced9"),
        Line2D([], [], ls="", marker="s", color="black", ms=3, label="query P66"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    ax.set_title(
        "hp_lehninger2 k=17, target GENCODE v49 human, 8 bins of ~2,470 proteins",
        loc="left",
    )
    pf.save(fig, OUT_FIG, formats=("pdf", "png"))
    print(f"wrote {OUT_FIG}.png/.pdf")


def main() -> None:
    # --plot-only redraws from the saved table, for when the search CSVs are gone.
    if "--plot-only" in sys.argv:
        plot(pl.read_csv(OUT_CSV))
    else:
        plot(build_table())
        print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
