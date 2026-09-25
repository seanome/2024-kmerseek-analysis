"""Compute cost of kmerseek 0.4 against 0.3, matched on dataset, query chunk, alphabet and k.

Inputs (data/kmerseek_0.4_vs_0.3_resource/):
  trace.0.3.ladder.txt.gz       Nextflow trace of the 0.3 dark-set ladder run (amazing_koch,
                                2026-09-16). Also holds the MMseqs2, phmmer and jackhmmer searches.
  nextflow_log.0.4.midi.tsv.gz  `nextflow log <run> -f name,status,exit,realtime,pcpu,peak_rss,
                                cpus,memory,hash` for every head of the 0.4 midi run, read on
                                2026-09-24 from a copy of the run's .nextflow cache. The run's own
                                trace file only covers the latest head, because Nextflow truncates
                                it at each launch, and the HTML report drops per-task rows past
                                its task limit.

Both runs search 2_000-query chunks of one proteome against Swiss-Prot with the query's own
clade removed, with 8 CPUs per search on Sherlock hns. A 0.4 search is paired with the 0.3
search on the same species, chunk, alphabet and k. 0.3 kept every k-mer, which is 0.4's
scaled 1.

Outputs:
  figures/kmerseek_0.4_vs_0.3_resource_matched.png
  tables/kmerseek_0.4_vs_0.3_resource_matched_summary.csv   median and max per tool
  tables/kmerseek_0.4_vs_0.3_resource_by_alphabet.csv       paired median ratios per alphabet
  tables/kmerseek_0.4_index_build_cost.csv                  0.4 index builds by scaling

Run from the repository root:
  python scripts/kmerseek_0_4_vs_0_3_resource.py
"""

import re
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DATA = Path("data/kmerseek_0.4_vs_0.3_resource")
FIG = Path("figures/kmerseek_0.4_vs_0.3_resource_matched.png")
TABLES = Path("tables")
KEY = ["species", "chunk", "alphabet", "k"]
UNIT_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86_400}
UNIT_GB = {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1, "TB": 1e3}


def seconds(s):
    if s in (None, "-"):
        return None
    return sum(
        float(v) * UNIT_SECONDS[u] for v, u in re.findall(r"([\d.]+)(ms|s|m|h|d)", s)
    )


def gigabytes(s):
    if s in (None, "-", "0"):
        return None
    value, unit = s.split()
    return float(value) * UNIT_GB[unit]


def load_tasks():
    v03 = pl.read_csv(
        DATA / "trace.0.3.ladder.txt.gz",
        separator="\t",
        infer_schema=False,
        quote_char=None,
    ).filter(pl.col("status").is_in(["COMPLETED", "CACHED"]))
    v04 = (
        pl.read_csv(
            DATA / "nextflow_log.0.4.midi.tsv.gz",
            separator="|",
            infer_schema=False,
            quote_char=None,
        )
        .filter(pl.col("status") == "COMPLETED")
        .rename({"pcpu": "%cpu"})
    )
    cols = ["name", "realtime", "%cpu", "peak_rss", "hash"]
    # A task appears once per head that saw it; the hash identifies it.
    d = pl.concat(
        [
            v03.unique("hash").select(cols).with_columns(ver=pl.lit("0.3")),
            v04.unique("hash").select(cols).with_columns(ver=pl.lit("0.4")),
        ]
    )
    d = d.with_columns(
        proc=pl.col("name").str.extract(r":(\w+) "),
        tag=pl.col("name").str.extract(r"\((.*)\)"),
        realtime_s=pl.col("realtime").map_elements(seconds, return_dtype=pl.Float64),
        rss_gb=pl.col("peak_rss").map_elements(gigabytes, return_dtype=pl.Float64),
        cpu=pl.col("%cpu").str.strip_suffix("%").cast(pl.Float64, strict=False) / 100,
    )
    return d.with_columns(
        species=pl.col("tag").str.extract(r"^(\w+)\.chunk"),
        clade=pl.col("tag").str.extract(r"minus_(\w+)"),
        chunk=pl.col("tag").str.extract(r"chunk_(\d+)"),
        alphabet=pl.col("tag").str.extract(r"\.([a-z][a-z0-9_]*\d)\.k\d+"),
        k=pl.col("tag").str.extract(r"\.k(\d+)").cast(pl.Int32, strict=False),
        scaled=pl.col("tag")
        .str.extract(r"\.s(\d+)\.")
        .cast(pl.Int32, strict=False)
        .fill_null(1),
        arm=pl.col("tag").str.extract(r"lc\w+\.(.*)$"),
        cpu_h=pl.col("realtime_s") * pl.col("cpu") / 3600,
    )


def main():
    d = load_tasks()
    search = d.filter(pl.col("proc") == "kmerseekSearch")
    k03 = search.filter(pl.col("ver") == "0.3").select(
        *KEY, rt3="realtime_s", rss3="rss_gb", cpu3="cpu_h"
    )
    # Exact k-mer matches only: the arm that 0.3 also ran.
    pairs = search.filter((pl.col("ver") == "0.4") & (pl.col("arm") == "exact")).join(
        k03, on=KEY
    )
    chunks = pairs.select("species", "chunk").unique()
    others = d.filter(
        pl.col("proc").is_in(["mmseqs2Search", "phmmerSearch", "jackhmmerSearch"])
        & (pl.col("ver") == "0.3")
    ).join(chunks, on=["species", "chunk"])
    k03_matched = search.filter(pl.col("ver") == "0.3").join(
        pairs.filter(pl.col("scaled") == 1).select(KEY).unique(), on=KEY
    )

    groups = [
        ("jackhmmer", others.filter(pl.col("proc") == "jackhmmerSearch"), "other"),
        ("phmmer", others.filter(pl.col("proc") == "phmmerSearch"), "other"),
        ("kmerseek 0.3\n(same searches\nas 0.4 scaled 1)", k03_matched, "k03"),
    ]
    groups += [
        (f"kmerseek 0.4\nscaled {s}", pairs.filter(pl.col("scaled") == s), "k04")
        for s in (1, 2, 5, 10)
    ]
    groups += [("MMseqs2", others.filter(pl.col("proc") == "mmseqs2Search"), "other")]
    colour = {"other": "#d9822b", "k03": "#8a8a8a", "k04": "#2b6cb0"}
    metrics = [
        ("realtime_s", 1 / 60, "Wall time per chunk (min)"),
        ("cpu_h", 1, "CPU time per chunk (CPU-hours)"),
        ("rss_gb", 1, "Peak memory per chunk (GB)"),
    ]

    rows = []
    for name, g, _ in groups:
        row = {"tool": name.replace("\n", " "), "n_searches": g.height}
        for m, f, _ in metrics:
            row[m + "_median"] = round(float(g[m].median()) * f, 4)
            row[m + "_max"] = round(float(g[m].max()) * f, 2)
        rows.append(row)
    summary = pl.DataFrame(rows).rename(
        {
            "realtime_s_median": "wall_min_median",
            "realtime_s_max": "wall_min_max",
            "cpu_h_median": "cpu_hours_median",
            "cpu_h_max": "cpu_hours_max",
        }
    )
    print(summary)

    by_alphabet = (
        pairs.group_by("alphabet", "scaled")
        .agg(
            pl.len().alias("n_pairs"),
            (pl.col("rt3") / pl.col("realtime_s")).median().alias("speedup_median"),
            (pl.col("rss3") / pl.col("rss_gb")).median().alias("memory_ratio_median"),
            (pl.col("cpu3") / pl.col("cpu_h")).median().alias("cpu_time_ratio_median"),
        )
        .sort("alphabet", "scaled")
    )
    overall = (
        pairs.group_by("scaled")
        .agg(
            pl.len().alias("n_pairs"),
            (pl.col("rt3") / pl.col("realtime_s")).median().alias("speedup_median"),
            (pl.col("rss3") / pl.col("rss_gb")).median().alias("memory_ratio_median"),
            (pl.col("cpu3") / pl.col("cpu_h")).median().alias("cpu_time_ratio_median"),
        )
        .sort("scaled")
    )
    print(overall)

    index = (
        d.filter((pl.col("proc") == "kmerseekIndex") & (pl.col("ver") == "0.4"))
        .group_by("scaled")
        .agg(
            pl.len().alias("n_indexes"),
            (pl.col("realtime_s").median() / 60).alias("wall_min_median"),
            pl.col("rss_gb").median().alias("peak_gb_median"),
            pl.col("rss_gb").max().alias("peak_gb_max"),
            pl.col("cpu_h").median().alias("cpu_hours_median"),
        )
        .sort("scaled")
    )
    print(index)

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(
        2, 3, hspace=0.45, wspace=0.42, top=0.84, bottom=0.07, left=0.06, right=0.98
    )
    rng = np.random.default_rng(0)
    for j, (m, f, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        for i, (_, g, c) in enumerate(groups):
            v = g[m].drop_nulls().to_numpy() * f
            v = v[v > 0]
            ax.scatter(
                i + rng.uniform(-0.28, 0.28, len(v)),
                v,
                s=5,
                color=colour[c],
                alpha=0.35,
                lw=0,
            )
            med = np.median(v)
            ax.plot([i - 0.36, i + 0.36], [med, med], color="black", lw=2)
            ax.text(
                i,
                med * 1.35,
                f"{med:.0f}" if med >= 10 else f"{med:.2g}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8),
            )
        ax.set_yscale("log")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(
            [g[0] for g in groups], fontsize=7.5, rotation=45, ha="right"
        )
        ax.set_ylabel(label)
        ax.set_title("ABC"[j], loc="left", fontweight="bold")

    for j, (m, old, f, label, unit) in enumerate(
        [
            ("realtime_s", "rt3", 1 / 60, "wall time", "min"),
            ("rss_gb", "rss3", 1, "peak memory", "GB"),
        ]
    ):
        ax = fig.add_subplot(gs[1, j])
        for s, marker in ((1, "o"), (10, "^")):
            q = pairs.filter(pl.col("scaled") == s)
            ax.scatter(
                q[old].to_numpy() * f,
                q[m].to_numpy() * f,
                s=10,
                marker=marker,
                facecolor="none" if s == 1 else colour["k04"],
                edgecolor=colour["k04"],
                alpha=0.6,
                lw=0.7,
            )
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", zorder=0)
        ax.set_xlabel(f"kmerseek 0.3 {label} ({unit})")
        ax.set_ylabel(f"kmerseek 0.4 {label} ({unit})")
        ax.set_title("DE"[j], loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    order = (
        by_alphabet.filter(pl.col("scaled") == 1)
        .sort("speedup_median")["alphabet"]
        .to_list()
    )
    for y, a in enumerate(order):
        for s, marker in ((1, "o"), (10, "^")):
            r = by_alphabet.filter((pl.col("alphabet") == a) & (pl.col("scaled") == s))
            if r.height:
                ax.scatter(
                    r["speedup_median"][0],
                    y,
                    marker=marker,
                    s=28,
                    facecolor="none" if s == 1 else colour["k04"],
                    edgecolor=colour["k04"],
                )
    ax.axvline(1, color="black", ls="--", lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    ax.set_xlabel("Median speed-up, 0.3 time / 0.4 time (x)")
    ax.set_title("F", loc="left", fontweight="bold")

    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color=colour["other"],
            label="Other tools (one 2,000-query chunk each)",
        ),
        Line2D([], [], ls="", marker="o", color=colour["k03"], label="kmerseek 0.3"),
        Line2D([], [], ls="", marker="o", color=colour["k04"], label="kmerseek 0.4"),
        Line2D([], [], color="black", lw=2, label="Median (number printed above)"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="none",
            mec=colour["k04"],
            label="D-F: 0.4 scaled 1 (every k-mer kept, as in 0.3)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="^",
            color=colour["k04"],
            label="D-F: 0.4 scaled 10 (1 in 10 k-mers kept)",
        ),
        Line2D([], [], color="black", ls="--", lw=1, label="D-F: no change from 0.3"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"kmerseek 0.4 against 0.3 on the same query chunk, alphabet and k-mer size ({pairs.height:,} matched "
        "searches, human, worm and yeast vs Swiss-Prot\nwith the query's clade removed, 8 CPUs each on Sherlock). "
        "A-C also show the other tools on the same chunks. Exact k-mer matches only.",
        fontsize=11,
        y=0.985,
    )
    fig.savefig(FIG, dpi=150)
    summary.write_csv(TABLES / "kmerseek_0.4_vs_0.3_resource_matched_summary.csv")
    by_alphabet.write_csv(TABLES / "kmerseek_0.4_vs_0.3_resource_by_alphabet.csv")
    index.write_csv(TABLES / "kmerseek_0.4_index_build_cost.csv")


if __name__ == "__main__":
    main()
