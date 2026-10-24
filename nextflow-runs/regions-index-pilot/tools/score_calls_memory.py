"""scoreCalls memory: measured peaks against the first ask the sherlock profile makes.

Reads build_summary/score_calls_measured.tsv: one row per label, the bytes of its gzipped
call files (summed over the 10 query chunks, from the kmerseekSearch work folders of the
full run on Sherlock) and what scoreCalls did with them in that run (trace
regions_index_pilot.2026-10-15.trace.txt: status, exit, peak_rss). That run used the old
bin/score_calls.py, which held every call in memory.

Reads build_summary/score_calls_streaming_measured.tsv: peaks of the old and the new
bin/score_calls.py (2026-10-16; the new one reads the call files in pieces), measured on
Sherlock compute nodes in SLURM jobs 48116363 and 48119285. One row per score_calls.py
run; a whole-protein task runs it twice (all calls, then annotated targets only), so its
peak is the larger of the two.

Writes build_summary/score_calls_memory.tsv (first and second ask per label) and the figure
build_summary/score_calls_memory.{pdf,svg,png}.

The formula is the one in nextflow.config (withName: scoreCalls). Its constants are read
from that file, so the table cannot drift from the config.

    python tools/score_calls_memory.py
"""

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

CONFIG = HERE / "nextflow.config"
P = {
    m.group(1): float(m.group(2))
    for m in re.finditer(
        r"^params\.(score_calls_\w+)\s*=\s*([0-9.]+)", CONFIG.read_text(), re.M
    )
}
KILL_GB = 24  # the flat withLabel: python ask the killed tasks ran under
# The constants fitted on 2026-10-16 to the old score_calls.py, for the figure only.
OLD = {"whole": (6, 32), "other": (1, 18)}


def peak_gb(kind: str, gb: float) -> float:
    if kind == "whole":
        return P["score_calls_whole_base_gb"] + P["score_calls_whole_slope"] * gb
    return P["score_calls_other_base_gb"] + P["score_calls_other_slope"] * gb


def ask_gb(kind: str, gb: float, attempt: int = 1) -> int:
    first = max(
        P["score_calls_floor_gb"],
        math.ceil(P["score_calls_headroom"] * peak_gb(kind, gb)),
    )
    return int(min(first * 2 ** (attempt - 1), P["score_calls_memory_max_gb"]))


def hours(gb: float, attempt: int = 1) -> int:
    h = P["score_calls_base_hours"] + P["score_calls_hours_per_gb"] * gb
    return math.ceil(h * attempt)


def main() -> None:
    t = pl.read_csv(HERE / "build_summary/score_calls_measured.tsv", separator="\t")
    t = t.with_columns(
        (pl.col("in_bytes") / 1e9).alias("calls_gb"),
        pl.col("label").str.split(".").list.get(1).alias("index"),
    ).with_columns(
        pl.when(pl.col("index") == "whole")
        .then(pl.lit("whole"))
        .otherwise(pl.lit("other"))
        .alias("kind")
    )
    rows = [
        dict(
            first_ask_gb=ask_gb(k, g),
            second_ask_gb=ask_gb(k, g, 2),
            first_time_h=hours(g),
            at_cap=ask_gb(k, g) >= P["score_calls_memory_max_gb"],
        )
        for k, g in t.select("kind", "calls_gb").iter_rows()
    ]
    t = pl.concat([t, pl.DataFrame(rows)], how="horizontal")
    t = t.with_columns(
        pl.when(pl.col("status") == "COMPLETED")
        .then(pl.col("peak_gb").round(2).cast(pl.Utf8))
        .when(pl.col("exit").cast(pl.Utf8).is_in(["135", "137"]))
        .then(pl.lit(f"killed at {KILL_GB}"))
        .when(pl.col("exit").cast(pl.Utf8) == "1")
        .then(pl.lit("text-column bug"))
        .otherwise(pl.lit("not run"))
        .alias("measured_peak_gb")
    ).sort("index", "calls_gb")
    out = t.select(
        "label",
        "index",
        pl.col("calls_gb").round(3),
        "measured_peak_gb",
        "first_ask_gb",
        "second_ask_gb",
        "first_time_h",
        "at_cap",
    )
    out.write_csv(HERE / "build_summary/score_calls_memory.tsv", separator="\t")

    done = t.filter(pl.col("status") == "COMPLETED")
    killed = t.filter(pl.col("exit").cast(pl.Utf8).is_in(["135", "137"]))
    # done and killed ran the old score_calls.py: context for the figure, not a check of
    # the new asks, which is made below against the new script's own peaks.
    print(
        f"labels {t.height}; finished in the full run (old script) {done.height}; "
        f"killed there {killed.height}"
    )
    print("first ask at the cap:", out.filter(pl.col("at_cap"))["label"].to_list())
    print("first asks summed (GB):", out["first_ask_gb"].sum())

    # Old and new score_calls.py on the same call files: one row per scoreCalls task.
    m = pl.read_csv(
        HERE / "build_summary/score_calls_streaming_measured.tsv", separator="\t"
    )
    m = (
        m.group_by("task_label", "script")
        .agg(
            pl.col("in_bytes").first(),
            pl.col("peak_gib").max(),
            pl.col("wall_s").sum(),
        )
        .with_columns(
            (pl.col("in_bytes") / 1e9).alias("calls_gb"),
            pl.when(pl.col("task_label").str.split(".").list.get(1) == "whole")
            .then(pl.lit("whole"))
            .otherwise(pl.lit("other"))
            .alias("kind"),
        )
        .sort("script", "kind", "calls_gb")
    )
    m = m.with_columns(
        pl.struct("kind", "calls_gb")
        .map_elements(lambda r: ask_gb(r["kind"], r["calls_gb"]), return_dtype=pl.Int64)
        .alias("first_ask_gb"),
        pl.col("calls_gb")
        .map_elements(hours, return_dtype=pl.Int64)
        .alias("first_time_h"),
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(
            m.select(
                "task_label",
                "script",
                pl.col("calls_gb").round(2),
                pl.col("peak_gib").round(2),
                "first_ask_gb",
                (pl.col("wall_s") / 60).round(1).alias("wall_min"),
                "first_time_h",
            )
        )
    new = m.filter(pl.col("script") == "new")
    print(
        "new script: smallest first ask / measured peak:",
        round((new["first_ask_gb"] / new["peak_gib"]).min(), 2),
        "; largest wall time / first time ask:",
        round((new["wall_s"] / 3600 / new["first_time_h"]).max(), 2),
    )

    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 95)
    colour = {"whole": pf.OKABE_ITO["blue"], "other": pf.OKABE_ITO["vermillion"]}
    name = {
        "whole": "whole-protein index",
        "other": "regions, shared-fit and Pfam indexes",
    }
    xs = [0.003 * 1.1**i for i in range(int(math.log(30 / 0.003, 1.1)) + 2)]
    for kind in ("whole", "other"):
        d = done.filter(pl.col("kind") == kind)
        ax.scatter(
            d["calls_gb"],
            d["peak_gb"],
            s=12,
            facecolor="none",
            edgecolor=colour[kind],
            lw=0.6,
            label=f"{name[kind]}: old script, measured peak (full run)",
        )
        k = killed.filter(pl.col("kind") == kind)
        ax.scatter(
            k["calls_gb"],
            [KILL_GB] * k.height,
            s=12,
            marker="^",
            facecolor="none",
            edgecolor=colour[kind],
            lw=0.6,
            label=f"{name[kind]}: old script, killed at 24 GB (needs more)",
        )
        ax.plot(
            xs,
            [OLD[kind][0] + OLD[kind][1] * x for x in xs],
            color=colour[kind],
            lw=0.6,
            ls=(0, (4, 2)),
            label=f"{name[kind]}: old script, fitted peak",
        )
        n = new.filter(pl.col("kind") == kind)
        ax.scatter(
            n["calls_gb"],
            n["peak_gib"],
            s=16,
            color=colour[kind],
            edgecolor=pf.OKABE_ITO["black"],
            lw=0.3,
            zorder=3,
            label=f"{name[kind]}: new script, measured peak",
        )
        ax.plot(
            xs,
            [ask_gb(kind, x) for x in xs],
            color=colour[kind],
            label=f"{name[kind]}: new script, first ask",
        )
    ax.axhline(
        P["score_calls_memory_max_gb"],
        color=pf.OKABE_ITO["black"],
        lw=0.5,
        ls=":",
        label=f"largest ask ({int(P['score_calls_memory_max_gb'])} GB, 1 TB nodes)",
    )
    todo = t.filter((pl.col("status") != "COMPLETED") & (pl.col("calls_gb") > 0))[
        "calls_gb"
    ]
    ax.scatter(
        todo,
        [0.006] * len(todo),
        marker="|",
        s=12,
        color=pf.GREY,
        lw=0.4,
        label="label still to score, at its call-file size",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.003, 40)
    ax.set_ylim(0.005, 2000)
    for axis, ticks in (
        (ax.xaxis, [0.01, 0.1, 1, 10]),
        (ax.yaxis, [0.01, 0.1, 1, 10, 100, 1000]),
    ):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_title(
        "Reading the call files in pieces: about 1.5 GB of memory per GB of calls,\n"
        "down from 18-32 GB; the largest label (27 GB of calls) peaks at 40 GB",
        loc="left",
    )
    ax.set_xlabel("Gzipped call files of one label (GB)")
    ax.set_ylabel("scoreCalls memory (GB)")
    pf.shared_legend(fig, ncol=1)
    pf.save(fig, str(HERE / "build_summary/score_calls_memory"))
    # 8-bit palette PNG; this branch's pubfig.save predates its own conversion step.
    png = HERE / "build_summary/score_calls_memory.png"
    with Image.open(png) as image:
        dpi = image.info.get("dpi")
        small = image.convert("RGB").quantize(colors=256, dither=Image.Dither.NONE)
    small.save(png, optimize=True, **({"dpi": dpi} if dpi else {}))


if __name__ == "__main__":
    main()
