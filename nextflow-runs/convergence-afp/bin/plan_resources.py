#!/usr/bin/env python3
"""First-attempt memory and time for every kmerseek setting, from measured history.

One kmerseek task here builds the index of all of Swiss-Prot for one setting, fits the
E-value statistics on it, searches the queries and deletes the index. Its first ask is,
in this order:

  1. this pipeline's own earlier attempts at the setting (Nextflow traces given with
     --trace): a finished attempt asks 1.25 x its peak memory and 1.5 x its run time; an
     attempt killed for memory (exit 137 or 135) or time (140, or no exit code) asks
     twice what it was given. The largest ask over its attempts is used.
  2. otherwise, the larger of
     a. the largest such ask among this pipeline's attempts at the same alphabet and
        scaled value (the pilot runs each alphabet's lowest and highest k, which bracket
        the rest), and
     b. the dark-set run's index tasks for the same alphabet, k and scaled. Each built
        the index and ran the same E-value fit on a reference of reviewed Swiss-Prot
        minus one clade (about 572_700 sequences, the size of this one). Read from its
        Nextflow traces (--dark-trace: a finished attempt asks 1.25 x its peak memory and
        1.5 x its run time) and from its committed table of raised first asks
        (--dark-measured). Its search tasks are not used: they searched 2_000 queries at
        a time, and this run searches 246 short ones.
  3. a setting with no history anywhere asks what the dark-set pipeline's model asks for
     its index task: 1.4 x max(45, 300 x 2^(-(bits - 12) / 2.5) x scaled^-0.7 x
     ka_queries / 500) GB, bits being k x log2(classes). Its parameters were fitted on
     that run's index tasks (kmerseek_index_build_gb, kmerseek_ka_fit_memory_* in its
     main.nf at ec91da5). Time has no model and takes the floor.
  4. never below --floor-gb and --floor-h.
 Memory is
capped at --cap-gb, the largest node; a setting whose ask reaches the cap is listed so it
can be dropped before launch rather than fail on every attempt.

Writes resources.tsv (setting_label, memory_gb, time_h, basis) and prints the summary
table that goes with the launch command.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import polars as pl

MEM_KILLS = {"137", "135"}
TIME_KILLS = {"140", "-", "", "2147483647"}


def gb(x: str | None) -> float | None:
    if x in (None, "", "-"):
        return None
    v, u = x.split()
    return float(v) * {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}[u]


def hours(x: str | None) -> float | None:
    if x in (None, "", "-"):
        return None
    tot = 0.0
    for v, u in re.findall(r"([\d.]+)(ms|s|m|h|d)", x):
        tot += float(v) * {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400}[u]
    return tot / 3600


def trace_asks(paths: list[Path]) -> pl.DataFrame:
    rows = []
    for p in paths:
        t = pl.read_csv(p, separator="\t", infer_schema_length=0)
        t = t.filter(pl.col("name").str.starts_with("kmerseekRun"))
        for r in t.iter_rows(named=True):
            exit_, status = (r.get("exit") or "").strip(), r.get("status")
            mem_given, time_given = gb(r.get("memory")), hours(r.get("time"))
            peak, run = gb(r.get("peak_rss")), hours(r.get("realtime"))
            mem = tim = None
            if status == "COMPLETED":
                mem = 1.25 * peak if peak else None
                tim = 1.5 * run if run else None
            elif exit_ in MEM_KILLS and mem_given:
                mem = 2 * mem_given
            elif exit_ in TIME_KILLS and time_given:
                tim = 2 * time_given
            rows.append(
                {
                    "setting_label": r["tag"],
                    "status": status,
                    "exit": exit_,
                    "mem_ask": mem,
                    "time_ask": tim,
                    "trace": p.name,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "setting_label": pl.Utf8,
            "status": pl.Utf8,
            "exit": pl.Utf8,
            "mem_ask": pl.Float64,
            "time_ask": pl.Float64,
            "trace": pl.Utf8,
        },
    )


def dark_set(measured: Path, traces: list[Path]) -> pl.DataFrame:
    """alphabet|ksize|scaled -> the dark-set index tasks' largest memory and time asks."""
    idx = pl.read_csv(measured / "kmerseek_index_memory_measured.tsv", separator="\t")
    parts = [
        idx.select(
            key=pl.format("{}|{}|{}", "alphabet", "ksize", "scaled"),
            dark_mem=pl.col("first_ask_gb").cast(pl.Float64),
            dark_time=pl.lit(None, dtype=pl.Float64),
        )
    ]
    pat = re.compile(
        r"^kmerseekIndex \(minus_[^.]+\.(?P<a>[a-z0-9_]+)\.k(?P<k>\d+)"
        r"(?:\.s(?P<s>\d+))?\.lc"
    )
    for p in traces:
        t = pl.read_csv(p, separator="\t", infer_schema_length=0).filter(
            pl.col("status") == "COMPLETED"
        )
        rows = []
        for r in t.iter_rows(named=True):
            m = pat.match(r["name"] or "")
            if not m:
                continue
            peak, run = gb(r.get("peak_rss")), hours(r.get("realtime"))
            rows.append(
                {
                    "key": f"{m['a']}|{m['k']}|{m['s'] or 1}",
                    "dark_mem": 1.25 * peak if peak else None,
                    "dark_time": 1.5 * run if run else None,
                }
            )
        print(f"{p}: {len(rows)} finished kmerseekIndex tasks", file=sys.stderr)
        if rows:
            parts.append(
                pl.DataFrame(
                    rows,
                    schema={
                        "key": pl.Utf8,
                        "dark_mem": pl.Float64,
                        "dark_time": pl.Float64,
                    },
                )
            )
    return (
        pl.concat(parts)
        .group_by("key")
        .agg(pl.col("dark_mem").max(), pl.col("dark_time").max())
    )


def model_gb(bits: float, scaled: int, ka_queries: int) -> float:
    fit = 300.0 * 2 ** (-(bits - 12.0) / 2.5) * scaled**-0.7 * (ka_queries / 500.0)
    return 1.4 * max(45.0, fit)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--dark-measured", type=Path, required=True)
    ap.add_argument(
        "--trace",
        type=Path,
        nargs="*",
        default=[],
        help="this pipeline's own Nextflow traces",
    )
    ap.add_argument(
        "--dark-trace",
        type=Path,
        nargs="*",
        default=[],
        help="the dark-set run's Nextflow traces",
    )
    ap.add_argument("--floor-gb", type=float, default=48)
    ap.add_argument("--floor-h", type=float, default=4)
    ap.add_argument("--cap-gb", type=float, default=500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s = pl.read_csv(args.settings, separator="\t").with_columns(
        key=pl.format("{}|{}|{}", "alphabet", "ksize", "scaled"),
        group=pl.format("{}|{}", "alphabet", "scaled"),
    )
    own = trace_asks(args.trace)
    own_best = own.group_by("setting_label").agg(
        pl.col("mem_ask").max(), pl.col("time_ask").max()
    )
    grp = (
        own.join(s.select("setting_label", "group"), on="setting_label")
        .group_by("group")
        .agg(
            pl.col("mem_ask").max().alias("grp_mem"),
            pl.col("time_ask").max().alias("grp_time"),
        )
    )
    dark = dark_set(args.dark_measured, args.dark_trace)

    t = (
        s.join(own_best, on="setting_label", how="left")
        .join(grp, on="group", how="left")
        .join(dark, on="key", how="left")
    )
    rows = []
    for r in t.iter_rows(named=True):

        def pick(own_v, others):
            if own_v is not None:
                return "own attempts", own_v
            others = [(b, v) for b, v in others if v is not None]
            return max(others, key=lambda x: x[1]) if others else (None, None)

        mem_b, mem = pick(
            r["mem_ask"],
            [
                ("same alphabet and scaled in this run", r["grp_mem"]),
                ("dark-set index task", r["dark_mem"]),
            ],
        )
        if mem_b is None:
            mem_b, mem = "dark-set model", model_gb(
                r["keyspace_bits"], r["scaled"], r["ka_queries"]
            )
        tim_b, tim = pick(
            r["time_ask"],
            [
                ("same alphabet and scaled in this run", r["grp_time"]),
                ("dark-set index task", r["dark_time"]),
            ],
        )
        if tim_b is None:
            tim_b, tim = "floor", 0.0
        rows.append(
            {
                "setting_label": r["setting_label"],
                "memory_gb": int(min(args.cap_gb, max(args.floor_gb, math.ceil(mem)))),
                "time_h": int(max(args.floor_h, math.ceil(tim))),
                "basis": f"memory: {mem_b if mem > args.floor_gb else 'floor'}; "
                f"time: {tim_b if tim > args.floor_h else 'floor'}",
                "at_cap": mem >= args.cap_gb,
            }
        )
    out = pl.DataFrame(rows)
    out.drop("at_cap").write_csv(args.out, separator="\t")

    n_fail = own.filter(pl.col("status") != "COMPLETED").height
    print(
        f"{out.height} settings; {len(args.trace)} own trace file(s), "
        f"{len(args.dark_trace)} dark-set trace file(s); "
        f"{n_fail} earlier failed attempt(s) read"
    )
    print(
        out.group_by("basis")
        .agg(
            pl.len().alias("n_settings"),
            pl.col("memory_gb").max().alias("max_gb"),
            pl.col("time_h").max().alias("max_h"),
        )
        .sort("basis")
    )
    cap = out.filter(pl.col("at_cap"))
    if cap.height:
        print(f"{cap.height} setting(s) at the {args.cap_gb:.0f} GB cap:")
        print(cap.select("setting_label", "basis"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
