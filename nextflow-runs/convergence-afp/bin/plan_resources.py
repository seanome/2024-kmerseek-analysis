#!/usr/bin/env python3
"""First-attempt memory and time for every kmerseek setting, from measured history.

Standard library only and no type hints, so `make plan` runs with any python3 from 3.6 on,
including the system one on a Sherlock login node.

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

Memory is capped at --cap-gb, the largest node; a setting whose ask reaches the cap is
listed so it can be dropped before launch rather than fail on every attempt.

Writes resources.tsv (setting_label, memory_gb, time_h, basis) and prints the summary
table that goes with the launch command.
"""

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MEM_KILLS = {"137", "135"}
TIME_KILLS = {"140", "-", "", "2147483647"}
UNITS_GB = {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}
UNITS_S = {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400}
DARK_INDEX = re.compile(
    r"^kmerseekIndex \(minus_[^.]+\.(?P<a>[a-z0-9_]+)\.k(?P<k>\d+)(?:\.s(?P<s>\d+))?\.lc"
)


def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t"))


def gb(x):
    if x in (None, "", "-"):
        return None
    v, u = x.split()
    return float(v) * UNITS_GB[u]


def hours(x):
    if x in (None, "", "-"):
        return None
    return sum(float(v) * UNITS_S[u] for v, u in re.findall(r"([\d.]+)(ms|s|m|h|d)", x)) / 3600


def bump(d, key, value):
    if value is not None and (d.get(key) is None or value > d[key]):
        d[key] = value


def own_asks(paths):
    mem, tim, failed = {}, {}, 0
    for p in paths:
        for r in read_tsv(p):
            if not (r.get("name") or "").startswith("kmerseekRun"):
                continue
            lab, status, exit_ = r["tag"], r.get("status"), (r.get("exit") or "").strip()
            if status == "COMPLETED":
                peak, run = gb(r.get("peak_rss")), hours(r.get("realtime"))
                bump(mem, lab, 1.25 * peak if peak else None)
                bump(tim, lab, 1.5 * run if run else None)
                continue
            failed += 1
            if exit_ in MEM_KILLS and gb(r.get("memory")):
                bump(mem, lab, 2 * gb(r["memory"]))
            elif exit_ in TIME_KILLS and hours(r.get("time")):
                bump(tim, lab, 2 * hours(r["time"]))
    return mem, tim, failed


def dark_asks(measured, traces):
    mem, tim = {}, {}
    for r in read_tsv(measured / "kmerseek_index_memory_measured.tsv"):
        bump(mem, (r["alphabet"], int(r["ksize"]), int(r["scaled"])), float(r["first_ask_gb"]))
    for p in traces:
        n = 0
        for r in read_tsv(p):
            m = DARK_INDEX.match(r.get("name") or "")
            if not m or r.get("status") != "COMPLETED":
                continue
            key = (m["a"], int(m["k"]), int(m["s"] or 1))
            peak, run = gb(r.get("peak_rss")), hours(r.get("realtime"))
            bump(mem, key, 1.25 * peak if peak else None)
            bump(tim, key, 1.5 * run if run else None)
            n += 1
        print(f"{p}: {n} finished kmerseekIndex tasks", file=sys.stderr)
    return mem, tim


def model_gb(bits, scaled, ka_queries):
    fit = 300.0 * 2 ** (-(bits - 12.0) / 2.5) * scaled**-0.7 * (ka_queries / 500.0)
    return 1.4 * max(45.0, fit)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--dark-measured", type=Path, required=True)
    ap.add_argument("--trace", type=Path, nargs="*", default=[],
                    help="this pipeline's own Nextflow traces")
    ap.add_argument("--dark-trace", type=Path, nargs="*", default=[],
                    help="the dark-set run's Nextflow traces")
    ap.add_argument("--floor-gb", type=float, default=48)
    ap.add_argument("--floor-h", type=float, default=4)
    ap.add_argument("--cap-gb", type=float, default=500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    settings = read_tsv(args.settings)
    own_mem, own_tim, n_failed = own_asks(args.trace)
    dark_mem, dark_tim = dark_asks(args.dark_measured, args.dark_trace)
    grp_mem, grp_tim = {}, {}
    for s in settings:
        g = (s["alphabet"], s["scaled"])
        bump(grp_mem, g, own_mem.get(s["setting_label"]))
        bump(grp_tim, g, own_tim.get(s["setting_label"]))

    def pick(own, others):
        if own is not None:
            return "own attempts", own
        others = [(b, v) for b, v in others if v is not None]
        return max(others, key=lambda x: x[1]) if others else (None, None)

    rows, at_cap = [], []
    for s in settings:
        lab, g = s["setting_label"], (s["alphabet"], s["scaled"])
        key = (s["alphabet"], int(s["ksize"]), int(s["scaled"]))
        mem_b, mem = pick(own_mem.get(lab), [("same alphabet and scaled in this run", grp_mem.get(g)),
                                             ("dark-set index task", dark_mem.get(key))])
        if mem_b is None:
            mem_b = "dark-set model"
            mem = model_gb(float(s["keyspace_bits"]), int(s["scaled"]), int(s["ka_queries"]))
        tim_b, tim = pick(own_tim.get(lab), [("same alphabet and scaled in this run", grp_tim.get(g)),
                                             ("dark-set index task", dark_tim.get(key))])
        if tim_b is None:
            tim_b, tim = "floor", 0.0
        rows.append({
            "setting_label": lab,
            "memory_gb": int(min(args.cap_gb, max(args.floor_gb, math.ceil(mem)))),
            "time_h": int(max(args.floor_h, math.ceil(tim))),
            "basis": f"memory: {mem_b if mem > args.floor_gb else 'floor'}; "
                     f"time: {tim_b if tim > args.floor_h else 'floor'}",
        })
        if mem >= args.cap_gb:
            at_cap.append(lab)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} settings; {len(args.trace)} own trace file(s), "
          f"{len(args.dark_trace)} dark-set trace file(s); {n_failed} earlier failed attempt(s) read")
    print(f"{'n':>4}  {'max GB':>6}  {'max h':>5}  basis")
    by = defaultdict(list)
    for r in rows:
        by[r["basis"]].append(r)
    for b, rs in sorted(by.items()):
        print(f"{len(rs):>4}  {max(r['memory_gb'] for r in rs):>6}  "
              f"{max(r['time_h'] for r in rs):>5}  {b}")
    asks = Counter(r["memory_gb"] for r in rows)
    print("first memory asks (GB):",
          ", ".join(f"{k}: {v}" for k, v in sorted(asks.items())))
    print(f"sum of first asks: {sum(r['memory_gb'] for r in rows)} GB, "
          f"{sum(r['time_h'] for r in rows)} task-hours at most")
    if at_cap:
        print(f"{len(at_cap)} setting(s) at the {args.cap_gb:.0f} GB cap: {', '.join(at_cap)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
