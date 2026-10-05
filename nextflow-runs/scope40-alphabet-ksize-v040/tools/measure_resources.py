#!/usr/bin/env python3
"""Peak memory and run time per process and alphabet-ksize pair, from Nextflow trace files.

  measure_resources.py <out.tsv> <trace.txt>...

main.nf reads the output (assets/resources_measured.tsv) to size each task's first attempt:
the measured peak of the nearest pair at or below its k, times 1.5. Per pair, the largest
COMPLETED attempt over every setting and every trace counts. A task that FAILED with a
memory kill (exit 137, or 135) or a time limit (140, or no exit at all) is listed too, at
twice what it asked for, so the next run starts above the ask that was killed.

Traces from before 2026-10-03 ran search and scoring as two processes (kmerseekSearch,
scoreArm); searchAndScore does both. For those, a pair's searchAndScore row is the larger
of the two peaks and the sum of the two run times. The old scoring read every region row
and searchAndScore reads one row per pair, so this errs high. A pair measured directly as
searchAndScore uses that measurement instead.
"""

OLD_PARTS = {"kmerseekSearch": "searchAndScore", "scoreArm": "searchAndScore"}

import csv
import re
import sys

UNITS = {"B": 1 / 2**30, "KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1.0, "TB": 2**10}


def gb(text):
    m = re.fullmatch(r"([\d.]+)\s*([KMGT]?B)", text.strip())
    return float(m.group(1)) * UNITS[m.group(2)] if m else None


def hours(text):
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(ms|s|m|h|d)", text):
        total += float(value) * {"ms": 1 / 3.6e6, "s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}[unit]
    return total if text.strip() not in ("", "-") else None


def main():
    out, traces = sys.argv[1], sys.argv[2:]
    best = {}
    # A search kmerseek refuses (no fit for that penalty) ends in a second with a tiny peak,
    # so a pair counts as measured only once its exact setting, which is never refused, has
    # completed: on 2026-10-02 gbmr7 k13 had only refused settings done and read as 17 MB.
    exact_done = set()
    for path in traces:
        with open(path) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["process"] not in ("kmerseekIndex", "searchAndScore", *OLD_PARTS):
                    continue
                m = re.match(r"(.+?)\.k(\d+)(?:\.|$)", r["tag"])
                if not m:
                    continue
                key = (r["process"], m.group(1), int(m.group(2)))  # parts merged below
                if r["process"] != "kmerseekIndex" and r["tag"].endswith(".exact") and \
                        r["status"] in ("COMPLETED", "CACHED"):
                    exact_done.add((OLD_PARTS.get(r["process"], r["process"]), m.group(1), int(m.group(2))))
                if r["status"] in ("COMPLETED", "CACHED"):
                    peak, t = gb(r.get("peak_rss", "")), hours(r.get("realtime", ""))
                    evidence = "completed"
                elif r["status"] == "FAILED" and r["exit"] in ("137", "135"):
                    # Killed for memory: it needed more than it asked; the time it ran is a floor.
                    peak, t = 2 * (gb(r.get("memory", "")) or 0), hours(r.get("realtime", ""))
                    evidence = f"memory kill at {r.get('memory', '?')}"
                elif r["status"] == "FAILED" and r["exit"] in ("140", "-", "") and \
                        (hours(r.get("realtime", "")) or 0) >= 0.9 * (hours(r.get("time", "")) or 1e9):
                    # A time limit only if it ran to its ask: an aborted run also leaves "-".
                    peak, t = gb(r.get("peak_rss", "")) or 0, 2 * (hours(r.get("time", "")) or 0)
                    evidence = f"time limit at {r.get('time', '?')}"
                else:
                    continue
                if peak is None:
                    continue
                cur = best.get(key)
                if cur is None or peak > cur["peak_gb"]:
                    best[key] = {"peak_gb": peak, "realtime_h": t or 0.0, "evidence": evidence}
                else:
                    cur["realtime_h"] = max(cur["realtime_h"], t or 0.0)
    # Merge the two old parts into searchAndScore: larger peak, summed run time.
    # A direct searchAndScore measurement replaces the estimate built from the old parts.
    merged = {}
    for (proc, alph, k), v in best.items():
        if proc not in OLD_PARTS:
            continue
        cur = merged.get((OLD_PARTS[proc], alph, k))
        if cur is None:
            merged[(OLD_PARTS[proc], alph, k)] = dict(v)
        else:
            cur["peak_gb"] = max(cur["peak_gb"], v["peak_gb"])
            cur["realtime_h"] += v["realtime_h"]
            if v["evidence"] != "completed":
                cur["evidence"] = v["evidence"]
    for key, v in best.items():
        if key[0] not in OLD_PARTS:
            merged[key] = dict(v)
    best = {k: v for k, v in merged.items() if k[0] == "kmerseekIndex" or k in exact_done
            or v["evidence"] != "completed"}
    with open(out, "w", newline="") as fh:
        fh.write("# Written by tools/measure_resources.py from: " + " ".join(traces) + "\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["process", "alphabet", "ksize", "peak_gb", "realtime_h", "evidence"])
        for (proc, alph, k), v in sorted(best.items()):
            w.writerow([proc, alph, k, f"{v['peak_gb']:.3f}", f"{v['realtime_h']:.4f}", v["evidence"]])
    print(f"{len(best)} process x pair rows from {len(traces)} trace file(s) -> {out}")


if __name__ == "__main__":
    main()
