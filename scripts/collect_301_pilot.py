#!/usr/bin/env python3
"""Notebook 301 pilot: one table row per kmerseek setting from the pilot's own files.

Standard library only. Reads, under --pilot, each setting folder's index.time and
search.time (GNU time -v), index.log and search.log (Karlin-Altschul fit, targets loaded),
check.tsv (from pilot_301_kmerseek.py check) and regions_filtered.tsv.gz, plus the
one-query timing logs one_query.<rep>.log. Writes --out (TSV).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path


def gnu_time(path: Path) -> tuple[float | None, float | None]:
    """(wall seconds, peak RSS in GB) from a GNU time -v file."""
    if not path.exists():
        return None, None
    t = path.read_text()
    wall = rss = None
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    if m:
        parts = [float(x) for x in m.group(1).split(":")]
        wall = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    if m:
        rss = int(m.group(1)) / 1024**2
    return wall, rss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for d in sorted(p for p in a.pilot.iterdir() if (p / "index.time").exists()):
        alphabet, k = d.name.split(".")[0], int(d.name.split(".")[1][1:])
        iw, ir = gnu_time(d / "index.time")
        sw, sr = gnu_time(d / "search.time")
        slog = (d / "search.log").read_text() if (d / "search.log").exists() else ""
        m = re.search(r"Loaded search cache: (\d+) targets", slog)
        refused = "no Karlin-Altschul fit" in slog
        row = {
            "setting": d.name,
            "alphabet": alphabet,
            "ksize": k,
            "karlin_altschul_fit": "refused" if refused else "fitted",
            "targets_loaded": int(m.group(1)) if m else None,
            "index_wall_s": iw,
            "index_peak_rss_gb": None if ir is None else round(ir, 1),
            "search_43_sequences_wall_s": None if refused else sw,
            "search_peak_rss_gb": None if refused or sr is None else round(sr, 1),
        }
        chk = d / "check.tsv"
        if chk.exists():
            for r in csv.DictReader(open(chk), delimiter="\t"):
                row[f"rows_{r['kind']}"] = int(r["rows"])
                row[f"rows_finite_region_evalue_{r['kind']}"] = int(
                    r["rows_finite_region_evalue"]
                )
                row[f"rows_region_subseq_matches_query_{r['kind']}"] = int(
                    r["rows_region_subseq_matches_query"]
                )
        f = d / "regions_filtered.tsv.gz"
        if f.exists():
            with gzip.open(f, "rt") as fh:
                row["rows_kept_by_filter_e10_on_window"] = sum(1 for _ in fh) - 1
        rows.append(row)
    for rep in (1, 2):
        log = a.pilot / f"one_query.{rep}.log"
        if log.exists():
            m = re.search(r"rep \d+: ([\d.]+) s wall, (\d+) KB", log.read_text())
            if m:
                rows.append(
                    {
                        "setting": f"hp_lehninger2.k19 one query (P12241), 4 cores, run {rep}",
                        "alphabet": "hp_lehninger2",
                        "ksize": 19,
                        "search_1_query_4_cores_wall_s": float(m.group(1)),
                        "search_peak_rss_gb": round(int(m.group(2)) / 1024**2, 1),
                    }
                )
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(open(a.out).read())


if __name__ == "__main__":
    main()
