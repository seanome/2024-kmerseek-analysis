#!/usr/bin/env python3
"""Predicted first asks and run times of the full time-split run (`make full`), per kmerseek
search, from the pilot's own trace.

    predict_full_run.py <pilot trace> <pilot_search_measured.csv> <darkset_search_measured_s1.csv> \
        <leaderboard_settings.tsv> <n full queries> <chunk size> <out.csv>

Mirrors main.nf's searchFirstGb and searchHours: the pilot searched the same indexes with
the same rule, so the pilot's ask (trace `memory`) is the full run's base ask, raised to
1.5x the pilot's peak where that is larger. Time: the dark-set rule (1.5x its longest chunk,
at least 2 h; 4 h with no measurement), raised to 1.5x the pilot's run time scaled to
2,000 queries. The predicted run time of a full-run chunk is the pilot's run time scaled by
queries; the smoke and pilot runs of hp_pbotc_1st_ed2 k29 (3.3 min for 4 queries, 4 min
for 440) show that most of a high-k search is loading the index, so this over-predicts.
Every column named predicted_* is a prediction, not a measurement.
"""

import math
import re
import sys

import polars as pl


def gib(text: str) -> float:
    num, unit = text.split()
    return float(num) * {"MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}[unit]


def main() -> None:
    trace, pilot_csv, dark_csv, settings, n_q, chunk, out = sys.argv[1:]
    n_q, chunk = int(n_q), int(chunk)
    chunks = [chunk] * (n_q // chunk) + ([n_q % chunk] if n_q % chunk else [])
    pilot = pl.read_csv(pilot_csv)
    dark = {}
    for r in pl.read_csv(dark_csv).iter_rows(named=True):
        dark[(r["alphabet"], r["k"])] = r[
            "h_max"
        ]  # main.nf: last row per alphabet|k wins
    t = pl.read_csv(trace, separator="\t", infer_schema=False).filter(
        pl.col("name").str.starts_with("kmerseekSearch ")
    )
    ask = {}
    for name, mem in t.select("name", "memory").iter_rows():
        m = re.match(r"kmerseekSearch \((\w+)\.k(\d+)\.(exact|extend)\.", name)
        ask[(m.group(1), int(m.group(2)), m.group(3))] = gib(mem)
    rows = []
    st = pl.read_csv(settings, separator="\t")
    for a, k in st.select("alphabet", "ksize").iter_rows():
        for arm in ("exact", "extend"):
            p = pilot.filter(
                (pl.col("alphabet") == a) & (pl.col("k") == k) & (pl.col("arm") == arm)
            ).row(0, named=True)
            first_gb = max(ask[(a, k, arm)], 1.5 * p["peak_rss_gb"])
            h = dark.get((a, k))
            first_h = max(2, math.ceil(1.5 * h)) if h is not None else 4
            first_h = max(
                first_h, math.ceil(1.5 * p["realtime_min"] / 60 * chunk / p["queries"])
            )
            pred_min = [p["realtime_min"] * c / p["queries"] for c in chunks]
            rows.append(
                {
                    "alphabet": a,
                    "k": k,
                    "arm": arm,
                    "pilot_ask_gb": round(ask[(a, k, arm)], 1),
                    "pilot_peak_gb": p["peak_rss_gb"],
                    "pilot_minutes_440_queries": p["realtime_min"],
                    "first_ask_gb": round(first_gb, 1),
                    "first_ask_hours": min(first_h, 24),
                    "predicted_minutes_2000_queries": round(max(pred_min), 1),
                    "predicted_cpu_hours_all_chunks": round(sum(pred_min) / 60 * 8, 1),
                    "chunks": len(chunks),
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(out)
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=250):
        print(df)
    print(
        f"{df.height} searches x {len(chunks)} chunks = {df.height * len(chunks)} tasks; "
        f"predicted node-hours (8 CPUs, scaled by queries): "
        f"{df['predicted_cpu_hours_all_chunks'].sum() / 8:.1f}; largest first ask "
        f"{df['first_ask_gb'].max()} GB; first asks over 1.5x pilot peak in all rows: "
        f"{(df['first_ask_gb'] >= 1.5 * df['pilot_peak_gb'] - 0.05).all()}"
    )


if __name__ == "__main__":
    main()
