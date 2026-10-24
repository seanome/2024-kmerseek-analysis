#!/usr/bin/env python3
"""First-attempt memory and time of every kmerseek index and search in the time-split
pipeline, computed the way main.nf computes them, with the evidence behind each ask.

    predict_resources.py --settings settings.tsv --n-chunks 1 [--trace trace.pilot.txt]

Index: 1.4 x max(45 GB build, Karlin-Altschul fit model), at least 48 GB, at least the
dark-set run's measured first ask for the same alphabet, k and scaled 1. Search: 1.5 x the
largest peak a dark-set chunk of 2_000 queries reached for the same alphabet and k; with
no history, the spectrum-load model at run time (shown as "model"). Time: 1.5 x the
longest dark-set chunk, at least 2 h; 4 h with no history. A search the dark-set runs saw
killed for memory asks at least that run's next ask (assets/kmerseek_search_memory_measured.tsv).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import polars as pl

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CLASSES = dict(protein20=20, uniprot18=18, hsdm17=17, wass14=14, mmseqs12=12, sdm12=12,
               funcgroups8=8, gbmr7=7, dayhoff6=6, wwmj5=5, gbmr4=4, polarity4=4,
               hp_lehninger_hpc3=3)


def predict(settings: Path) -> pl.DataFrame:
    st = pl.read_csv(settings, separator="\t")
    im: dict[tuple, float] = {}
    for line in (ASSETS / "kmerseek_index_memory_measured.tsv").read_text().splitlines()[1:]:
        c = line.split("\t")
        key = (c[0], int(c[1]), int(c[2]))
        im[key] = max(im.get(key, 0.0), float(c[3]))
    sm = {(r["alphabet"], r["k"]): r
          for r in pl.read_csv(ASSETS / "darkset_search_measured_s1.csv").iter_rows(named=True)}
    killed: dict[tuple, float] = {}
    for line in (ASSETS / "kmerseek_search_memory_measured.tsv").read_text().splitlines()[1:]:
        c = line.split("\t")
        m = re.match(r"^minus_[A-Za-z]+\.(\w+)\.k(\d+)\.lc\w+$", c[0])
        if m:
            key = (m.group(1), int(m.group(2)))
            killed[key] = max(killed.get(key, 0.0), float(c[1]))
    rows = []
    for a, k, _c in st.iter_rows():
        bits = k * math.log2(CLASSES.get(a, 2))
        nq = 500 if bits <= 30 else int(round(min(3000, 500 * 2 ** ((bits - 30) / 3)) / 50) * 50)
        fit = 300 * 2 ** (-(bits - 12) / 2.5) * (nq / 500)
        model = max(48.0, 1.4 * max(45.0, fit))
        meas = im.get((a, k, 1), 0.0)
        m = sm.get((a, k))
        rows.append(dict(
            alphabet=a, k=k, keyspace_bits=round(bits, 1), ka_queries=nq,
            index_ask_gb=round(min(500.0, max(model, meas))),
            index_basis="dark-set measured" if meas > model else "model",
            search_ask_gb=(round(max(24.0, 1.5 * m["gb_max"], killed.get((a, k), 0.0))) if m
                           else round(killed[(a, k)]) if (a, k) in killed else None),
            search_hours=max(2, math.ceil(1.5 * m["h_max"])) if m else 4,
            search_basis=("dark-set killed ask" if killed.get((a, k), 0) > (1.5 * m["gb_max"] if m else 0)
                          else "dark-set peak" if m else "model at run time"),
            darkset_chunk_hours_median=round(m["h_med"], 2) if m else None))
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    t = predict(a.settings)
    if a.out:
        t.write_csv(a.out)
    n_search = 2 * t.height * a.n_chunks
    with pl.Config(tbl_rows=300, tbl_width_chars=220):
        print(t.select(
            pl.len().alias("indexes"),
            pl.col("index_ask_gb").max().alias("index_ask_max_gb"),
            pl.col("index_ask_gb").median().alias("index_ask_median_gb"),
            (pl.col("index_basis") != "model").sum().alias("index_asks_from_measurement"),
            pl.col("search_ask_gb").max().alias("search_ask_max_gb"),
            pl.col("search_ask_gb").median().alias("search_ask_median_gb"),
            pl.col("search_ask_gb").is_null().sum().alias("alphabet_k_without_search_history")))
        print(f"searches: {n_search} ({t.height} indexes x 2 settings x {a.n_chunks} chunks)")
        print(t.sort("index_ask_gb", descending=True).head(8))
        print(t.sort("search_ask_gb", descending=True, nulls_last=True).head(8))
        print(t.filter(pl.col("search_ask_gb").is_null()).group_by("alphabet").agg("k").sort("alphabet"))


if __name__ == "__main__":
    main()
