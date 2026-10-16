#!/usr/bin/env python3
"""Figure: each search tool's Fmax rows beside the no-search baseline, two truth cells.

a  Swiss-Prot TRANSMEM features: every search tool row against the Kyte-Doolittle scan.
b  Pfam-A, all instances: every search tool row against the Pfam family frequency floor.

One dot per search-tool row (setting x target proteome x dedup mode) from
results/leaderboard/results.parquet (PR 148); one dot per baseline from
no_search_baselines.parquet (scripts/build_no_search_baseline_table.py). The baseline
row sits at the bottom of each panel, as on the leaderboard.

    python scripts/plot_no_search_baselines.py --baselines <...>/no_search_baselines.parquet \\
        --results results/leaderboard/results.parquet --out figures/no_search_baselines_midi_plus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

EXPERIMENT = "qfo_region_midi_plus"
NAME = {"kmerseek": "kmerseek", "hhblits": "HHblits", "hmmer3_phmmer": "phmmer",
        "hmmer3_jackhmmer": "jackhmmer", "mmseqs2_seqseq": "MMseqs2",
        "mmseqs2_iterative": "MMseqs2 iterative", "foldseek": "Foldseek",
        "prostt5": "ProstT5", "hmmscan": "hmmscan (Pfam HMMs)",
        "kyte_doolittle": "Kyte-Doolittle scan", "pfam_family_frequency": "Pfam family frequency"}
PANELS = [
    ("a", "Swiss-Prot features", "feature_type", "TRANSMEM", "kyte_doolittle",
     "Fmax on Swiss-Prot TRANSMEM"),
    ("b", "Pfam-A domain instances", "all", "all", "pfam_family_frequency",
     "Fmax on Pfam-A domains"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baselines", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    base = pl.read_parquet(a.baselines).filter(pl.col("applicable")
                                               & (pl.col("metric") == "fmax"))
    res = pl.read_parquet(a.results).filter((pl.col("experiment") == EXPERIMENT)
                                            & (pl.col("metric") == "fmax"))
    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)
    search_c, base_c = pf.OKABE_ITO["blue"], pf.OKABE_ITO["vermillion"]
    for ax, (letter, truth, axis, cell, btool, xlabel) in zip(axs, PANELS):
        s = res.filter((pl.col("truth_source") == truth) & (pl.col("stratum_axis") == axis)
                       & (pl.col("stratum_bin") == cell))
        b = base.filter((pl.col("tool") == btool) & (pl.col("truth_source") == truth)
                        & (pl.col("stratum_axis") == axis) & (pl.col("stratum_bin") == cell))
        if b.height != 1:
            raise SystemExit(f"panel {letter}: {b.height} baseline rows")
        order = (s.group_by("tool").agg(pl.col("value").max()).sort("value")["tool"]
                 .to_list())
        # Baseline at the bottom (y = 0), search tools above it, best on top.
        for y, tool in enumerate(order, start=1):
            v = s.filter(pl.col("tool") == tool)["value"]
            ax.scatter(v, [y] * len(v), s=6, color=search_c, alpha=0.5, linewidths=0,
                       label="search tool: one setting on one target proteome")
        ax.scatter(b["value"], [0], s=14, color=base_c, linewidths=0,
                   label="no search: the query sequence alone")
        ax.axvline(b["value"][0], color=base_c, lw=0.5, ls=":")
        n = int(s["n_instances"][0])
        ax.set_yticks(range(len(order) + 1),
                      [NAME[btool]] + [f"{NAME.get(t, t)} ({s.filter(pl.col('tool') == t).height})"
                                       for t in order])
        ax.set_xlabel(f"{xlabel} (n = {n} instances)")
        ax.set_xlim(0, 1)
        pf.panel_label(ax, letter)
        print(f"{letter}: {NAME[btool]} {b['value'][0]:.3f}; "
              + "; ".join(f"{t} best {s.filter(pl.col('tool') == t)['value'].max():.3f}"
                          for t in reversed(order)))
    pf.shared_legend(fig)
    pf.save(fig, a.out)


if __name__ == "__main__":
    main()
