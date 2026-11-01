#!/usr/bin/env python3
"""Notebook 357: Benjamini-Hochberg q-values, the significant anchors and the counts.

Replaces `collect` in clade_splash.py (still there, unused) at the full run's scale: the
128 test tables are 5.9 GB of parquet, and reading them all at once was killed at 32 GB.
Here each (setting, gap, labelling) is read on its own, with only the columns it needs.

A separate file so that the tasks of the other scripts keep their cache key (Nextflow puts
the content of the bin/ scripts a task calls into its hash).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from clade_splash import bh, decode_anchor, entropy_bits, parse_setting

# Written for every tested anchor; the long per-anchor strings (class names and weights)
# are kept only for the significant ones.
ALL_COLUMNS = [
    "setting",
    "gap",
    "labelling",
    "anchor",
    "n_species",
    "table_total",
    "n_clades",
    "n_targets",
    "p_oasis",
    "effect_size",
    "p_comp",
]


def cmd_collect(a: argparse.Namespace) -> None:
    lf = pl.scan_parquet(a.inputs)
    groups = (
        lf.select("setting", "gap", "labelling")
        .unique()
        .collect()
        .sort("setting", "gap", "labelling")
    )
    all_dir = Path(a.all_out)
    all_dir.mkdir(parents=True, exist_ok=True)
    sig_parts, counts = [], []
    for setting, gap, labelling in groups.iter_rows():
        g = lf.filter(
            (pl.col("setting") == setting)
            & (pl.col("gap") == gap)
            & (pl.col("labelling") == labelling)
        )
        sub = g.select(ALL_COLUMNS).collect()
        sub = sub.with_columns(pl.Series("q_oasis", bh(sub["p_oasis"].to_numpy())))
        name = f"{setting.replace(':', '_')}_gap{gap}_{labelling}.parquet"
        sub.write_parquet(all_dir / name)
        keep = sub.filter(pl.col("q_oasis") < a.alpha).select("anchor", "q_oasis")
        counts.append(
            {
                "setting": setting,
                "gap": gap,
                "labelling": labelling,
                "n_anchors_tested": sub.height,
                "n_significant": keep.height,
            }
        )
        if keep.height:
            full = g.join(keep.lazy(), on="anchor", how="inner").collect()
            # Composition scan: BH over the significant anchors of this (setting, gap, labelling).
            full = full.with_columns(pl.Series("q_comp", bh(full["p_comp"].to_numpy())))
            sig_parts.append(full)
        del sub
    sig = pl.concat(sig_parts) if sig_parts else pl.DataFrame()
    if sig.height:
        letters, bits = [], []
        for anchor, setting in sig.select("anchor", "setting").iter_rows():
            alphabet, k = parse_setting(setting)
            letters.append(decode_anchor(int(anchor), alphabet, k))
            bits.append(entropy_bits(int(anchor), alphabet, k))
        sig = sig.with_columns(
            pl.Series("anchor_letters", letters, dtype=pl.String),
            pl.Series("anchor_entropy_bits", np.array(bits, dtype=float)),
        )
    sig.write_parquet(a.sig_out)
    c = pl.DataFrame(counts)
    c.write_csv(a.counts_out, separator="\t")
    print(c)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--all-out",
        required=True,
        help="folder: one parquet per (setting, gap, labelling)",
    )
    p.add_argument("--sig-out", required=True)
    p.add_argument("--counts-out", required=True)
    a = p.parse_args(argv)
    cmd_collect(a)


if __name__ == "__main__":
    main()
