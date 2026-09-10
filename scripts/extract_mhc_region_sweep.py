#!/usr/bin/env python3
"""Pull kmerseek matched regions for the curated MHC genes across the full k sweep.

`extract_mhc.py` product D keeps seven hand-picked alphabet/ksize combos over all 221
MHC-window genes, which is the right trade for the domain-call notebooks. The
localisation figures need the opposite trade: every k of a few alphabets, so coverage can
be plotted against k, but only the 24 curated MHC genes, so the row count stays small.

Query- and target-side interval columns are both kept. The target side is what makes the
notebook-212-style paired histogram possible, and it is the half `extract_mhc.py` has no
use for.
"""
import argparse, glob, os, re, sys
from concurrent.futures import ProcessPoolExecutor

import polars as pl

KEEP = ["query_name", "target_name", "containment", "max_containment",
        "n_intersecting_hashes", "region_start", "region_end",
        "target_start", "target_end", "region_length", "region_poisson_score"]


def acc(col):
    """sp|ACC|NAME -> ACC. Extract-then-coalesce: polars evaluates both arms of a
    when/then over every row, so a split-based version throws on rows with no pipe."""
    return pl.coalesce(col.str.extract(r"^[^|]*\|([^|]+)\|", 1), col)


def one(args):
    path, keep_acc = args
    b = os.path.basename(path)
    m = re.match(r"human_vs_(\w+?)\.([\w]+)\.k(\d+)\.lc(true|false)\.regions\.parquet", b)
    if not m:
        return None
    sp, alpha, k, lc = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    try:
        df = (pl.scan_parquet(path)
                .with_columns(query_acc=acc(pl.col("query_name")),
                              target_acc=acc(pl.col("target_name")))
                .filter(pl.col("query_acc").is_in(keep_acc))
                .select(KEEP + ["query_acc", "target_acc"])
                .collect())
    except Exception as e:
        print(f"  SKIP {b}: {type(e).__name__} {e}", file=sys.stderr)
        return None
    if df.is_empty():
        return None
    return df.with_columns(species=pl.lit(sp), alphabet=pl.lit(alpha),
                           ksize=pl.lit(k, dtype=pl.Int64), lc=pl.lit(lc == "true"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mhc-core", required=True)
    p.add_argument("--alphabets", default="protein20,dayhoff6,hp_kyte_doolittle2,"
                   "hp_lehninger2,hp_lehninger_c_nonpolar2,hp_lehninger_hpc3,"
                   "hp_pbotc_1st_ed2,hp_thomas_dill2,hp_thomas_dill_no_c2")
    p.add_argument("--lc", default="true", choices=["true", "false", "both"])
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()

    keep = set(open(a.mhc_core).read().split())
    alphabets = set(a.alphabets.split(","))
    files = []
    for f in sorted(glob.glob(f"{a.results}/kmerseek/human_vs_*.regions.parquet")):
        m = re.match(r"human_vs_(\w+?)\.([\w]+)\.k(\d+)\.lc(true|false)\.regions\.parquet",
                     os.path.basename(f))
        if not m or m.group(2) not in alphabets:
            continue
        if a.lc != "both" and m.group(4) != a.lc:
            continue
        files.append(f)
    print(f"core genes: {len(keep)}   alphabets: {len(alphabets)}   files: {len(files)}",
          flush=True)

    with ProcessPoolExecutor(a.threads) as ex:
        frames = [d for d in ex.map(one, [(f, keep) for f in files], chunksize=4)
                  if d is not None]
    if not frames:
        print("EMPTY", flush=True)
        return
    out = pl.concat(frames, how="diagonal_relaxed")
    out.write_parquet(a.out, compression="zstd")
    print(f"{out.height:_} rows -> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)", flush=True)
    print(out.group_by("alphabet").agg(pl.col("ksize").min().alias("k_min"),
                                       pl.col("ksize").max().alias("k_max"),
                                       pl.len().alias("rows")).sort("alphabet"))


if __name__ == "__main__":
    main()
