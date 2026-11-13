#!/usr/bin/env python3
"""Share of each reading frame's k-mers found in Swiss-Prot, for Botryllus reads whose true frame is known.

Usage: frame_shares.py <scores.tsv from pec-tools frame-scores> <real_blastx.tsv> <k> <out_stem>

Reads: simulated 150 nt reads cut from new-genome CDS (true frame in the name, sim|gene|set|start|frame),
and real PRJNA1219309 reads whose best DIAMOND blastx hit to a new-genome protein (identity >= 95 %)
covers >= 90 % of the read (true frame from the hit's start and strand).
Each read's six frames are grouped by how they relate to the true frame:
  true frame; same strand, 1 nt off; same strand, 2 nt off; opposite strand (three frames).
Frames 0-2 are offsets 0-2 of the read, 3-5 offsets 0-2 of its reverse complement (pec-tools order).
Share = k-mers found in Swiss-Prot / k-mer windows in that frame.
Writes <out_stem>_hist.tsv (reads per 0.02 bin of share, by read source, frame group and whether the
frame has a stop codon) and <out_stem>_summary.tsv (median share and share of frames at 0 per group).
"""
import sys

import polars as pl

scores_p, blastx_p, k, out_stem = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
s = pl.read_csv(scores_p, separator="\t")

b = (pl.read_csv(blastx_p, separator="\t", has_header=False,
                 new_columns=["read_id", "fun_gene", "pident", "length", "qstart", "qend", "qlen", "bitscore"])
     .filter(pl.col("pident") >= 95).sort(["read_id", "bitscore"], descending=[False, True])
     .unique("read_id", keep="first")
     .filter((pl.col("qend") - pl.col("qstart")).abs() + 1 >= 0.9 * pl.col("qlen"))
     .with_columns(true_frame=pl.when(pl.col("qstart") < pl.col("qend")).then((pl.col("qstart") - 1) % 3)
                   .otherwise(3 + (pl.col("qlen") - pl.col("qstart")) % 3),
                   source=pl.lit("real reads wholly in a CDS"))
     .select("read_id", "true_frame", "source"))
sim = (s.filter(pl.col("read_id").str.starts_with("sim|"))
       .with_columns(true_frame=pl.col("read_id").str.split("|").list.get(4).cast(pl.Int64),
                     source=pl.lit("simulated reads cut from CDS")))
real = s.join(b, on="read_id", how="inner")
d = pl.concat([sim, real], how="diagonal_relaxed")

long = pl.concat([
    d.select("read_id", "source", "true_frame",
             pl.lit(i).alias("frame"), pl.col(f"m{i}").alias("matches"), (pl.col(f"stops{i}") > 0).alias("has_stop"),
             ((pl.col("read_len") - (i % 3)) // 3 - k + 1).clip(0).alias("windows"))
    for i in range(6)])
same = (pl.col("frame") // 3) == (pl.col("true_frame") // 3)
shift = (pl.col("frame") - pl.col("true_frame")) % 3
long = (long.filter(pl.col("windows") > 0)
        .with_columns(share=pl.col("matches") / pl.col("windows"),
                      group=pl.when(~same).then(pl.lit("opposite strand (3 frames)"))
                      .when(shift == 0).then(pl.lit("true frame"))
                      .when(shift == 1).then(pl.lit("same strand, 1 nt off"))
                      .otherwise(pl.lit("same strand, 2 nt off"))))
hist = (long.with_columns(bin=((pl.col("share") / 0.02).floor() * 0.02).clip(0, 0.98).round(2))
        .group_by("source", "group", "has_stop", "bin").agg(pl.len().alias("frames"))
        .sort("source", "group", "has_stop", "bin"))
hist.write_csv(f"{out_stem}_hist.tsv", separator="\t")
summ = (long.group_by("source", "group").agg(
    pl.len().alias("frames"), pl.col("read_id").n_unique().alias("reads"),
    pl.col("share").median().round(3).alias("median_share"),
    pl.col("share").quantile(0.9).round(3).alias("q90_share"),
    (pl.col("share") == 0).mean().round(3).alias("share_at_0"),
    (pl.col("share") >= 0.1).mean().round(3).alias("at_least_0.1"),
    (pl.col("share") >= 0.5).mean().round(3).alias("at_least_0.5"),
    pl.col("has_stop").mean().round(3).alias("with_stop"))
    .sort("source", "group"))
summ.write_csv(f"{out_stem}_summary.tsv", separator="\t")
with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200):
    print(summ)
