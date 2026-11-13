#!/usr/bin/env python3
"""Does the second target of a SPLASH anchor follow the colony or the treatment? (notebook 361, H3)

Usage: target_share_by_pair.py <contrasts_dir> <splash_bin_dir> <target_distances.parquet> <out_dir> [max_anchors]

Every real contrast pairs each control sample with a PEC sample from the same colony (design.tsv,
column `pair`). If an anchor's two top targets are two alleles of one gene, the share of the second
target in a sample is set by the colony's genotype: it is about the same in the control and PEC
samples of one pair and differs between pairs. If the second target is a real change after PEC,
the share moves the same way in every pair.

For the kept SPLASH anchors of each contrast (at most max_anchors, a random sample, default
150,000), per-sample target counts are read from SPLASH's own dump (result_satc, with satc_dump).
With n1, n2 the counts of the two most abundant targets (pooled over samples, from
target_distances.parquet), each sample with n1 + n2 >= 10 gets share = n2 / (n1 + n2). Per anchor:
- within_pair: mean |share_PEC - share_control| over pairs with both samples counted;
- between_pairs: mean |share_a - share_b| over samples a, b from different pairs;
- same_direction: share_PEC - share_control has the same sign (none zero) in every counted pair,
  with at least 4 pairs counted. With 4 to 6 pairs and random signs this happens 3-12 % of the time.
An anchor is "follows the pair" when within_pair < between_pairs / 3.

Writes <out_dir>/target_share_by_pair.tsv (per contrast and Hamming bin) and
target_share_by_pair_anchors.parquet (per anchor, not committed).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

contrasts_dir, splash_bin, dist_pq, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 150_000
CONTRASTS = ["control", "young_2h", "young_24h"]
MIN_COUNT = 10
work = out / "satc_text"
work.mkdir(parents=True, exist_ok=True)

dist = pl.read_parquet(dist_pq).filter(pl.col("anchor_set") == "splash")
rows, per_anchor = [], []
for c in CONTRASTS:
    so = contrasts_dir / c / "splash_out"
    d = dist.filter(pl.col("contrast") == c)
    if d.height > MAX:
        d = d.sample(n=MAX, seed=361)
    anchors = work / f"{c}.anchors.txt"
    anchors.write_text("\n".join(d["anchor"].to_list()) + "\n")
    bins = sorted(so.glob("result_satc/bin*.satc"), key=lambda p: int(p.stem[3:]))
    bl = work / f"{c}.bins.txt"
    bl.write_text("\n".join(str(p) for p in bins) + "\n")
    txt = work / f"{c}.counts.txt"
    if not txt.exists():
        subprocess.run([str(splash_bin / "satc_dump"), "--anchor_list", str(anchors), "--sample_names",
                        str(so / "sample_name_to_id.mapping.txt"), "--format", "splash", "--n_bins", str(len(bins)),
                        str(bl), str(txt) + ".part"], check=True, stdout=subprocess.DEVNULL)
        Path(str(txt) + ".part").rename(txt)
    design = pl.read_csv(contrasts_dir / c / "design.tsv", separator="\t").select("sample", "condition", "pair")
    top = d.select("anchor", "hamming", "edit", pl.col("most_freq_target_1").alias("t1"), pl.col("most_freq_target_2").alias("t2"))
    cnt = (pl.scan_csv(txt, separator=" ", has_header=False, new_columns=["n", "seq", "sample"],
                       schema_overrides={"n": pl.Int64, "seq": pl.Utf8, "sample": pl.Utf8})
           .with_columns(pl.col("seq").str.slice(0, 31).alias("anchor"), pl.col("seq").str.slice(31, 31).alias("target"))
           .join(top.lazy(), on="anchor")
           .filter((pl.col("target") == pl.col("t1")) | (pl.col("target") == pl.col("t2")))
           .group_by("anchor", "sample")
           .agg(pl.col("n").filter(pl.col("target") == pl.col("t1")).sum().alias("n1"),
                pl.col("n").filter(pl.col("target") == pl.col("t2")).sum().alias("n2"))
           .collect(engine="streaming")
           .filter(pl.col("n1") + pl.col("n2") >= MIN_COUNT)
           .with_columns((pl.col("n2") / (pl.col("n1") + pl.col("n2"))).alias("share"))
           .join(design, on="sample"))
    # within-pair differences
    w = (cnt.pivot(on="condition", index=["anchor", "pair"], values="share")
         .drop_nulls(["control", "PEC"]).with_columns((pl.col("PEC") - pl.col("control")).alias("diff"))
         .group_by("anchor").agg(pl.col("diff").abs().mean().alias("within_pair"), pl.len().alias("pairs_counted"),
                                 ((pl.col("diff") > 0).all() | (pl.col("diff") < 0).all()).alias("one_sign")))
    # between-pair differences: all sample pairs from different pairs, done with numpy per anchor
    g = cnt.group_by("anchor").agg("share", "pair")
    between = []
    for a, s, p in g.iter_rows():
        s, p = np.asarray(s), np.asarray(p)
        m = p[:, None] != p[None, :]
        between.append(float(np.abs(s[:, None] - s[None, :])[m].mean()) if m.any() else np.nan)
    g = g.select("anchor").with_columns(pl.Series("between_pairs", between))
    res = (top.select("anchor", "hamming", "edit").join(w, on="anchor").join(g, on="anchor")
           .filter(pl.col("pairs_counted") >= 4)
           .with_columns((pl.col("within_pair") < pl.col("between_pairs") / 3).alias("follows_pair"),
                         pl.col("one_sign").alias("same_direction"),
                         pl.when(pl.col("hamming") == 1).then(pl.lit("1")).when(pl.col("hamming") <= 3).then(pl.lit("2-3"))
                         .when(pl.col("hamming") <= 10).then(pl.lit("4-10")).otherwise(pl.lit(">10")).alias("hamming_bin"),
                         pl.lit(c).alias("contrast")))
    per_anchor.append(res)
    print(c, "anchors sampled", d.height, "with >= 4 counted pairs", res.height, flush=True)
    for hb, sub in [("all", res)] + [(b, res.filter(pl.col("hamming_bin") == b)) for b in ["1", "2-3", "4-10", ">10"]]:
        rows.append(dict(contrast=c, hamming_bin=hb, anchors=sub.height,
                         median_within_pair=sub["within_pair"].median(), median_between_pairs=sub["between_pairs"].median(),
                         frac_follows_pair=sub["follows_pair"].mean(), frac_same_direction=sub["same_direction"].mean(),
                         # chance of one sign in all counted pairs if each pair's sign were a coin toss
                         same_direction_by_chance=(2.0 ** (1 - sub["pairs_counted"].cast(pl.Int64))).mean()))
pl.concat(per_anchor).write_parquet(out / "target_share_by_pair_anchors.parquet")
t = pl.DataFrame(rows)
t.write_csv(out / "target_share_by_pair.tsv", separator="\t")
with pl.Config(tbl_rows=50, tbl_cols=10, tbl_width_chars=200):
    print(t)
