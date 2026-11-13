#!/usr/bin/env python3
"""Per read: does the best-scoring frame pick the true frame, and what does a share cutoff add?

Usage: frame_cutoff_check.py <real_blastx.tsv> <out.tsv> <setting=scores.tsv:k> ...

Same reads and true frames as frame_shares.py. A frame's share is its k-mers found in Swiss-Prot
over its k-mer windows. Rule "stop-free": only frames with no stop codon can be chosen (orpheum's
rule, without its low-complexity filter); "any": all six frames. For each cutoff, among reads whose
best share is at or above it:
  right_alone  the true frame has the best share and no other frame ties it
  tied         the true frame ties for best with at least one wrong frame (the choice is a coin toss)
  wrong        a wrong frame has a higher share than the true frame
Also, per frame without choosing: the share of reads whose true frame passes the cutoff, and the
share with at least one wrong frame passing it.
"""
import sys

import polars as pl

blastx_p, out_p = sys.argv[1], sys.argv[2]
truth = (pl.read_csv(blastx_p, separator="\t", has_header=False,
                     new_columns=["read_id", "g", "pident", "length", "qstart", "qend", "qlen", "bits"])
         .filter(pl.col("pident") >= 95).sort(["read_id", "bits"], descending=[False, True])
         .unique("read_id", keep="first")
         .filter((pl.col("qend") - pl.col("qstart")).abs() + 1 >= 0.9 * pl.col("qlen"))
         .with_columns(tf=pl.when(pl.col("qstart") < pl.col("qend")).then((pl.col("qstart") - 1) % 3)
                       .otherwise(3 + (pl.col("qlen") - pl.col("qstart")) % 3), src=pl.lit("real"))
         .select("read_id", "tf", "src"))
rows = []
for arg in sys.argv[3:]:
    name, rest = arg.split("=")
    path, k = rest.rsplit(":", 1)
    k = int(k)
    s = pl.read_csv(path, separator="\t")
    sim = s.filter(pl.col("read_id").str.starts_with("sim|")).with_columns(
        tf=pl.col("read_id").str.split("|").list.get(4).cast(pl.Int64), src=pl.lit("simulated"))
    d = pl.concat([sim, s.join(truth, on="read_id")], how="diagonal_relaxed")
    d = d.with_columns([(pl.col(f"m{i}") / ((pl.col("read_len") - i % 3) // 3 - k + 1)).alias(f"f{i}")
                        for i in range(6)])
    for rule in ["stop-free", "any"]:
        x = d.with_columns([pl.when((pl.col(f"stops{i}") == 0) if rule == "stop-free" else pl.lit(True))
                            .then(pl.col(f"f{i}")).otherwise(-1.0).alias(f"g{i}") for i in range(6)])
        x = x.with_columns(best=pl.max_horizontal([pl.col(f"g{i}") for i in range(6)]),
                           tval=pl.coalesce([pl.when(pl.col("tf") == i).then(pl.col(f"g{i}")) for i in range(6)]))
        x = x.with_columns(nbest=pl.sum_horizontal([(pl.col(f"g{i}") == pl.col("best")).cast(pl.Int32)
                                                    for i in range(6)]))
        x = x.with_columns(right=(pl.col("tval") == pl.col("best")) & (pl.col("nbest") == 1),
                           tied=(pl.col("tval") == pl.col("best")) & (pl.col("nbest") > 1))
        for src in ["simulated", "real"]:
            xs = x.filter(pl.col("src") == src)
            for c in [0.1, 0.5, 0.9]:
                y = xs.filter(pl.col("best") >= c)
                wrong_pass = pl.sum_horizontal([((pl.col("tf") != i) & (pl.col(f"g{i}") >= c)).cast(pl.Int32)
                                                for i in range(6)]) > 0
                rows.append(dict(setting=name, rule=rule, reads=src, cutoff=c, reads_total=xs.height,
                                 reads_kept=round(y.height / xs.height, 3),
                                 right_alone=round(y["right"].mean(), 3), tied=round(y["tied"].mean(), 3),
                                 wrong=round(1 - y["right"].mean() - y["tied"].mean(), 3),
                                 true_frame_passes=round((xs["tval"] >= c).mean(), 3),
                                 a_wrong_frame_passes=round(xs.select(wrong_pass.mean()).item(), 3)))
t = pl.DataFrame(rows)
t.write_csv(out_p, separator="\t")
with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250):
    print(t.filter(pl.col("rule") == "stop-free"))
