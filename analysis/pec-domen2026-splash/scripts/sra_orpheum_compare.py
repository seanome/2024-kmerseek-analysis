#!/usr/bin/env python3
"""Compare orpheum 1.0.4's frames with pec-tools best-frame (protein k7, orpheum rule) on the
same sample of frame-test reads with known frames.

orpheum writes one peptide per frame that passes its cutoff (share of 7-mers in Swiss-Prot >= 0.5),
with the frame (1, 2, 3, -1, -2, -3) and the share in the header. Frame -1 is offset 0 on the
reverse complement, which is pec-tools frame 3.

Usage: sra_orpheum_compare.py <orpheum_coding.faa> <sample.fastq> <scores.protein.k7.tsv> <truth_genes.tsv> <real_blastx.tsv> <out.tsv>
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sra_frame_test_score import truth_real  # noqa: E402

faa, fq, scores_p, genes_p, blastx, out = sys.argv[1:]
rows = []
for line in open(faa):
    if line.startswith(">"):
        rid, fr, jac = line[1:].strip().split("__")
        f = int(fr.split(":")[1])
        rows.append(
            {
                "read_id": rid,
                "orpheum_frame": f - 1 if f > 0 else 2 - f,
                "orpheum_share": float(jac.split(":")[1]),
            }
        )
orp = pl.DataFrame(rows)
ids = [line[1:].split()[0] for i, line in enumerate(open(fq)) if i % 4 == 0]
sample = pl.DataFrame({"read_id": ids})
k = 7
sc = pl.read_csv(scores_p, separator="\t").join(sample, on="read_id")
sc = sc.with_columns(
    [
        ((pl.col("read_len") - (i % 3)) // 3 - k + 1).clip(0).alias(f"w{i}")
        for i in range(6)
    ]
)
best = pl.col("best_orpheum")
pick = lambda p: pl.coalesce(
    [pl.when(best == i).then(pl.col(f"{p}{i}")) for i in range(6)]
)  # noqa: E731
sc = sc.with_columns(share=(pick("m") / pick("w")).fill_null(0.0))
truth = truth_real(blastx, pl.read_csv(genes_p, separator="\t"))
sim = sc.filter(pl.col("read_id").str.starts_with("sim|")).with_columns(
    true_frame=pl.col("read_id").str.split("|").list.get(4).cast(pl.Int64),
    source=pl.lit("simulated from CDS"),
)
real = (
    sc.filter(~pl.col("read_id").str.starts_with("sim|"))
    .join(truth.select("read_id", "true_frame", "source"), on="read_id", how="left")
    .with_columns(pl.col("source").fill_null("real: no protein hit"))
)
d = pl.concat([sim, real], how="diagonal_relaxed").select(
    "read_id", "source", "true_frame", "best_orpheum", "share"
)
# orpheum may report several frames for one read; count the read right if any reported frame is right
o = orp.group_by("read_id").agg(
    pl.col("orpheum_frame"), pl.len().alias("orpheum_frames")
)
d = (
    d.join(o, on="read_id", how="left")
    .with_columns(
        orpheum_any=pl.col("orpheum_frames").is_not_null(),
        orpheum_right=pl.col("orpheum_frame")
        .list.contains(pl.col("true_frame"))
        .fill_null(False),
        ours_kept=pl.col("share") >= 0.5,
    )
    .with_columns(
        ours_right=pl.col("ours_kept")
        & (pl.col("best_orpheum") == pl.col("true_frame"))
    )
)
summ = (
    d.group_by("source")
    .agg(
        reads=pl.len(),
        orpheum_reads_kept=pl.col("orpheum_any").sum(),
        orpheum_reads_more_than_one_frame=(pl.col("orpheum_frames") > 1).sum(),
        orpheum_right=pl.col("orpheum_right").sum(),
        pectools_reads_kept_ge_0_5=pl.col("ours_kept").sum(),
        pectools_right=pl.col("ours_right").sum(),
        both_kept=(pl.col("orpheum_any") & pl.col("ours_kept")).sum(),
    )
    .sort("source")
)
summ.write_csv(out, separator="\t")
print(summ)
