"""Score the read frame choice against reads whose true frame is known.

Simulated reads carry their true frame in the name (sim|gene|set|start|frame). Real reads get
theirs from a DIAMOND blastx against the new-genome proteins whose CDS passed the check: best hit
by bit score with identity >= 95 %. If the hit covers >= 90 % of the read, the read is "whole
read coding"; otherwise "partly coding" (it runs into a UTR, an intron or another gene; the frame
of its coding part is still known). Reads with no such hit have no known frame ("no protein hit").

Frame index follows pec-tools: 0-2 = offsets 0-2 on the read, 3-5 = offsets 0-2 on its reverse
complement.

Each scores file comes from `pec-tools frame-scores` and is scored under both rules: "stopfree"
(only frames without a stop codon may be chosen, orpheum's rule) and "any" (all six frames).
The setting name must end in k<k>, e.g. "protein k7".

Usage: python sra_frame_test_score.py <truth_genes.tsv> <blastx.tsv> <out_dir> <setting=scores.tsv> ...
Writes frame_test_summary.tsv (one row per setting, read source, gene set and cutoff) and
frame_test_hist.tsv (reads per bin of the chosen frame's matching share, split by correct or not).
"""

import sys
from pathlib import Path

import polars as pl

CUTOFFS = [round(0.05 * i, 2) for i in range(21)]  # floats, so the column stays Float64


def truth_real(blastx: str, genes: pl.DataFrame) -> pl.DataFrame:
    cols = [
        "read_id",
        "fun_gene",
        "pident",
        "length",
        "qstart",
        "qend",
        "qlen",
        "bitscore",
    ]
    b = pl.read_csv(blastx, separator="\t", has_header=False, new_columns=cols)
    b = (
        b.filter(pl.col("pident") >= 95)
        .sort(["read_id", "bitscore"], descending=[False, True])
        .unique("read_id", keep="first")
    )
    fwd = pl.col("qstart") < pl.col("qend")
    return (
        b.with_columns(
            true_frame=pl.when(fwd)
            .then((pl.col("qstart") - 1) % 3)
            .otherwise(3 + (pl.col("qlen") - pl.col("qstart")) % 3),
            source=pl.when(
                (pl.col("qend") - pl.col("qstart")).abs() + 1 >= 0.9 * pl.col("qlen")
            )
            .then(pl.lit("real: whole read coding"))
            .otherwise(pl.lit("real: partly coding")),
        )
        .join(genes.select("fun_gene", "set"), on="fun_gene", how="left")
        .with_columns(pl.col("set").fill_null("other genes"))
        .select("read_id", "true_frame", "source", "set")
    )


def main():
    genes_p, blastx, out = sys.argv[1:4]
    out = Path(out)
    genes = pl.read_csv(genes_p, separator="\t")
    real_truth = truth_real(blastx, genes)
    summ, hists = [], []
    for arg in sys.argv[4:]:
        name, path = arg.split("=", 1)
        k = int(name.split("k")[-1])
        s = pl.read_csv(path, separator="\t")
        # windows of each frame: frames 0-2 and 3-5 are offsets 0-2 of the read and its reverse complement
        s = s.with_columns(
            [
                ((pl.col("read_len") - (i % 3)) // 3 - k + 1).clip(0).alias(f"w{i}")
                for i in range(6)
            ]
        )
        sim = (
            s.filter(pl.col("read_id").str.starts_with("sim|"))
            .with_columns(parts=pl.col("read_id").str.split("|"))
            .with_columns(
                set=pl.col("parts").list.get(2),
                true_frame=pl.col("parts").list.get(4).cast(pl.Int64),
                source=pl.lit("simulated from CDS"),
            )
            .drop("parts")
        )
        real = (
            s.filter(~pl.col("read_id").str.starts_with("sim|"))
            .join(real_truth, on="read_id", how="left")
            .with_columns(
                source=pl.col("source").fill_null("real: no protein hit"),
                set=pl.col("set").fill_null("none"),
            )
        )
        base = pl.concat([sim, real], how="diagonal_relaxed")
        for rule in ["stopfree", "any"]:
            setting = f"{name}, {rule}"
            best = pl.col(f"best_{rule}")
            pick = lambda prefix: pl.coalesce(
                [pl.when(best == i).then(pl.col(f"{prefix}{i}")) for i in range(6)]
            )  # noqa: E731
            d = base.with_columns(
                best_frame=best,
                matches=pick("m").fill_null(0),
                windows=pick("w").fill_null(0),
            )
            d = d.with_columns(
                allowed=best >= 0,
                frac=pl.when(pl.col("windows") > 0)
                .then(pl.col("matches") / pl.col("windows"))
                .otherwise(0.0),
                any_match=pl.col("matches") > 0,
            ).with_columns(
                correct=(pl.col("best_frame") == pl.col("true_frame"))
                & pl.col("any_match")
            )
            for c in CUTOFFS:
                kept = (pl.col("frac") >= c) & pl.col("any_match")
                summ.append(
                    d.group_by("source", "set")
                    .agg(
                        reads=pl.len(),
                        allowed=pl.col("allowed").sum(),
                        any_match=pl.col("any_match").sum(),
                        correct=pl.col("correct").sum(),
                        kept=kept.sum(),
                        kept_correct=(kept & pl.col("correct")).sum(),
                    )
                    .with_columns(setting=pl.lit(setting), cutoff=pl.lit(float(c)))
                )
            hists.append(
                d.filter(pl.col("any_match"))
                .with_columns(
                    bin=((pl.col("frac") * 20).floor().clip(0, 19) / 20).round(2)
                )
                .group_by("source", "set", "correct", "bin")
                .agg(reads=pl.len())
                .with_columns(setting=pl.lit(setting))
            )
            print(
                setting,
                d.group_by("source")
                .agg(
                    pl.len(),
                    pl.col("allowed").mean(),
                    pl.col("any_match").mean(),
                    pl.col("correct").mean(),
                )
                .sort("source"),
                flush=True,
            )
    pl.concat(summ).sort("setting", "source", "set", "cutoff").write_csv(
        out / "frame_test_summary.tsv", separator="\t"
    )
    pl.concat(hists).sort("setting", "source", "set", "correct", "bin").write_csv(
        out / "frame_test_hist.tsv", separator="\t"
    )


if __name__ == "__main__":
    main()
