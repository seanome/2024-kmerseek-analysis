#!/usr/bin/env python3
"""Join per-sample counts into one table: feature, sample_01 .. sample_06.

    count_matrix.py salmon    out.tsv sample_01.quant.sf ... sample_06.quant.sf
    count_matrix.py jellyfish out.tsv sample_01.kmers.tsv ... sample_06.kmers.tsv

The sample name is the file name up to the first dot. salmon: NumReads per transcript,
every transcript kept. jellyfish: canonical 31-mer counts (`jellyfish dump -c -t`); a k-mer
absent from a sample's dump counts 0 there, and only k-mers seen at least 10 times over the
6 samples are kept (notebooks/360_data/preregistered.md, k-mer path step 1).
"""

from __future__ import annotations

import sys
from functools import reduce
from pathlib import Path

import polars as pl

MIN_TOTAL_KMER = 10


def main(mode: str, out: str, files: list[str]) -> None:
    frames = []
    for f in sorted(files):
        sample = Path(f).name.split(".")[0]
        if mode == "salmon":
            d = pl.read_csv(f, separator="\t").select(pl.col("Name").alias("feature"), pl.col("NumReads").alias(sample))
        elif mode == "jellyfish":
            d = pl.read_csv(f, separator="\t", has_header=False, new_columns=["feature", sample],
                            schema_overrides={"feature": pl.Utf8, sample: pl.Int64})
        else:
            raise SystemExit(f"unknown mode {mode}")
        frames.append(d)
    samples = [f"sample_{i:02d}" for i in range(1, 7)]
    if sorted(d.columns[1] for d in frames) != samples:
        raise SystemExit(f"expected {samples}, got {[d.columns[1] for d in frames]}")
    m = reduce(lambda a, b: a.join(b, on="feature", how="full", coalesce=True), frames).fill_null(0)
    m = m.select("feature", *samples)
    if mode == "jellyfish":
        n0 = m.height
        m = m.filter(pl.sum_horizontal(samples) >= MIN_TOTAL_KMER)
        print(f"{n0} k-mers in any dump, {m.height} with total >= {MIN_TOTAL_KMER}", file=sys.stderr)
    m.sort("feature").write_csv(out, separator="\t")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
