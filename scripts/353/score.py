#!/usr/bin/env python3
"""Score one setting's queries for notebook 353 (definitions in notebooks/353_prereg.md).

    score.py --split dev --records records.tsv --queries queries.tsv \
        --regions regions.tsv.gz --out-dir <dir>

Writes into <dir>:
  real_scores.tsv     one row per real (self-retrieval) query: for each cut-off E* in
                      CUTOFFS, found_<E*> (a region against its own record at E <= E* whose
                      target interval overlaps the window), placed_<E*> (found, and on the
                      true diagonal: target_start - region_start = offset), n_other_<E*>
                      (regions at E <= E* against records of other proteins), and the best
                      own-record region's E-value and its source (ka or run).
  shuffle_scores.tsv  one row per shuffle: n_regions_<E*> and n_other_<E*> (regions against
                      records of a different protein than the window's).
  seg_flags.tsv       (held_out) per window: share of residues SEG masks, and the flag
                      (share >= 0.15).
  examples.tsv.gz     every kept region of 300 real and 300 shuffle queries drawn with seed
                      353, with both sequences, for printing in the notebook.
  evalue_sources.tsv  how many kept rows used each region_evalue_source.
Runs inside the kmerseek image (polars).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

CUTOFFS = (0.01, 0.1, 1.0, 10.0)
SEG_FLAG_SHARE = 0.15  # notebook 344's low-complexity cut
N_EXAMPLES = 300


def tag(e: float) -> str:
    return f"{e:g}"


def read_seg(path: Path, queries: pl.DataFrame) -> pl.DataFrame:
    """segmasker -outfmt interval: '>id' lines, then 'start - end' (0-based, inclusive)."""
    masked: dict[str, int] = {}
    qid = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            qid = line[1:].split()[0]
            masked.setdefault(qid, 0)
        elif line.strip():
            a, b = (int(x) for x in line.split("-"))
            masked[qid] += b - a + 1
    seg = pl.DataFrame(
        {"query_id": list(masked), "seg_masked": list(masked.values())},
        schema={"query_id": pl.String, "seg_masked": pl.Int64},
    )
    # The window's flag comes from its unchanged query (0 within, 0 cross), whose sequence is the window.
    unchanged = queries.filter(
        (pl.col("kind") == "real")
        & (pl.col("n_within") == 0)
        & (pl.col("n_cross") == 0)
    )
    return (
        unchanged.join(seg, on="query_id", how="left")
        .with_columns(pl.col("seg_masked").fill_null(0))
        .select(
            "record_id",
            "length",
            (pl.col("seg_masked") / pl.col("length")).alias("seg_share"),
            (pl.col("seg_masked") / pl.col("length") >= SEG_FLAG_SHARE).alias(
                "low_complexity"
            ),
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    records = pl.read_csv(args.records, separator="\t", infer_schema_length=0).select(
        "record_id", "source", "base_accession"
    )
    queries = pl.read_csv(
        args.queries,
        separator="\t",
        schema_overrides={
            "sub_positions": pl.String,
            "shuffle_index": pl.String,
            "query_id": pl.String,
        },
    ).join(records, on="record_id", how="left")
    regions = pl.read_csv(
        args.regions,
        separator="\t",
        schema_overrides={
            "query_name": pl.String,
            "target_name": pl.String,
            "region_evalue_source": pl.String,
            "region_subseq": pl.String,
            "target_subseq": pl.String,
        },
        infer_schema_length=10000,
    )
    regions = (
        regions.join(
            queries.select(
                pl.col("query_id").alias("query_name"),
                "record_id",
                "base_accession",
                "offset",
                "length",
                "kind",
            ),
            on="query_name",
            how="left",
        )
        .join(
            records.select(
                pl.col("record_id").alias("target_name"),
                pl.col("base_accession").alias("target_accession"),
            ),
            on="target_name",
            how="left",
        )
        .with_columns(
            (pl.col("target_name") == pl.col("record_id")).alias("own_record"),
            (pl.col("target_accession") != pl.col("base_accession")).alias(
                "other_protein"
            ),
            (
                (pl.col("target_start") < pl.col("offset") + pl.col("length"))
                & (pl.col("target_end") > pl.col("offset"))
            ).alias("overlaps_window"),
            (
                (pl.col("target_start") - pl.col("region_start")) == pl.col("offset")
            ).alias("on_diagonal"),
        )
    )
    missing_targets = regions.filter(pl.col("target_accession").is_null()).height
    assert (
        missing_targets == 0
    ), f"{missing_targets} regions name a target not in records.tsv"

    real = queries.filter(pl.col("kind") == "real")
    real_regions = regions.filter(pl.col("kind") == "real")
    aggs = []
    for e in CUTOFFS:
        t = tag(e)
        hit = pl.col("region_evalue") <= e
        found = hit & pl.col("own_record") & pl.col("overlaps_window")
        aggs += [
            found.any().alias(f"found_{t}"),
            (found & pl.col("on_diagonal")).any().alias(f"placed_{t}"),
            (hit & pl.col("other_protein")).sum().alias(f"n_other_{t}"),
        ]
    own = (
        real_regions.filter(pl.col("own_record") & pl.col("overlaps_window"))
        .sort("region_evalue")
        .group_by("query_name", maintain_order=True)
        .first()
    )
    per = real_regions.group_by("query_name").agg(aggs)
    real_scores = (
        real.select(
            "query_id",
            "record_id",
            "source",
            "length",
            "offset",
            "n_within",
            "n_cross",
            "frac_hydrophobic",
        )
        .join(per.rename({"query_name": "query_id"}), on="query_id", how="left")
        .join(
            own.select(
                pl.col("query_name").alias("query_id"),
                pl.col("region_evalue").alias("best_own_evalue"),
                pl.col("region_evalue_source").alias("best_own_source"),
            ),
            on="query_id",
            how="left",
        )
    )
    fill = {
        c: False for c in real_scores.columns if c.startswith(("found_", "placed_"))
    } | {c: 0 for c in real_scores.columns if c.startswith("n_other_")}
    real_scores = real_scores.with_columns(
        [pl.col(c).fill_null(v) for c, v in fill.items()]
    )
    real_scores.write_csv(args.out_dir / "real_scores.tsv", separator="\t")

    shuf = queries.filter(pl.col("kind") != "real")
    shuf_regions = regions.filter(pl.col("kind") != "real")
    saggs = []
    for e in CUTOFFS:
        t = tag(e)
        hit = pl.col("region_evalue") <= e
        saggs += [
            hit.sum().alias(f"n_regions_{t}"),
            (hit & pl.col("other_protein")).sum().alias(f"n_other_{t}"),
        ]
    sper = (
        shuf_regions.group_by("query_name")
        .agg(saggs)
        .rename({"query_name": "query_id"})
    )
    shuffle_scores = shuf.select(
        "query_id",
        "record_id",
        "source",
        "kind",
        "length",
        "shuffle_index",
        "frac_hydrophobic",
    ).join(sper, on="query_id", how="left")
    shuffle_scores = shuffle_scores.with_columns(
        [
            pl.col(c).fill_null(0)
            for c in shuffle_scores.columns
            if c.startswith(("n_regions_", "n_other_"))
        ]
    )
    shuffle_scores.write_csv(args.out_dir / "shuffle_scores.tsv", separator="\t")

    seg_path = args.queries.with_name("queries.seg.txt")
    if seg_path.exists():
        read_seg(seg_path, queries).write_csv(
            args.out_dir / "seg_flags.tsv", separator="\t"
        )

    picks = pl.concat(
        [
            real.select("query_id").sample(min(N_EXAMPLES, real.height), seed=353),
            shuf.select("query_id").sample(min(N_EXAMPLES, shuf.height), seed=353),
        ]
    )
    ex = regions.join(
        picks.rename({"query_id": "query_name"}), on="query_name", how="inner"
    )
    with open(args.out_dir / "examples.tsv.gz", "wb") as fh:
        import gzip

        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as g:
            ex.write_csv(g, separator="\t")
    regions.group_by("region_evalue_source").len().sort(
        "region_evalue_source"
    ).write_csv(args.out_dir / "evalue_sources.tsv", separator="\t")
    print(
        f"real queries {real.height}, shuffles {shuf.height}, kept regions {regions.height}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
