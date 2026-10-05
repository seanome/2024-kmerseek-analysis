"""Join the per-job outputs of `delimitation_pre_transfer.py --results` into two tables.

Reads every `<arm>.<species>.domains.parquet` and `<arm>.<species>.hits.parquet` in --in-dir
and writes --out (one row per arm, species and Pfam instance) and the same path with
`.hits.parquet` (one row per region that overlaps an instance). Each row gets `species`,
`tool`, and for kmerseek `alphabet`, `ksize` and `lc`, read off the file name; these are
null for the comparison tools. The hits table is written by streaming, so it never has to
fit in memory.

Usage (on Sherlock, inside a job):
    python scripts/concat_delimitation_pre_transfer.py --in-dir $SCRATCH/238-delimitation/per-job \
        --out $SCRATCH/238-delimitation/238_delimitation_pre_transfer_full.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

KMERSEEK_ARM = re.compile(r"kmerseek\.(.+)_k(\d+)_lc(True|False)$")


def labels(stem: str) -> dict:
    """`kmerseek.hp_pbotc_1st_ed2_k19_lcTrue.mouse` -> species, tool, alphabet, ksize, lc."""
    arm, species = stem.rsplit(".", 1)
    m = KMERSEEK_ARM.match(arm)
    if m:
        return dict(
            species=species,
            tool="kmerseek",
            alphabet=m.group(1),
            ksize=int(m.group(2)),
            lc=m.group(3) == "True",
        )
    return dict(species=species, tool=arm, alphabet=None, ksize=None, lc=None)


def with_labels(lf: pl.LazyFrame, stem: str) -> pl.LazyFrame:
    lab = labels(stem)
    return lf.with_columns(
        pl.lit(lab["species"]).alias("species"),
        pl.lit(lab["tool"]).alias("tool"),
        pl.lit(lab["alphabet"], dtype=pl.String).alias("alphabet"),
        pl.lit(lab["ksize"], dtype=pl.Int64).alias("ksize"),
        pl.lit(lab["lc"], dtype=pl.Boolean).alias("lc"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stems = sorted(
        f.name.removesuffix(".domains.parquet")
        for f in args.in_dir.glob("*.domains.parquet")
    )
    missing = [s for s in stems if not (args.in_dir / f"{s}.hits.parquet").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} jobs have a domains file and no hits file: {missing[:5]}"
        )
    print(f"{len(stems)} jobs in {args.in_dir}", flush=True)

    dom = pl.concat(
        [
            with_labels(pl.scan_parquet(args.in_dir / f"{s}.domains.parquet"), s)
            for s in stems
        ]
    ).collect()
    dom.write_parquet(args.out, compression="zstd")
    print(
        f"{dom.height:_} rows, {dom['arm'].n_unique()} arms, {dom['species'].n_unique()} species -> {args.out}",
        flush=True,
    )

    hits_out = Path(str(args.out).replace(".parquet", ".hits.parquet"))
    pl.concat(
        [
            with_labels(pl.scan_parquet(args.in_dir / f"{s}.hits.parquet"), s)
            for s in stems
        ]
    ).sink_parquet(hits_out, compression="zstd")
    n = pl.scan_parquet(hits_out).select(pl.len()).collect().item()
    print(
        f"{n:_} rows -> {hits_out} ({hits_out.stat().st_size / 1e9:.2f} GB)", flush=True
    )


if __name__ == "__main__":
    main()
