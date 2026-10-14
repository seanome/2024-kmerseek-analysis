"""Draw N1 from the N1 pool so its identity distribution matches T2's, same count.

Usage:
    python3 -I select_n1.py DATA_DIR [--seed 293]

Identity = fident of the single best MMseqs2 alignment (method C1, copy -1) from
mmseqs_t2.parquet and mmseqs_n1_pool.parquet. Pairs are binned in 10-point identity bins
(0-10%, ..., 90-100%); pairs with no alignment at E <= 1e-3 form their own bin. From each
bin N1 draws as many pool pairs as T2 has there, at random with a fixed seed. Shortfalls
are written to n1_bin_counts.tsv.
"""

import argparse
from pathlib import Path

import polars as pl


def identity_bins(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .filter(pl.col("copy") == -1)
        .filter(pl.col("method").is_null() | (pl.col("method") == "C1"))
        .select("accession_a", "accession_b", "fident")
        .unique(
            subset=["accession_a", "accession_b"], keep="first", maintain_order=True
        )
        .with_columns(
            pl.when(pl.col("fident").is_null())
            .then(pl.lit("no alignment"))
            .otherwise(
                (pl.col("fident").clip(0, 0.9999) * 10)
                .floor()
                .cast(pl.Int64)
                .cast(pl.Utf8)
            )
            .alias("identity_bin")
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--seed", type=int, default=293)
    args = p.parse_args()
    data = Path(args.data_dir)
    t2 = identity_bins(data / "mmseqs_t2.parquet")
    pool = identity_bins(data / "mmseqs_n1_pool.parquet").join(
        pl.read_parquet(data / "n1_pool.parquet").select(
            "accession_a", "accession_b", "family_multiset"
        ),
        on=["accession_a", "accession_b"],
    )
    need = t2.group_by("identity_bin").len().sort("identity_bin")
    picks, rows = [], []
    for b, n in need.iter_rows():
        cands = pool.filter(pl.col("identity_bin") == b).sort(
            "accession_a", "accession_b"
        )
        got = cands.sample(n=min(n, cands.height), seed=args.seed, shuffle=True)
        picks.append(got)
        rows.append((b, n, got.height))
    n1 = pl.concat(picks).with_columns(pl.lit("N1").alias("pair_set"))
    n1.write_parquet(data / "n1_pairs.parquet")
    pl.DataFrame(
        rows, schema=["identity_bin", "n_needed", "n_found"], orient="row"
    ).write_csv(data / "n1_bin_counts.tsv", separator="\t")
    print(f"N1 pairs={n1.height}; bins (bin, needed, found): {rows}")


if __name__ == "__main__":
    main()
