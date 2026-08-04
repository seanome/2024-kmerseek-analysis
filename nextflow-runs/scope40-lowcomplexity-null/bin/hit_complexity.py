#!/usr/bin/env python3
"""
Turn one kmerseek search result into per-hit and per-k-mer complexity tables.

kmerseek emits one row per matched *region*, with `moltype_seq` holding the HP
encoding of that region and `region_length` its length. When a region is
contiguous — region_length == ksize + n_intersecting_hashes - 1 — every length-k
window of `moltype_seq` is a genuine matched k-mer, so the region expands
losslessly into its constituent k-mers. Non-contiguous regions (~0.5% of hits at
k=26) span unmatched positions too, so which windows actually matched is
ambiguous; they are dropped and counted in the sidecar stats rather than
silently contaminating the complexity distribution.

Two outputs:

  *.hits.parquet   One row per hit, carrying `m_hist` — the histogram of
                   minority counts over that hit's matched k-mers, indexed
                   m = 0 .. k//2. This is the pivot for the whole analysis: a
                   hit survives a mask at floor m* combined with a shared-k-mer
                   floor c (notebook 085's filter) iff sum(m_hist[m*:]) >= c.
                   Storing the full histogram keeps both knobs open rather than
                   baking in one threshold.

  *.kmers.parquet  Per minority count m, the number of matched-k-mer
                   occurrences and the number of distinct k-mers — the
                   k-mer-level complexity distribution.

Usage:
    hit_complexity.py --results r.csv --seqset shuf01 --alphabet hp-thomas-dill \
        --ksize 26 --scop-domains scope_domains.tsv --out-prefix out
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import polars as pl

from hp_alphabets import encode, label_for

# Columns the analysis needs; anything else in the CSV is dropped up front so
# that the large real-run results never get fully materialised.
KEEP = [
    "query_name",
    "target_name",
    "containment",
    "n_intersecting_hashes",
    "poisson_pvalue",
    "enrichment",
    "query_subseq",
    "moltype_seq",
    "region_length",
]


def verify_encoding(df: pl.DataFrame, cli_flag: str, n_check: int = 500) -> int:
    """Re-encode kmerseek's own `query_subseq` and assert it reproduces
    `moltype_seq`. Guards against the Python HP tables drifting from the Rust
    ones — if they ever diverge, every complexity number here is wrong."""
    sub = df.head(n_check)
    mismatches = 0
    for row in sub.iter_rows(named=True):
        if encode(row["query_subseq"], cli_flag) != row["moltype_seq"]:
            mismatches += 1
    return mismatches


def window_minority_counts(seqs: list[str], ksize: int) -> np.ndarray:
    """Minority count of every length-k window, for equal-length HP strings.

    Returns an (n_seqs, n_windows) int array. Uses a cumulative H count so the
    whole batch is vectorised rather than looped per window.
    """
    n, length = len(seqs), len(seqs[0])
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(n, length)
    is_h = (arr == ord("H")).astype(np.int32)
    cum = np.zeros((n, length + 1), dtype=np.int32)
    np.cumsum(is_h, axis=1, out=cum[:, 1:])
    starts = np.arange(length - ksize + 1)
    n_h = cum[:, starts + ksize] - cum[:, starts]
    return np.minimum(n_h, ksize - n_h)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--seqset", required=True, help="real, or shuf01..shuf10")
    parser.add_argument("--alphabet", required=True, help="kmerseek CLI flag")
    parser.add_argument("--ksize", type=int, required=True)
    parser.add_argument("--scop-domains", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    k = args.ksize
    n_bins = k // 2 + 1
    stats: dict[str, object] = {
        "seqset": args.seqset,
        "alphabet": label_for(args.alphabet),
        "ksize": k,
    }

    hits_path = f"{args.out_prefix}.hits.parquet"
    kmers_path = f"{args.out_prefix}.kmers.parquet"
    stats_path = f"{args.out_prefix}.stats.tsv"

    def write_empty(reason: str) -> int:
        stats["status"] = reason
        for path, schema in (
            (hits_path, {"query_domain": pl.Utf8}),
            (kmers_path, {"minority_count": pl.Int64}),
        ):
            pl.DataFrame(schema=schema).write_parquet(path)
        pl.DataFrame([stats]).write_csv(stats_path, separator="\t")
        print(f"{args.seqset} {args.alphabet} k={k}: {reason}", file=sys.stderr)
        return 0

    if os.path.getsize(args.results) == 0:
        return write_empty("empty_results_file")

    lazy = pl.scan_csv(args.results, infer_schema_length=10_000)
    available = lazy.collect_schema().names()
    missing = [c for c in KEEP if c not in available]
    if missing:
        print(f"FATAL: results missing columns {missing}", file=sys.stderr)
        return 1

    df = (
        lazy.select(KEEP)
        .with_columns(
            [
                pl.col("query_name").str.split(" ").list.get(0).alias("query_domain"),
                pl.col("target_name").str.split(" ").list.get(0).alias("target_domain"),
            ]
        )
        # Self-hits are trivially perfect and would swamp the null.
        .filter(pl.col("query_domain") != pl.col("target_domain"))
        .drop(["query_name", "target_name"])
        .collect()
    )

    stats["n_hits_raw"] = len(df)
    if df.is_empty():
        return write_empty("no_non_self_hits")

    mismatches = verify_encoding(df, args.alphabet)
    stats["encoding_check_mismatches"] = mismatches
    if mismatches:
        print(
            f"FATAL: Python HP table for {args.alphabet} disagrees with kmerseek's "
            f"moltype_seq in {mismatches} of the first rows checked",
            file=sys.stderr,
        )
        return 1

    df = df.with_columns(
        (pl.col("region_length") == k + pl.col("n_intersecting_hashes") - 1).alias("contiguous")
    )
    n_noncontig = int((~df["contiguous"]).sum())
    stats["n_hits_noncontiguous_dropped"] = n_noncontig
    df = df.filter(pl.col("contiguous")).drop(["contiguous", "query_subseq"])
    stats["n_hits"] = len(df)
    if df.is_empty():
        return write_empty("no_contiguous_hits")

    # Expand each region into its matched k-mers, batched by region length so
    # the window arithmetic stays vectorised.
    hist = np.zeros((len(df), n_bins), dtype=np.int32)
    kmer_occurrences = np.zeros(n_bins, dtype=np.int64)
    distinct_by_m: list[set[str]] = [set() for _ in range(n_bins)]

    df = df.with_row_index("row_idx")
    for (length,), group in df.group_by(["region_length"], maintain_order=True):
        seqs = group["moltype_seq"].to_list()
        rows = group["row_idx"].to_numpy()
        counts = window_minority_counts(seqs, k)
        np.add.at(hist, (np.repeat(rows, counts.shape[1]), counts.ravel()), 1)
        kmer_occurrences += np.bincount(counts.ravel(), minlength=n_bins)
        for seq, row_counts in zip(seqs, counts):
            for start, m in enumerate(row_counts):
                distinct_by_m[m].add(seq[start : start + k])

    hist_lists = hist.tolist()
    m_index = np.arange(n_bins)
    m_max = np.array([int(m_index[np.nonzero(h)[0][-1]]) for h in hist])
    m_min = np.array([int(m_index[np.nonzero(h)[0][0]]) for h in hist])

    hits = (
        df.drop(["row_idx", "moltype_seq"])
        .with_columns(
            [
                pl.Series("m_hist", hist_lists, dtype=pl.List(pl.Int32)),
                pl.Series("m_max", m_max, dtype=pl.Int32),
                pl.Series("m_min", m_min, dtype=pl.Int32),
                pl.lit(args.seqset).alias("seqset"),
                pl.lit(label_for(args.alphabet)).alias("alphabet"),
                pl.lit(k, dtype=pl.Int32).alias("ksize"),
            ]
        )
    )

    # SCOP labels are joined for every seqset, not just `real`: on a shuffled
    # seqset they are the null's validity check (a correct shuffle should show
    # no fold-level enrichment among its hits).
    scop = pl.read_csv(args.scop_domains, separator="\t")
    scop_q = scop.select(["domain_id", "scop_fold", "scop_superfamily", "scop_family"]).rename(
        {
            "domain_id": "query_domain",
            "scop_fold": "q_fold",
            "scop_superfamily": "q_superfamily",
            "scop_family": "q_family",
        }
    )
    scop_t = scop.select(["domain_id", "scop_fold", "scop_superfamily", "scop_family"]).rename(
        {
            "domain_id": "target_domain",
            "scop_fold": "t_fold",
            "scop_superfamily": "t_superfamily",
            "scop_family": "t_family",
        }
    )
    hits = (
        hits.join(scop_q, on="query_domain", how="left")
        .join(scop_t, on="target_domain", how="left")
        .with_columns(
            [
                (pl.col("q_fold") == pl.col("t_fold")).alias("same_fold"),
                (pl.col("q_superfamily") == pl.col("t_superfamily")).alias("same_superfamily"),
                (pl.col("q_family") == pl.col("t_family")).alias("same_family"),
            ]
        )
        .drop(["q_fold", "t_fold", "q_superfamily", "t_superfamily", "q_family", "t_family"])
    )
    hits.write_parquet(hits_path)

    kmers = pl.DataFrame(
        {
            "minority_count": m_index,
            "n_kmer_occurrences": kmer_occurrences,
            "n_distinct_kmers": [len(s) for s in distinct_by_m],
        }
    ).with_columns(
        [
            pl.lit(args.seqset).alias("seqset"),
            pl.lit(label_for(args.alphabet)).alias("alphabet"),
            pl.lit(k, dtype=pl.Int32).alias("ksize"),
        ]
    )
    kmers.write_parquet(kmers_path)

    stats["status"] = "ok"
    stats["n_matched_kmer_occurrences"] = int(kmer_occurrences.sum())
    stats["n_same_superfamily"] = int(hits["same_superfamily"].fill_null(False).sum())
    pl.DataFrame([stats]).write_csv(stats_path, separator="\t")
    print(
        f"{args.seqset} {args.alphabet} k={k}: {len(df):,} hits, "
        f"{int(kmer_occurrences.sum()):,} matched k-mers, "
        f"{n_noncontig:,} non-contiguous dropped",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
