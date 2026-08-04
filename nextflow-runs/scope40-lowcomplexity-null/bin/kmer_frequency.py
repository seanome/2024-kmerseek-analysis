#!/usr/bin/env python3
"""
Reference HP k-mer frequency table, grouped by minority count.

This is the independent cross-check on the shuffle-based calibration. The
shuffle route asks "at what complexity do spurious hits stop happening?"; this
route asks the mechanistic question directly: how common is an individual
k-mer in the database, as a function of its minority count?

What makes a k-mer dangerous is its per-k-mer database frequency — a k-mer
present in a large share of sequences carries almost no evidence when shared,
and inflates the Poisson rate the p-value model assumes is small. Minority count
is a proxy for that frequency because compositions of higher complexity are
spread over exponentially more distinct k-mers (C(k, m) of them), so each
individual one is rarer. Plotting per-k-mer document frequency against minority
count should therefore show a sharp knee where the poly-dominant classes sit
orders of magnitude above the rest; m* belongs just above that knee, and should
land near the value the shuffle line picks independently.

Document frequency (fraction of *sequences* containing the k-mer at least once)
is the headline statistic rather than raw occurrence count, because that is what
kmerseek's containment and Poisson model actually see.

Usage:
    kmer_frequency.py --fasta scope40.fa --alphabet hp-thomas-dill --ksize 26 \
        --out-prefix out
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from math import comb, log2

import numpy as np
import polars as pl

from hp_alphabets import complexity_bits, encode, label_for
from shuffle_fasta import parse_fasta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--alphabet", required=True, help="kmerseek CLI flag")
    parser.add_argument("--ksize", type=int, required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="how many of the most frequent k-mers to write out for inspection",
    )
    args = parser.parse_args()

    k = args.ksize
    occurrences: defaultdict[str, int] = defaultdict(int)
    doc_freq: defaultdict[str, int] = defaultdict(int)
    n_seqs = 0
    n_kmer_slots = 0
    n_skipped_noncanonical = 0

    for _, seq in parse_fasta(args.fasta):
        n_seqs += 1
        hp = encode(seq, args.alphabet)
        seen: set[str] = set()
        for i in range(len(hp) - k + 1):
            n_kmer_slots += 1
            kmer = hp[i : i + k]
            # kmerseek skips k-mers spanning non-canonical residues; 'X' marks
            # those positions, so windows containing one are skipped here too.
            if "X" in kmer:
                n_skipped_noncanonical += 1
                continue
            occurrences[kmer] += 1
            seen.add(kmer)
        for kmer in seen:
            doc_freq[kmer] += 1

    if not occurrences:
        print("FATAL: no k-mers extracted; is ksize longer than every sequence?", file=sys.stderr)
        return 1

    kmers = list(occurrences.keys())
    n_h = np.array([kmer.count("H") for kmer in kmers], dtype=np.int32)
    minority = np.minimum(n_h, k - n_h)
    occ = np.array([occurrences[kmer] for kmer in kmers], dtype=np.int64)
    docs = np.array([doc_freq[kmer] for kmer in kmers], dtype=np.int64)
    doc_fraction = docs / n_seqs

    per_kmer = pl.DataFrame(
        {
            "kmer": kmers,
            "minority_count": minority,
            "n_occurrences": occ,
            "n_sequences": docs,
            "doc_fraction": doc_fraction,
        }
    )

    # Per-minority-count summary: the curve the knee is read off.
    summary = (
        per_kmer.group_by("minority_count")
        .agg(
            [
                pl.len().alias("n_distinct_observed"),
                pl.col("n_occurrences").sum().alias("total_occurrences"),
                pl.col("doc_fraction").mean().alias("mean_doc_fraction"),
                pl.col("doc_fraction").median().alias("median_doc_fraction"),
                pl.col("doc_fraction").quantile(0.99).alias("p99_doc_fraction"),
                pl.col("doc_fraction").max().alias("max_doc_fraction"),
            ]
        )
        .sort("minority_count")
    )

    summary = summary.with_columns(
        [
            # How many distinct k-mers *could* have this composition, versus how
            # many are actually observed — the saturation that drives the knee.
            pl.col("minority_count")
            .map_elements(lambda m: float(complexity_bits(k, int(m))), return_dtype=pl.Float64)
            .alias("complexity_bits"),
            pl.col("minority_count")
            .map_elements(lambda m: log2(comb(k, int(m))), return_dtype=pl.Float64)
            .alias("log2_n_possible"),
            pl.lit(label_for(args.alphabet)).alias("alphabet"),
            pl.lit(k, dtype=pl.Int32).alias("ksize"),
        ]
    ).with_columns(
        (pl.col("minority_count") / k).alias("minority_fraction"),
    )

    summary.write_parquet(f"{args.out_prefix}.freq_by_minority.parquet")

    top = per_kmer.sort("doc_fraction", descending=True).head(args.top_n).with_columns(
        [
            pl.lit(label_for(args.alphabet)).alias("alphabet"),
            pl.lit(k, dtype=pl.Int32).alias("ksize"),
        ]
    )
    top.write_parquet(f"{args.out_prefix}.top_kmers.parquet")

    stats = pl.DataFrame(
        [
            {
                "alphabet": label_for(args.alphabet),
                "ksize": k,
                "n_sequences": n_seqs,
                "n_kmer_slots": n_kmer_slots,
                "n_skipped_noncanonical": n_skipped_noncanonical,
                "n_distinct_kmers": len(kmers),
                "n_total_occurrences": int(occ.sum()),
            }
        ]
    )
    stats.write_csv(f"{args.out_prefix}.freq_stats.tsv", separator="\t")

    print(
        f"{args.alphabet} k={k}: {n_seqs:,} seqs, {len(kmers):,} distinct k-mers, "
        f"{int(occ.sum()):,} occurrences",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
