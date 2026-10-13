#!/usr/bin/env python3
"""Build the labeled benchmark the ranking metrics are scored on.

All against all over the 998 human proteins that have a Pfam domain in the
midi-plus truth set, then label every matched region against that truth.

Two things this script fixes that a naive version gets wrong:

  * An all-against-all search reports A->B and B->A as separate rows, so every
    match is present twice. The two copies are not identical: the E-value and
    the mean IDF are computed query-side, so the same physical match scores
    differently depending on which protein is the query. Only region length
    agrees. We keep one row per unordered pair, the orientation with the better
    E-value.

  * How much of the region has to sit on the domain changes the answer. Four
    rules are written out as separate columns so the choice is explicit.

Run build_labeled_benchmark.py before score_ranking_metrics.py.
"""
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

TRUTH = Path("/Users/olga/data/qfo-pfam-region-midi-plus/truth/human_domain_truth.parquet")
PROTEOME = Path("/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
                "/Eukaryota/UP000005640_9606.fasta")


def write_query_fasta(truth: pl.DataFrame, out: Path) -> int:
    """Pull the truth-set proteins out of the QfO human proteome by accession."""
    want = set(truth["accession"].to_list())
    kept, name, seq = 0, None, []
    with out.open("w") as fh:
        for line in PROTEOME.open():
            if line.startswith(">"):
                if name and name in want:
                    fh.write(f">{name}\n{''.join(seq)}\n")
                    kept += 1
                parts = line[1:].strip().split("|")
                name = parts[1] if len(parts) > 2 else line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
        if name and name in want:
            fh.write(f">{name}\n{''.join(seq)}\n")
            kept += 1
    return kept


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def label(hits: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Add one boolean column per correctness rule.

    Every rule needs the SAME Pfam family on both sides; they differ only in how
    much of the region has to sit on the domain. Pfam coordinates are 1-based
    inclusive, kmerseek region coordinates 0-based half-open.
    """
    dom = defaultdict(list)
    for r in truth.iter_rows(named=True):
        dom[r["accession"]].append((r["pfam_id"], r["domain_start"] - 1, r["domain_end"]))

    rules = ["any_overlap", "frac20_of_region", "frac20_of_domain", "iou20"]
    out = {k: np.zeros(len(hits), bool) for k in rules}
    cols = hits.select(["query_name", "target_name", "region_start", "region_end",
                        "target_start", "target_end"]).rows()
    for i, (q, t, qs, qe, ts, te) in enumerate(cols):
        sides = []
        for acc, s0, s1 in ((q, qs, qe), (t, ts, te)):
            got = {k: set() for k in rules}
            rlen = s1 - s0
            for fam, d0, d1 in dom.get(acc, []):
                o = overlap(s0, s1, d0, d1)
                if o <= 0:
                    continue
                dlen = d1 - d0
                union = rlen + dlen - o
                got["any_overlap"].add(fam)
                if o >= 0.20 * rlen:
                    got["frac20_of_region"].add(fam)
                if o >= 0.20 * dlen:
                    got["frac20_of_domain"].add(fam)
                if union > 0 and o / union >= 0.20:
                    got["iou20"].add(fam)
            sides.append(got)
        for k in rules:
            out[k][i] = bool(sides[0][k] & sides[1][k])
    for k in rules:
        hits = hits.with_columns(pl.Series(f"correct_{k}", out[k]))
    return hits


def deduplicate(hits: pl.DataFrame) -> pl.DataFrame:
    """One row per unordered pair: the orientation with the better E-value."""
    key = [
        str(tuple(sorted([(q, qs, qe), (t, ts, te)])))
        for q, t, qs, qe, ts, te in hits.select(
            ["query_name", "target_name", "region_start", "region_end",
             "target_start", "target_end"]).rows()
    ]
    return (hits.with_columns(pl.Series("pair_key", key))
                .sort(["pair_key", "region_evalue"])
                .unique(subset=["pair_key"], keep="first"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kmerseek", required=True, help="path to the kmerseek binary")
    p.add_argument("--outdir", default=".", type=Path)
    p.add_argument("--alphabet", default="hp_lehninger2")
    p.add_argument("--ksize", type=int, default=24)
    p.add_argument("--extend-mismatch-penalty", type=float, default=1.63)
    p.add_argument("--extend-xdrop", type=float, default=6.52)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    truth = pl.read_parquet(TRUTH)
    fa = args.outdir / "human_pfam_truth.fa"
    n = write_query_fasta(truth, fa)
    print(f"{len(truth):_} truth domains, {truth['accession'].n_unique():_} proteins, "
          f"{n:_} written to {fa}")

    index = args.outdir / "truth.rocksdb"
    common = ["--alphabet", args.alphabet, "--ksize", str(args.ksize),
              "--extend-mismatch-penalty", str(args.extend_mismatch_penalty),
              "--extend-xdrop", str(args.extend_xdrop)]
    subprocess.run([args.kmerseek, "index", "--input", str(fa), "--output", str(index),
                    "--scaled", "1", "--remove-low-complexity", "--ka-queries", "200",
                    "--ka-seed", "1", *common], check=True)
    raw = args.outdir / "truth_allvall.csv"
    subprocess.run([args.kmerseek, "search", "--query", str(fa), "--target", str(index),
                    "--max-query-pvalue", "1.0", "--min-region-score", "0",
                    "--output", str(raw), *common], check=True)

    hits = (pl.read_csv(raw)
              .with_columns(pl.col("query_name").str.strip_chars(),
                            pl.col("target_name").str.strip_chars())
              .filter(pl.col("query_name") != pl.col("target_name")))
    print(f"{len(hits):_} non-self region rows")
    pairs = deduplicate(hits)
    print(f"-> {len(pairs):_} independent matches after removing mirrored rows")
    pairs = label(pairs, truth)
    for c in [c for c in pairs.columns if c.startswith("correct_")]:
        print(f"  {c:<28}{int(pairs[c].sum()):>7_} correct")
    dest = args.outdir / "labeled_pairs.parquet"
    pairs.write_parquet(dest)
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
