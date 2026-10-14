"""Build N2: random Swiss-Prot pairs matched to T1 and T2 by the number of shared k-mers.

Usage:
    python3 -I build_n2_matched_random.py DATA_DIR SWISSPROT_FASTA_GZ OFFSET_PARQUET...
        [--sample-proteins 20_000] [--seed 293]

The matching variable is n_shared_kmer_hits at protein20, k = 5 (shared 5-mer occurrence
pairs after dropping 5-mers seen more than 5 times in either protein), read from the
offset-test results of T1 and T2 (copy = -1). Pairs are binned by
floor(log2(n_shared_kmer_hits + 1)); N2 draws, from each bin, as many random pairs as T1
and T2 have together in that bin.

Random pairs: a random sample of --sample-proteins Swiss-Prot proteins; every pair of
them sharing at least one 5-mer is found with an inverted index (5-mers found in more than
50 sampled proteins are skipped while searching, to keep the search small). Pairs sharing
no 5-mer fill bin 0. The exact n_shared_kmer_hits of every drawn pair is recomputed with
kmer_offset_utils.shared_kmer_hits, and the pair is placed in the bin of that number.
If a bin has fewer random pairs than needed, all of them are used and the shortfall is
written to n2_bin_counts.tsv.
"""

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import kmer_offset_utils as ku  # noqa: E402

MATCH_ALPHABET, MATCH_K = "protein20", 5
MAX_PROTEINS_PER_KMER = 50


def log2_bin(n: int) -> int:
    return int(math.floor(math.log2(n + 1)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("swissprot")
    p.add_argument("offset_parquets", nargs="+")
    p.add_argument("--sample-proteins", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=293)
    args = p.parse_args()
    data = Path(args.data_dir)
    rng = random.Random(args.seed)

    target = (
        pl.concat(
            [pl.read_parquet(f) for f in args.offset_parquets], how="diagonal_relaxed"
        )
        .filter(copy=-1, alphabet=MATCH_ALPHABET, ksize=MATCH_K, n_block=ku.N_BLOCKS[0])
        .select("pair_set", "n_shared_kmer_hits")
        .with_columns(
            pl.col("n_shared_kmer_hits").map_elements(log2_bin, pl.Int64).alias("bin")
        )
    )
    need = Counter(target["bin"].to_list())

    seqs = ku.read_fasta(args.swissprot)
    accs = sorted(seqs)
    sample = rng.sample(accs, args.sample_proteins)
    index: dict[str, list[int]] = defaultdict(list)
    for i, acc in enumerate(sample):
        for kmer in ku.kmer_positions(ku.encode(seqs[acc], MATCH_ALPHABET), MATCH_K):
            index[kmer].append(i)
    # Every pair of sampled proteins sharing a 5-mer, as one int64 code x * n + y (x < y),
    # counted with numpy; a dict of tuples needed 12 GB for 40_000 proteins.
    n = len(sample)
    chunks = []
    for members in index.values():
        if 2 <= len(members) <= MAX_PROTEINS_PER_KMER:
            m = np.array(members, dtype=np.int64)
            x, y = np.triu_indices(len(m), 1)
            chunks.append(m[x] * n + m[y])
    del index
    codes, n_shared = np.unique(np.concatenate(chunks), return_counts=True)
    shared_codes = set(codes.tolist())
    by_bin: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for code, c in zip(codes.tolist(), n_shared.tolist()):
        by_bin[log2_bin(c)].append((code // n, code % n))
    rows = []
    shortfall = []
    used = set()
    for b in sorted(need):
        got = 0
        if b == 0:
            while got < need[b]:
                x, y = rng.sample(range(len(sample)), 2)
                if min(x, y) * n + max(x, y) in shared_codes or (x, y) in used:
                    continue
                used.add((x, y))
                rows.append((sample[x], sample[y], 0))
                got += 1
        else:
            # Candidates whose approximate bin is b or b+1 (dropping 5-mers seen more than
            # 5 times can only lower the exact count).
            cands = by_bin.get(b, []) + by_bin.get(b + 1, [])
            rng.shuffle(cands)
            for x, y in cands:
                if got >= need[b]:
                    break
                hits, _ = ku.shared_kmer_hits(
                    seqs[sample[x]], seqs[sample[y]], MATCH_ALPHABET, MATCH_K
                )
                if log2_bin(len(hits)) != b or (x, y) in used:
                    continue
                used.add((x, y))
                rows.append((sample[x], sample[y], len(hits)))
                got += 1
        shortfall.append((b, need[b], got))
    n2 = pl.DataFrame(
        rows,
        schema=["accession_a", "accession_b", "n_shared_kmer_hits_match"],
        orient="row",
    ).with_columns(pl.lit("N2").alias("pair_set"))
    n2.write_parquet(data / "n2_pairs.parquet")
    pl.DataFrame(
        shortfall, schema=["log2_bin", "n_needed", "n_found"], orient="row"
    ).write_csv(data / "n2_bin_counts.tsv", separator="\t")
    print(f"N2 pairs={n2.height}; bins (bin, needed, found): {shortfall}")


if __name__ == "__main__":
    main()
