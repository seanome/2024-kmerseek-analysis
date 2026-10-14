"""Run the k-mer offset test on a table of protein pairs, real and shuffled.

Usage:
    python3 -I run_offset_test.py PAIRS_PARQUET FASTA OUT_PARQUET [--shuffles 10] [--seed 293]
        [--workers 8]

PAIRS_PARQUET needs columns pair_set, accession_a, accession_b. FASTA holds every
accession (UniProt headers sp|ACC|NAME or tr|ACC|NAME). For every pair, every alphabet
and k in kmer_offset_utils.KSIZES and every n_block, writes one row for the real pair
(copy = -1) and one for each shuffled copy (copy = 0..shuffles-1), where B's residues are
shuffled with seed = seed * 1_000_003 + pair_index * 101 + copy.
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import kmer_offset_utils as ku  # noqa: E402

SEQS: dict[str, str] = {}
ARGS = None


def run_pair(task: tuple[int, dict]) -> list[dict]:
    idx, pair = task
    seq_a, seq_b = SEQS[pair["accession_a"]], SEQS[pair["accession_b"]]
    rows = []
    copies = [(-1, seq_b)] + [
        (c, ku.shuffled(seq_b, ARGS.seed * 1_000_003 + idx * 101 + c))
        for c in range(ARGS.shuffles)
    ]
    for copy, b in copies:
        for alphabet, k in ku.all_settings():
            for row in ku.offset_test(seq_a, b, alphabet, k):
                rows.append(
                    {
                        "pair_set": pair["pair_set"],
                        "accession_a": pair["accession_a"],
                        "accession_b": pair["accession_b"],
                        "copy": copy,
                        "length_a": len(seq_a),
                        "length_b": len(b),
                        **row,
                    }
                )
    return rows


def init(seqs, args):
    global SEQS, ARGS
    SEQS, ARGS = seqs, args


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pairs")
    p.add_argument("fasta")
    p.add_argument("out")
    p.add_argument("--shuffles", type=int, default=10)
    p.add_argument("--seed", type=int, default=293)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    pairs = pl.read_parquet(args.pairs).select("pair_set", "accession_a", "accession_b")
    seqs = ku.read_fasta(Path(args.fasta))
    missing = {
        a for a in pairs["accession_a"].to_list() + pairs["accession_b"].to_list()
    } - set(seqs)
    if missing:
        raise SystemExit(f"{len(missing)} accessions missing from {args.fasta}")
    tasks = list(enumerate(pairs.iter_rows(named=True)))
    with Pool(args.workers, initializer=init, initargs=(seqs, args)) as pool:
        rows = [r for chunk in pool.imap(run_pair, tasks, chunksize=4) for r in chunk]
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(args.out)
    print(f"pairs={pairs.height} rows={len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
