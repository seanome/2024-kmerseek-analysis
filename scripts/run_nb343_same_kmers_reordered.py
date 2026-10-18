#!/usr/bin/env python3
"""Notebook 343 sweep: can a human protein be reordered without changing its reduced k-mers?

For a random sample of GENCODE v49 canonical human proteins, at each alphabet and k, draw
reduced strings with the protein's exact length-k word counts (de Bruijn graph walks) and,
as the baseline, block shuffles. Writes one row per protein and setting.

Runs on the laptop: pure Python, no kmerseek, about 1 GB of memory in total.

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
        scripts/run_nb343_same_kmers_reordered.py --workers 6
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import same_kmers_reordered_utils as u  # noqa: E402

FASTA = Path(
    "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
)
OUT = (
    Path(__file__).resolve().parents[1]
    / "tables"
    / "343_same_kmers_reordered_gencode_v49_human.tsv"
)

#: Pre-registered in notebook 343's first cell.
N_PROTEINS = 1_000
SAMPLE_SEED = 343
MIN_LENGTH = 50
N_DRAWS = 200
N_NEEDED = 10
MIN_CHANGED = 0.30
SETTINGS = [
    ("hp_pbotc_1st_ed2", k) for k in (8, 10, 12, 14, 16, 18, 20, 24, 28, 32)
] + [("dayhoff6", k) for k in (4, 5, 6, 7, 8, 9, 10, 12, 14, 16)]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """(header, sequence) pairs, trailing stop symbols removed."""
    records, name, seq = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(seq).rstrip("*")))
            name, seq = line[1:], []
        else:
            seq.append(line.strip())
    if name is not None:
        records.append((name, "".join(seq).rstrip("*")))
    return records


def eligible_proteins() -> list[tuple[str, str, str]]:
    """(ENSP id, gene symbol, sequence) for proteins of only the 20 standard amino acids."""
    out = []
    for header, seq in read_fasta(FASTA):
        f = header.split("|")
        if len(seq) >= MIN_LENGTH and set(seq) <= u.STANDARD_AA:
            out.append((f[0], f[6], seq))
    return out


def work(task):
    ensp, gene, seq, seed = task
    rows = []
    for alphabet, k in SETTINGS:
        row = u.evaluate_protein(
            seq,
            alphabet,
            k,
            seed + k,
            n_draws=N_DRAWS,
            n_needed=N_NEEDED,
            min_changed=MIN_CHANGED,
        )
        rows.append({"ensp": ensp, "gene": gene, **row})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=N_PROTEINS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    allp = eligible_proteins()
    picked = random.Random(SAMPLE_SEED).sample(allp, args.n)
    print(f"{len(allp)} eligible proteins; testing {len(picked)}", flush=True)
    tasks = [(e, g, s, SAMPLE_SEED * 1_000 + i) for i, (e, g, s) in enumerate(picked)]
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(
                    f"{i}/{len(tasks)} proteins, {time.time() - t0:.0f} s", flush=True
                )
    df = pl.DataFrame(rows).sort("ensp", "alphabet", "k")
    df = df.with_columns(n_eligible=pl.lit(len(allp)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out, separator="\t")
    print(f"wrote {df.height} rows to {args.out}")


if __name__ == "__main__":
    main()
