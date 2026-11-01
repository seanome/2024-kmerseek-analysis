#!/usr/bin/env python3
"""Notebook 275, Figure 1 option 1: how many human proteins share an exact seed with a query.

For each alphabet and k, and for each of N random human proteins used as queries, count the
other proteins of the human proteome that share at least one exact k-mer with the query, in
that alphabet. Each query is counted twice: as it is, and with its residues shuffled (same
length and composition, so every protein it shares a k-mer with is a chance hit). The
shuffled count is the x axis of the seed-length frontier in notebook 275; the y axis, the
share of Pfam seed pairs at 20-30% identity that share a seed, comes from notebook 230's
per-pair table and is computed in the notebook.

A k-mer that holds a residue outside the alphabet (X, U, *) is skipped, and no k-mer spans
two proteins.

    python scripts/run_275_seed_frontier.py

Environment:
    NB275_PROTEOME   human proteome FASTA (GENCODE v49 canonical translations, 19,732 proteins)
    NB275_OUT        output TSV (default tables/275_seed_frontier_hits_per_query.tsv)
    NB275_N_QUERIES  number of random query proteins (default 300)

On Sherlock: notebooks/275_seed_frontier.sbatch.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

PROTEOME = Path(os.environ.get(
    "NB275_PROTEOME", "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"))
OUT = Path(os.environ.get("NB275_OUT", ROOT / "tables/275_seed_frontier_hits_per_query.tsv"))
N_QUERIES = int(os.environ.get("NB275_N_QUERIES", "300"))
SEED = 0
# k values per alphabet: from where nearly every protein shares a seed with any query to
# where almost none does. The code of a k-mer, sum of class * size^position, fits in int64.
KS = {
    "hp_lehninger2": list(range(8, 45)),
    "hp_thomas_dill2": list(range(8, 45)),
    "dayhoff6": list(range(4, 21)),
    "protein20": list(range(3, 13)),
}
LUT_MAX = 1 << 27  # code spaces up to this size use a lookup table; larger ones a sorted index


def read_fasta(path: Path) -> list[tuple[str, str]]:
    out, name, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    out.append((name, "".join(buf)))
    return out


def kmer_codes(classes: np.ndarray, size: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Code of the k-mer starting at every position, and whether it is valid (no GAP inside)."""
    n = classes.size - k + 1
    if n <= 0:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    bad = np.concatenate(([0], np.cumsum(classes == hc.GAP)))
    valid = (bad[k:] - bad[:-k]) == 0
    c = np.where(classes == hc.GAP, 0, classes).astype(np.int64)
    codes = np.zeros(n, np.int64)
    for j in range(k):
        codes = codes * size + c[j:j + n]
    return codes, valid


def main() -> None:
    records = read_fasta(PROTEOME)
    names = [r[0] for r in records]
    seqs = [r[1].rstrip("*") for r in records]
    n_prot = len(seqs)
    rng = np.random.default_rng(SEED)
    q_idx = np.sort(rng.choice(n_prot, size=min(N_QUERIES, n_prot), replace=False))
    shuffled = {int(i): "".join(rng.permutation(list(seqs[i]))) for i in q_idx}
    print(f"{n_prot:,} proteins, {sum(map(len, seqs)):,} residues; {len(q_idx)} queries", flush=True)

    rows = []
    for alphabet, ks in KS.items():
        table, size = hc.TABLES[alphabet], hc.SIZES[alphabet]
        enc = lambda s: table[np.frombuffer(s.encode(), dtype=np.uint8)]
        # Proteins joined with one GAP between them, so no k-mer spans two proteins.
        parts, pid_parts = [], []
        for p, s in enumerate(seqs):
            parts += [enc(s), np.array([hc.GAP], np.uint8)]
            pid_parts.append(np.full(len(s) + 1, p, np.int32))
        db, db_pid = np.concatenate(parts), np.concatenate(pid_parts)
        for k in ks:
            t0 = time.time()
            codes, valid = kmer_codes(db, size, k)
            codes, pids = codes[valid], db_pid[:len(valid)][valid]
            space = size ** k
            if space <= LUT_MAX:
                lut = np.zeros(space, bool)
            else:
                order = np.argsort(codes, kind="stable")
                codes_sorted, pids_sorted = codes[order], pids[order]
            seen = np.zeros(n_prot, bool)
            for i in q_idx:
                for kind, qs in (("real", seqs[i]), ("shuffled", shuffled[int(i)])):
                    qc, qv = kmer_codes(enc(qs), size, k)
                    q = np.unique(qc[qv])
                    if q.size == 0:
                        hit = np.zeros(0, np.int32)
                    elif space <= LUT_MAX:
                        lut[q] = True
                        hit = pids[lut[codes]]
                        lut[q] = False
                    else:
                        lo = np.searchsorted(codes_sorted, q, side="left")
                        hi = np.searchsorted(codes_sorted, q, side="right")
                        n = hi - lo
                        if n.sum() == 0:
                            hit = np.zeros(0, np.int32)
                        else:
                            starts = np.repeat(lo - np.concatenate(([0], np.cumsum(n)[:-1])), n)
                            hit = pids_sorted[starts + np.arange(n.sum())]
                    seen[hit] = True
                    seen[i] = False  # the query itself is not a hit
                    n_hit = int(seen.sum())
                    seen[hit] = False
                    rows.append({"alphabet": alphabet, "k": k, "query": names[i].split("|")[0],
                                 "gene": names[i].split("|")[6], "query_length": len(seqs[i]),
                                 "query_kind": kind, "n_query_kmers": int(q.size),
                                 "n_proteins_sharing_seed": n_hit, "n_proteins": n_prot})
            print(f"{alphabet} k={k}: {time.time() - t0:.1f} s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT, separator="\t")
    print(f"wrote {OUT} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
