#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
Window-vs-window cosine search for the E3 PLM arm.

Reduces window-level similarity to the protein-pair score notebook 340's harness
expects: the score of a (human, species) pair is the best cosine similarity over all
window pairs, which is the PLM analogue of "these two proteins share a domain-sized
stretch".

Emits the harness's 4-column TSV: human_accession, species_accession, score, evalue.
There is no E-value for a cosine similarity, so column 4 is written as NaN and
notebook 340 thresholds on column 3 at matched false-positive rate like every other
arm.
"""

from __future__ import annotations

import argparse
import gzip

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=True)
    ids = np.array([str(x) for x in z["ids"]])
    accs = np.array([i.split(":")[0] for i in ids])
    return accs, z["windows"].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True, help="human .npz from esm2_window_embed.py")
    ap.add_argument("--target", required=True, help="species .npz")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    q_acc, q_vec = load(args.query)
    t_acc, t_vec = load(args.target)

    t_uniq, t_idx = np.unique(t_acc, return_inverse=True)
    q_uniq, q_idx = np.unique(q_acc, return_inverse=True)

    best = {}
    for start in range(0, q_vec.shape[0], args.chunk):
        stop = min(start + args.chunk, q_vec.shape[0])
        sims = q_vec[start:stop] @ t_vec.T  # cosine: vectors are L2-normalised
        rows = q_idx[start:stop]
        # collapse target windows to their protein by max
        per_protein = np.full((stop - start, t_uniq.size), -1.0, dtype=np.float32)
        np.maximum.at(per_protein.T, t_idx, sims.T)
        for r in range(stop - start):
            qi = rows[r]
            order = np.argpartition(per_protein[r], -args.top_k)[-args.top_k :]
            for ti in order:
                key = (qi, ti)
                v = float(per_protein[r, ti])
                if v > best.get(key, -2.0):
                    best[key] = v

    with gzip.open(args.out, "wt") as fh:
        for (qi, ti), v in best.items():
            fh.write(f"{q_uniq[qi]}\t{t_uniq[ti]}\t{v:.6f}\tnan\n")
    print(f"wrote {len(best)} pair scores to {args.out}")


if __name__ == "__main__":
    main()
