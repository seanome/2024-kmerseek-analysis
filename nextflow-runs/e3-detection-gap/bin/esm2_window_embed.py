#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
ESM-2 windowed embeddings for the E3 PLM arm.

Why windows and not whole proteins: E3's unit of analysis is the domain, so a
whole-protein mean-pooled embedding answers the wrong question.  Sliding windows of
``--window`` residues with ``--stride`` overlap keep the representation at roughly
domain scale, which is what makes the PLM arm comparable to an HP k-mer footprint.

GPU arm.  Do not run this on the laptop — hand it to a GPU box or a Batch GPU queue.

Output: a compressed .npz with
    ids      (n_windows,)  "<accession>:<start>-<end>", 1-based inclusive
    windows  (n_windows, d) float16 L2-normalised mean-pooled embeddings
"""

from __future__ import annotations

import argparse
import re

import numpy as np


HEADER = re.compile(r">(?:sp|tr)\|([A-Z0-9]+)\|")


def read_fasta(path):
    acc, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if acc is not None:
                    yield acc, "".join(seq)
                m = HEADER.match(line)
                acc = m.group(1) if m else line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if acc is not None:
        yield acc, "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import esm
    import torch

    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no GPU visible. This arm is a GPU arm; on CPU a full proteome "
            "will not finish in a useful amount of time.",
            flush=True,
        )
    model = model.to(device)
    layer = model.num_layers
    converter = alphabet.get_batch_converter()

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    batch: list[tuple[str, str]] = []

    def flush(batch):
        if not batch:
            return
        _, _, toks = converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            rep = model(toks, repr_layers=[layer])["representations"][layer]
        for i, (acc, seq) in enumerate(batch):
            # strip BOS; per-residue representations are rep[i, 1 : len(seq) + 1]
            per_res = rep[i, 1 : len(seq) + 1].float().cpu().numpy()
            for start in range(0, max(len(seq) - args.window + 1, 1), args.stride):
                end = min(start + args.window, len(seq))
                v = per_res[start:end].mean(axis=0)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                ids.append(f"{acc}:{start + 1}-{end}")
                vecs.append(v.astype(np.float16))

    for acc, seq in read_fasta(args.fasta):
        seq = seq[: args.max_len]
        if len(seq) < args.window:
            continue
        batch.append((acc, seq))
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
    flush(batch)

    np.savez_compressed(
        args.out, ids=np.array(ids), windows=np.vstack(vecs) if vecs else np.zeros((0, 1))
    )
    print(f"wrote {len(ids)} windows to {args.out}")


if __name__ == "__main__":
    main()
