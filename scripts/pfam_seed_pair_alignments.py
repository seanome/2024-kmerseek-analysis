"""Sample aligned sequence pairs from every Pfam-A seed alignment.

Pfam seed alignments are curated, so the aligned columns are the family's own statement of
which residues correspond. For each family up to `--per-family` pairs of seed sequences
are drawn at random, projected onto the columns where at least one of the two has a
residue, and written with both gapped strings and the identity over columns where both
have a residue.

Usage:
    python scripts/pfam_seed_pair_alignments.py --out /Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet
"""

from __future__ import annotations

import argparse
import gzip
import random
from itertools import combinations

import polars as pl

SEED = "/Users/olga/data/pfam/Pfam-A.seed.gz"


def iter_families(path: str):
    acc = ident = None
    seqs: dict[str, list[str]] = {}
    order: list[str] = []
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("//"):
                yield acc, ident, order, seqs
                acc = ident = None
                seqs, order = {}, []
            elif line.startswith("#=GF AC"):
                acc = line.split()[2].split(".")[0]
            elif line.startswith("#=GF ID"):
                ident = line.split()[2]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                name, aln = line.split()
                if name not in seqs:
                    seqs[name] = []
                    order.append(name)
                seqs[name].append(aln)


def project_pair(a: str, b: str) -> tuple[str, str, int, int]:
    """Drop columns that are gaps in both; return gapped strings, aligned length, identities."""
    qa, ta = [], []
    lali = ident = 0
    for x, y in zip(a, b):
        xg = not x.isalpha()
        yg = not y.isalpha()
        if xg and yg:
            continue
        x = x.upper() if not xg else "-"
        y = y.upper() if not yg else "-"
        qa.append(x)
        ta.append(y)
        if not xg and not yg:
            lali += 1
            ident += x == y
    return "".join(qa), "".join(ta), lali, ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    n_fam = 0
    for acc, ident, order, seqs in iter_families(SEED):
        n_fam += 1
        if len(order) < 2:
            continue
        pairs = list(combinations(order, 2))
        if len(pairs) > args.per_family:
            pairs = rng.sample(pairs, args.per_family)
        for q, t in pairs:
            qa, ta, lali, nid = project_pair("".join(seqs[q]), "".join(seqs[t]))
            if lali == 0:
                continue
            rows.append({
                "family": acc, "family_id": ident, "n_seed": len(order),
                "query": q, "target": t, "qaln": qa, "taln": ta,
                "q_len": sum(c != "-" for c in qa), "t_len": sum(c != "-" for c in ta),
                "lali": lali, "seqid_ali": nid / lali,
            })
        if n_fam % 2000 == 0:
            print(f"{n_fam} families, {len(rows)} pairs", flush=True)
    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    print(df.shape, df["family"].n_unique(), "families")
    print(df["seqid_ali"].describe())


if __name__ == "__main__":
    main()
