#!/usr/bin/env python3
"""Rank every reference entry for every query with the methods that need no search.

Each ranked list orders all reference entries for one query, best first:

  composition_aa20         Euclidean distance between the 20 amino-acid frequencies of
                           query and target (frequencies sum to 1). Smaller is better.
  composition_<alphabet>   the same in one kmerseek alphabet's classes, one list per
                           alphabet in assets/alphabet_classes.tsv. The notebook reads the
                           one that matches the chosen kmerseek setting.
  period_hp_autocorr       cosine distance between the query's and the target's
                           autocorrelation of the two-letter HP string (hp_pbotc_1st_ed2,
                           hydrophobic +1, polar -1) at lags 1 to 30. A lag at or past a
                           sequence's length counts as 0. A target whose 30 values are all
                           0 gets distance 1.
  period_hp_moment         cosine distance between hydrophobic moment spectra: for each
                           angle from 80 to 180 degrees in 5-degree steps, |sum_n h_n
                           exp(i n angle)| / N, with h the Eisenberg consensus scale
                           (assets/eisenberg_scale.tsv, copied from Biopython).
  motif_regex              for a query whose lineage has a published repeat consensus
                           (assets/known_motifs.tsv): residues covered by matches of the
                           consensus regex, larger is better; entries with no match share
                           the last rank. The regex does not read the query, so null
                           draws get no row.

Ties share the best rank of the group (rank "min"). Output keeps, per list and query,
the top --top-n entries and every truth entry, with each one's rank, so recall at the
first false positive and the area under the precision-recall curve can be computed from
it. n_ranked is the length of the full list.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

HP_ALPHABET = "hp_pbotc_1st_ed2"
LAGS = np.arange(1, 31)
ANGLES_DEG = np.arange(80, 181, 5)
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    names, seqs, chunks = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if chunks or names:
                    seqs.append("".join(chunks))
                names.append(line[1:].split()[0])
                chunks = []
            else:
                chunks.append(line.strip())
    seqs.append("".join(chunks))
    return names, seqs


class Concat:
    """All sequences as one uint8 array of residue codes, with protein boundaries."""

    def __init__(self, seqs: list[str]):
        self.lengths = np.array([len(s) for s in seqs], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.codes = np.frombuffer("".join(seqs).encode(), dtype=np.uint8)
        self.protein = np.repeat(np.arange(len(seqs)), self.lengths)
        self.pos = np.arange(len(self.codes)) - np.repeat(self.starts, self.lengths)

    def segment_sum(
        self, values: np.ndarray, which: np.ndarray | None = None
    ) -> np.ndarray:
        idx = self.protein if which is None else self.protein[which]
        return np.bincount(idx, weights=values, minlength=len(self.lengths))


def lookup(table: dict[str, float], default: float = 0.0) -> np.ndarray:
    lut = np.full(256, default)
    for aa, v in table.items():
        lut[ord(aa)] = v
    return lut


def composition(c: Concat, classes: dict[str, int], n_classes: int) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int64)
    for aa, k in classes.items():
        lut[ord(aa)] = k
    cls = lut[c.codes]
    keep = cls >= 0  # X, B, Z, U, O and other non-standard letters are not counted
    flat = np.bincount(
        c.protein[keep] * n_classes + cls[keep], minlength=len(c.lengths) * n_classes
    )
    m = flat.reshape(len(c.lengths), n_classes).astype(float)
    tot = m.sum(axis=1, keepdims=True)
    return np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)


def hp_autocorr(c: Concat, hydrophobic: set[str]) -> np.ndarray:
    x = np.where(
        np.isin(
            c.codes,
            np.frombuffer("".join(sorted(hydrophobic)).encode(), dtype=np.uint8),
        ),
        1.0,
        -1.0,
    )
    out = np.zeros((len(c.lengths), len(LAGS)))
    for j, lag in enumerate(LAGS):
        prod = x[:-lag] * x[lag:]
        same = c.protein[:-lag] == c.protein[lag:]
        s = np.bincount(
            c.protein[:-lag][same], weights=prod[same], minlength=len(c.lengths)
        )
        n = c.lengths - lag
        out[:, j] = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    return out


def hp_moment(c: Concat, scale: dict[str, float]) -> np.ndarray:
    h = lookup(scale)[c.codes]
    out = np.zeros((len(c.lengths), len(ANGLES_DEG)))
    for j, deg in enumerate(ANGLES_DEG):
        theta = np.deg2rad(deg) * c.pos
        re_ = c.segment_sum(h * np.cos(theta))
        im_ = c.segment_sum(h * np.sin(theta))
        out[:, j] = np.hypot(re_, im_) / np.maximum(c.lengths, 1)
    return out


def euclid(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.sqrt(((t - q) ** 2).sum(axis=1))


def cosine_distance(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(q)
    tn = np.linalg.norm(t, axis=1)
    d = np.ones(len(t))
    ok = (tn > 0) & (qn > 0)
    d[ok] = 1.0 - (t[ok] @ q) / (tn[ok] * qn)
    return d


def ranks_min(score: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score; ties share the best rank of their group."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    r = np.empty(len(score), dtype=np.int64)
    first = np.r_[True, s[1:] != s[:-1]]
    group_rank = np.maximum.accumulate(np.where(first, np.arange(1, len(s) + 1), 0))
    r[order] = group_rank
    return r


def keep_rows(method, list_name, qid, score, names, truth_idx, top_n, n_total):
    r = ranks_min(score)
    keep = np.unique(
        np.concatenate([np.argsort(-score, kind="stable")[:top_n], truth_idx])
    )
    return {
        "method": [method] * len(keep),
        "list_name": [list_name] * len(keep),
        "query_id": [qid] * len(keep),
        "target": [names[i] for i in keep],
        "score": score[keep].tolist(),
        "rank": r[keep].tolist(),
        "n_ranked": [n_total] * len(keep),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--draws", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--alphabet-classes", type=Path, required=True)
    ap.add_argument("--motifs", type=Path, required=True)
    ap.add_argument(
        "--eisenberg", type=Path, required=True, help="assets/eisenberg_scale.tsv"
    )
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names, tseqs = read_fasta(args.reference)
    qnames, qseqs = read_fasta(args.queries)
    t, q = Concat(tseqs), Concat(qseqs)
    n_total = len(names)
    index = {n: i for i, n in enumerate(names)}
    truth_idx = np.array(
        sorted(index[a] for a in pl.read_csv(args.truth, separator="\t")["Entry"])
    )
    draws = pl.read_csv(args.draws, separator="\t")
    lineage_of = dict(zip(draws["query_id"], draws["lineage"]))
    kind_of = dict(zip(draws["query_id"], draws["kind"]))

    cls = pl.read_csv(args.alphabet_classes, separator="\t")
    hp = set(
        cls.filter((pl.col("alphabet") == HP_ALPHABET) & (pl.col("class_index") == 0))[
            "residue"
        ]
    )

    es = pl.read_csv(args.eisenberg, separator="\t", comment_prefix="#")
    eisenberg = dict(zip(es["residue"], es["hydrophobicity"]))

    feats = {"composition_aa20": ({aa: i for i, aa in enumerate(AA20)}, 20)}
    for a in sorted(cls["alphabet"].unique()):
        sub = cls.filter(pl.col("alphabet") == a)
        feats[f"composition_{a}"] = (
            dict(zip(sub["residue"], sub["class_index"])),
            int(sub["class_index"].max()) + 1,
        )

    chunks = []
    for list_name, (classes, k) in feats.items():
        tm, qm = composition(t, classes, k), composition(q, classes, k)
        for i, qid in enumerate(qnames):
            chunks.append(
                keep_rows(
                    "composition",
                    list_name,
                    qid,
                    -euclid(qm[i], tm),
                    names,
                    truth_idx,
                    args.top_n,
                    n_total,
                )
            )
        print(f"{list_name}: done", file=sys.stderr)
    for list_name, fn in [
        ("period_hp_autocorr", lambda c: hp_autocorr(c, hp)),
        ("period_hp_moment", lambda c: hp_moment(c, eisenberg)),
    ]:
        tm, qm = fn(t), fn(q)
        for i, qid in enumerate(qnames):
            chunks.append(
                keep_rows(
                    "period",
                    list_name,
                    qid,
                    -cosine_distance(qm[i], tm),
                    names,
                    truth_idx,
                    args.top_n,
                    n_total,
                )
            )
        print(f"{list_name}: done", file=sys.stderr)

    motifs = pl.read_csv(args.motifs, separator="\t")
    for m in motifs.iter_rows(named=True):
        rx = re.compile(m["regex"])
        cover = np.array(
            [sum(len(x.group(0)) for x in rx.finditer(s)) for s in tseqs], dtype=float
        )
        for qid in qnames:
            if lineage_of[qid] == m["lineage"] and kind_of[qid] == "real":
                chunks.append(
                    keep_rows(
                        "motif",
                        "motif_regex",
                        qid,
                        cover,
                        names,
                        truth_idx,
                        args.top_n,
                        n_total,
                    )
                )
        print(
            f"motif {m['lineage']}: {int((cover > 0).sum())} entries match",
            file=sys.stderr,
        )

    out = pl.concat([pl.DataFrame(c) for c in chunks])
    out.write_parquet(args.out, compression="zstd")
    print(
        out.group_by("list_name")
        .agg(pl.col("query_id").n_unique().alias("n_queries"), pl.len().alias("rows"))
        .sort("list_name")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
