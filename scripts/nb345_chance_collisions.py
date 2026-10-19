#!/usr/bin/env python3
"""How many hp_pbotc_1st_ed k = 30 words would two proteomes of a clade share by chance alone?

Notebook 345 section 2c. Written for the adversarial check on PR 172, to explain why the
synthetic Eukaryota proteomes share more words than the real ones. It does not use
nextflow-runs/proteome-novelty/bin/novelty.py, so it is a second, independent count.

Model: each proteome j has W_j windows of 30 standard residues. Its words follow a
first-order Markov chain on the two letters h and p, with that proteome's own h/p pair
frequencies. A word w is present in proteome j with probability 1 - exp(-W_j p_j(w)), and
proteomes are independent. The expected number of words present in at least two proteomes
is summed over all 2^30 words. p_j(w) depends only on w's first letter and its counts of
hh, hp, ph and pp steps, so the sum runs over those classes, each weighted by how many
words fall in it.

Reads the QfO 2020_04 FASTAs (QFO_DIR) and assets/proteomes.tsv. Writes
tables/345/345_chance_collisions.parquet (one row per clade) and
tables/345/345_chance_hp_composition.parquet (one row per proteome).
Runs on a laptop in about 3 minutes; reads one proteome at a time.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
QFO = Path(os.environ.get("QFO_DIR", "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"))
TABLE = ROOT / "nextflow-runs" / "proteome-novelty" / "assets" / "proteomes.tsv"
OUT = ROOT / "tables" / "345"
CLADES = ["Archaea", "Bacteria", "Eukaryota", "Fungi", "Vertebrata", "Mammalia"]
K = 30
HYDROPHOBIC = "ACFILMPVWY"  # hp_pbotc_1st_ed h; the other 10 standard residues are p
STANDARD = "ACDEFGHIKLMNPQRSTVWY"

LUT = np.full(256, 2, dtype=np.uint8)  # 0 = h, 1 = p, 2 = not a standard residue
for aa in STANDARD:
    LUT[ord(aa)] = 0 if aa in HYDROPHOBIC else 1


def read_sequences(path):
    seqs, cur = [], []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def hp_chain(path):
    """Windows of K standard residues, h/p pair counts (2 x 2)."""
    pairs = np.zeros((2, 2))
    windows = 0
    for s in read_sequences(path):
        c = LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        ok = (c[:-1] < 2) & (c[1:] < 2)
        np.add.at(pairs, (c[:-1][ok], c[1:][ok]), 1)
        bad = np.concatenate([[0], np.cumsum(c == 2)])
        if len(c) >= K:
            windows += int(((bad[K:] - bad[:-K]) == 0).sum())
    return windows, pairs


def word_classes():
    """(first letter, n_hh, n_hp, n_ph, n_pp) -> number of length-K h/p words in that class."""
    dp = defaultdict(int)
    for f in (0, 1):
        dp[(f, f, 0, 0, 0, 0)] = 1
    for _ in range(K - 1):
        nxt = defaultdict(int)
        for (f, last, *steps), m in dp.items():
            for x in (0, 1):
                s = list(steps)
                s[2 * last + x] += 1
                nxt[(f, x, *s)] += m
        dp = nxt
    classes = defaultdict(int)
    for (f, _last, *steps), m in dp.items():
        classes[(f, *steps)] += m
    keys = np.array(list(classes), dtype=float)
    counts = np.array(list(classes.values()), dtype=float)
    assert counts.sum() == 2**K
    return keys, counts


def main():
    proteomes = pl.read_csv(TABLE, separator="\t")
    keys, counts = word_classes()
    chain = {}
    for r in proteomes.iter_rows(named=True):
        if any(c in CLADES for c in r["clades"].split(",")):
            chain[r["proteome"]] = hp_chain(QFO / r["fasta"])
            print(r["oscode"], chain[r["proteome"]][0], flush=True)

    comp = []
    for r in proteomes.iter_rows(named=True):
        if r["proteome"] in chain:
            w, pairs = chain[r["proteome"]]
            comp.append({"proteome": r["proteome"], "oscode": r["oscode"], "clades": r["clades"],
                         "windows": w, "h_fraction": float(pairs.sum(1)[0] / pairs.sum()),
                         "h_after_h": float(pairs[0, 0] / pairs[0].sum()),
                         "p_after_p": float(pairs[1, 1] / pairs[1].sum())})
    pl.DataFrame(comp).write_parquet(OUT / "345_chance_hp_composition.parquet")

    rows = []
    for clade in CLADES:
        members = proteomes.filter(pl.col("clades").str.split(",").list.contains(clade))["proteome"]
        log_absent = np.zeros(len(counts))  # log Pr(word absent from every proteome)
        odds_sum = np.zeros(len(counts))  # sum over j of q_j / (1 - q_j)
        windows = 0
        for p in members:
            w, pairs = chain[p]
            windows += w
            pi = pairs.sum(1) / pairs.sum()
            T = pairs / pairs.sum(1, keepdims=True)
            logp = (np.log(pi[keys[:, 0].astype(int)]) + keys[:, 1] * np.log(T[0, 0])
                    + keys[:, 2] * np.log(T[0, 1]) + keys[:, 3] * np.log(T[1, 0]) + keys[:, 4] * np.log(T[1, 1]))
            lam = w * np.exp(logp)
            log_absent += -lam
            odds_sum += np.expm1(lam)  # q / (1 - q) with q = 1 - exp(-lam)
        absent = np.exp(log_absent)
        exactly_one = absent * odds_sum
        rows.append({"clade": clade, "n_proteomes": len(members), "windows": windows, "possible_words": 2**K,
                     "expected_distinct": float((counts * (1 - absent)).sum()),
                     "expected_shared_N": float((counts * (1 - absent - exactly_one)).sum())})
        print(rows[-1], flush=True)
    pl.DataFrame(rows).write_parquet(OUT / "345_chance_collisions.parquet")


if __name__ == "__main__":
    main()
