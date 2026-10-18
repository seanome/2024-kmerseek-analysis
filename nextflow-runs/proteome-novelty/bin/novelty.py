#!/usr/bin/env python3
"""Recurring-word accounting for notebook 343: does one more proteome add new local sequence?

A word is a k-mer of a proteome written in an alphabet (`hp_pbotc_1st_ed`, two letters, or
`protein`, the 20 amino acids). A word *recurs* in a set of proteomes when at least two of
them contain it. U(n) is the number of recurring words among n proteomes.

For every clade the pipeline keeps one number per distinct word: the bitmask of the clade
proteomes that contain it (bit i = proteome i of the clade, at most 64 proteomes). Words
with the same bitmask are interchangeable for every count below, so the stored table is
(mask, n_words). From it, the average of U(n) over every ordering of the proteomes has an
exact formula (hypergeometric, see `expected_recurring`), so no random orderings are drawn.

Subcommands
-----------
null       write a synthetic proteome: same protein lengths, residues drawn from a
           first-order Markov chain fitted to that proteome's own residue pairs
masks      clade FASTAs -> (mask, n_words) parquet, plus the words in exactly two proteomes
mmseqs     MMseqs2 cluster TSV -> the same (mask, n_families) parquet, one family per cluster
rarefy     (mask, n) parquet -> discovery curve, Heaps' law fit, held-out prediction errors
examples   longest stretch of newly recurring words for one specimen, with both sequences
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Amino acid -> letter, from kmerseek's encoding tables (kmerseek/src/rust/hp_alphabets.rs),
# the same groups as notebooks/ortholog_analysis_utils.py ALPHABET_AA_GROUPS.
ALPHABETS: dict[str, dict[str, str]] = {
    "protein": {aa: aa for aa in AMINO_ACIDS},
    "hp_pbotc_1st_ed": {"h": "ACFILMPVWY", "p": "DEGHKNQRST"},
}

INVALID = 255
IDX_BITS = 6  # proteome index packed under the word while sorting; clades hold <= 64


# --------------------------------------------------------------------------------------------
# FASTA


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    names, seqs, chunks = [], [], []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if names:
                    seqs.append("".join(chunks))
                names.append(line[1:].split()[0])
                chunks = []
            else:
                chunks.append(line.strip())
    if names:
        seqs.append("".join(chunks))
    return names, seqs


def write_fasta(path: Path, names: list[str], seqs: list[str]) -> None:
    with gzip.open(path, "wt") as fh:
        for n, s in zip(names, seqs):
            fh.write(f">{n}\n")
            for i in range(0, len(s), 60):
                fh.write(s[i : i + 60] + "\n")


# --------------------------------------------------------------------------------------------
# Encoding


def alphabet_table(alphabet: str) -> tuple[np.ndarray, int]:
    """256-entry byte -> symbol code table; non-standard residues (X, U, B, Z, *) -> INVALID."""
    groups = ALPHABETS[alphabet]
    table = np.full(256, INVALID, dtype=np.uint8)
    for code, aas in enumerate(groups.values()):
        for aa in aas:
            table[ord(aa)] = code
            table[ord(aa.lower())] = code
    return table, len(groups)


def word_bits(alphabet: str, ksize: int) -> float:
    return ksize * math.log2(len(ALPHABETS[alphabet]))


def encode_words(seqs: list[str], alphabet: str, ksize: int, positions: bool = True):
    """Every valid window of every sequence as an integer word.

    Returns (words, protein index, 0-based start), or the words alone when `positions` is
    False. A window holding a non-standard residue is
    skipped. Sequences are joined with one INVALID separator, so no window spans two proteins.
    """
    table, base = alphabet_table(alphabet)
    if word_bits(alphabet, ksize) > 64 - IDX_BITS:
        raise SystemExit(f"{alphabet} k={ksize} does not fit in {64 - IDX_BITS} bits")
    joined = ("\x00".join(seqs) + "\x00").encode("ascii", "replace")
    codes = table[np.frombuffer(joined, dtype=np.uint8)]
    n_windows = len(codes) - ksize + 1
    if n_windows <= 0:
        empty = np.empty(0, np.uint64)
        return (empty, np.empty(0, np.int64), np.empty(0, np.int64)) if positions else empty
    bad = np.concatenate([[0], np.cumsum(codes == INVALID)])
    valid = (bad[ksize:] - bad[:-ksize]) == 0
    words = np.zeros(n_windows, dtype=np.uint64)
    b = np.uint64(base)
    for j in range(ksize):
        words *= b
        words += codes[j : j + n_windows].astype(np.uint64)
    del bad
    if not positions:
        return words[valid]
    starts_of = np.cumsum([0] + [len(s) + 1 for s in seqs])
    pos = np.nonzero(valid)[0]
    prot = np.searchsorted(starts_of, pos, side="right") - 1
    return words[pos], prot, pos - starts_of[prot]


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def fracminhash(words: np.ndarray, scaled: int) -> np.ndarray:
    """Keep a word when its hash falls below 2^64 / scaled. scaled = 1 keeps every word."""
    if scaled == 1:
        return words
    with np.errstate(over="ignore"):
        h = splitmix64(words)
    return words[h < np.uint64((2**64 - 1) // scaled)]


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    x = x.astype(np.uint64)
    c = np.zeros(len(x), dtype=np.int64)
    for shift in range(0, 64, 8):
        c += POP8[((x >> np.uint64(shift)) & np.uint64(255)).astype(np.int64)]
    return c


POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


# --------------------------------------------------------------------------------------------
# null


def cmd_null(a: argparse.Namespace) -> None:
    names, seqs = read_fasta(a.fasta)
    table, _ = alphabet_table("protein")
    first = np.zeros(20)
    pairs = np.zeros((20, 20))
    for s in seqs:
        c = table[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
        if len(c) and c[0] != INVALID:
            first[c[0]] += 1
        ok = (c[:-1] != INVALID) & (c[1:] != INVALID)
        np.add.at(pairs, (c[:-1][ok], c[1:][ok]), 1)
    # A residue never seen as a first letter of a pair still needs a row to step from.
    pairs += 1e-9
    trans_cum = np.cumsum(pairs / pairs.sum(1, keepdims=True), axis=1)
    first_cum = np.cumsum(first / first.sum())

    rng = np.random.default_rng(a.seed)
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    # Longest first, so the proteins still being extended at step t are a prefix.
    order = np.argsort(-lengths, kind="stable")
    L = lengths[order]
    offset = np.concatenate([[0], np.cumsum(L)])
    flat = np.zeros(int(offset[-1]), dtype=np.uint8)
    prev = np.minimum(np.searchsorted(first_cum, rng.random(len(L))), 19)
    n_active = int((L > 0).sum())
    flat[offset[:n_active]] = prev[:n_active]
    for t in range(1, int(L.max()) if len(L) else 0):
        n_active = int(np.searchsorted(-L, -t, side="left"))  # proteins longer than t
        u = rng.random(n_active)
        nxt = np.minimum((u[:, None] > trans_cum[prev[:n_active]]).sum(1), 19)
        prev[:n_active] = nxt
        flat[offset[:n_active] + t] = nxt
    letters = np.frombuffer(AMINO_ACIDS.encode(), dtype=np.uint8)
    text = letters[flat].tobytes().decode()
    null_seqs = [""] * len(seqs)
    for rank, i in enumerate(order):
        null_seqs[i] = text[offset[rank] : offset[rank + 1]]
    write_fasta(a.out, [f"null_{n}" for n in names], null_seqs)


# --------------------------------------------------------------------------------------------
# masks


def clade_masks(fastas: list[Path], alphabet: str, ksize: int, scaled: int):
    """Sorted distinct words and the bitmask of proteomes containing each."""
    if len(fastas) > 64:
        raise SystemExit("a clade holds at most 64 proteomes")
    packed_parts, n_windows = [], []
    for i, f in enumerate(fastas):
        _, seqs = read_fasta(f)
        w = encode_words(seqs, alphabet, ksize, positions=False)
        del seqs
        n_windows.append(len(w))
        u = fracminhash(np.unique(w), scaled)
        packed_parts.append((u << np.uint64(IDX_BITS)) | np.uint64(i))
        print(f"  {f.name}: {len(w):_} windows, {len(u):_} distinct words", file=sys.stderr)
    packed = np.concatenate(packed_parts)
    del packed_parts
    packed.sort()
    words = packed >> np.uint64(IDX_BITS)
    bits = np.left_shift(np.uint64(1), packed & np.uint64(2**IDX_BITS - 1))
    del packed
    starts = np.concatenate([[0], np.nonzero(words[1:] != words[:-1])[0] + 1])
    masks = np.bitwise_or.reduceat(bits, starts)
    return words[starts], masks, n_windows


def cmd_masks(a: argparse.Namespace) -> None:
    fastas = [Path(p) for p in a.fastas]
    words, masks, n_windows = clade_masks(fastas, a.alphabet, a.ksize, a.scaled)
    m, n = np.unique(masks, return_counts=True)
    pl.DataFrame({"mask": m, "n": n.astype(np.int64)}).write_parquet(a.out)
    if a.pairs_out:
        two = popcount(masks) == 2
        pl.DataFrame({"word": words[two], "mask": masks[two]}).write_parquet(a.pairs_out)
    json.dump(
        {"n_windows": n_windows, "n_distinct_words": int(len(words)), "n_masks": int(len(m))},
        open(a.summary, "w"),
    )


# --------------------------------------------------------------------------------------------
# mmseqs


def cmd_mmseqs(a: argparse.Namespace) -> None:
    """Cluster TSV (representative, member); member names are '<proteome>__<accession>'."""
    order = {p: i for i, p in enumerate(a.proteomes)}
    tsv = pl.read_csv(a.clusters, separator="\t", has_header=False, new_columns=["rep", "member"])
    tsv = tsv.with_columns(
        pl.col("member").str.split("__").list.first().replace_strict(order, return_dtype=pl.Int64).alias("i")
    )
    fam = tsv.group_by("rep").agg(pl.col("i").unique())
    masks = np.array([sum(1 << int(i) for i in idx) for idx in fam["i"].to_list()], dtype=np.uint64)
    m, n = np.unique(masks, return_counts=True)
    pl.DataFrame({"mask": m, "n": n.astype(np.int64)}).write_parquet(a.out)


# --------------------------------------------------------------------------------------------
# rarefy


def hist_in_subset(masks: np.ndarray, n: np.ndarray, subset: int, size: int) -> np.ndarray:
    """c[m] = number of words held by exactly m proteomes of `subset`."""
    m = popcount(masks & np.uint64(subset))
    return np.bincount(m, weights=n, minlength=size + 1)[: size + 1]


def expected_recurring(c: np.ndarray, size: int) -> np.ndarray:
    """Average U(n), n = 1..size, over every ordering of `size` proteomes.

    A word held by m of them recurs among the first n when at least two of its m holders
    are among those n: 1 - Pr(0 holders) - Pr(1 holder), hypergeometric.
    """
    out = np.zeros(size)
    for n in range(1, size + 1):
        tot = math.comb(size, n)
        p = np.array(
            [
                1.0 - (math.comb(size - m, n) + m * math.comb(size - m, n - 1)) / tot if m >= 2 else 0.0
                for m in range(size + 1)
            ]
        )
        out[n - 1] = float(np.dot(c, p))
    return out


def true_added(masks: np.ndarray, n: np.ndarray, subset: int, nxt: int) -> float:
    """Words that become recurring when proteome `nxt` joins `subset`: held by exactly one
    proteome of the subset and by `nxt`."""
    in_obs = popcount(masks & np.uint64(subset))
    return float(n[(in_obs == 1) & ((masks & np.uint64(1 << nxt)) != 0)].sum())


def heaps_fit(u: np.ndarray, n_max: int) -> tuple[float, float]:
    """Least squares of log U(n) = log kappa + gamma log n over n = 2..n_max with U(n) > 0."""
    n = np.arange(2, n_max + 1)
    y = u[1:n_max]
    ok = y > 0
    if ok.sum() < 2:
        return float("nan"), float("nan")
    gamma, logk = np.polyfit(np.log(n[ok]), np.log(y[ok]), 1)
    return float(math.exp(logk)), float(gamma)


def splits_for(clade: str, size: int, n_splits: int) -> list[tuple[tuple[int, ...], int]]:
    """(observed members, next member) pairs. Same for every setting of a clade."""
    n_hidden = math.ceil(0.2 * size)
    n_obs = size - n_hidden
    total = math.comb(size, n_obs) * n_hidden
    rng = random.Random(f"nb343:{clade}:{size}")
    seen: set = set()
    out = []
    while len(out) < min(n_splits, total):
        perm = list(range(size))
        rng.shuffle(perm)
        key = (tuple(sorted(perm[:n_obs])), perm[n_obs])
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def cmd_rarefy(a: argparse.Namespace) -> None:
    df = pl.read_parquet(a.masks)
    masks = df["mask"].to_numpy().astype(np.uint64)
    n = df["n"].to_numpy().astype(np.float64)
    size = len(a.proteomes)
    full = (1 << size) - 1

    curve = expected_recurring(hist_in_subset(masks, n, full, size), size)
    kappa, gamma = heaps_fit(curve, size)
    pl.DataFrame({"n": np.arange(1, size + 1), "U": curve}).with_columns(
        pl.lit(a.clade).alias("clade"), pl.lit(a.setting).alias("setting")
    ).write_parquet(a.curve_out)

    # Newly recurring words if each member were sequenced last: held by it and one other.
    pc = popcount(masks)
    last = []
    for i, p in enumerate(a.proteomes):
        sel = (pc == 2) & ((masks & np.uint64(1 << i)) != 0)
        last.append({"proteome": p, "added_if_last": float(n[sel].sum())})
    pl.DataFrame(last).with_columns(
        pl.lit(a.clade).alias("clade"), pl.lit(a.setting).alias("setting")
    ).write_parquet(a.last_out)

    rows = []
    for obs, nxt in splits_for(a.clade, size, a.n_splits):
        sub = sum(1 << i for i in obs)
        n_obs = len(obs)
        u_obs = expected_recurring(hist_in_subset(masks, n, sub, n_obs), n_obs)
        k_hat, g_hat = heaps_fit(u_obs, n_obs)
        pred = k_hat * ((n_obs + 1) ** g_hat - n_obs**g_hat)
        true = true_added(masks, n, sub, nxt)
        rows.append(
            {
                "observed": ",".join(a.proteomes[i] for i in obs),
                "next": a.proteomes[nxt],
                "n_observed": n_obs,
                "kappa_hat": k_hat,
                "gamma_hat": g_hat,
                "predicted_added": pred,
                "true_added": true,
                "relative_error": abs(pred - true) / true if true > 0 else float("nan"),
            }
        )
    pl.DataFrame(rows).with_columns(
        pl.lit(a.clade).alias("clade"), pl.lit(a.setting).alias("setting")
    ).write_parquet(a.splits_out)
    json.dump(
        {"clade": a.clade, "setting": a.setting, "size": size, "kappa": kappa, "gamma": gamma,
         "U_full": float(curve[-1])},
        open(a.fit_out, "w"),
    )


# --------------------------------------------------------------------------------------------
# examples


def longest_runs(prot: np.ndarray, start: np.ndarray, partner: np.ndarray, top: int):
    """Runs of consecutive window starts in one protein sharing one partner proteome."""
    if len(prot) == 0:
        return []
    o = np.lexsort((start, partner, prot))
    p, s, q = prot[o], start[o], partner[o]
    brk = np.concatenate([[True], (p[1:] != p[:-1]) | (q[1:] != q[:-1]) | (s[1:] != s[:-1] + 1)])
    ids = np.cumsum(brk) - 1
    lens = np.bincount(ids)
    first = np.nonzero(brk)[0]
    best = np.argsort(-lens, kind="stable")[:top]
    return [(int(p[first[r]]), int(s[first[r]]), int(lens[r]), int(q[first[r]])) for r in best]


def match_line(a: str, b: str) -> str:
    return "".join("|" if x == y else " " for x, y in zip(a, b))


def cmd_examples(a: argparse.Namespace) -> None:
    fastas = [Path(p) for p in a.fastas]
    if a.specimen == "auto":
        # Most newly recurring words when added last; ties go to the earlier proteome id.
        last = pl.read_parquet(a.last).sort(["added_if_last", "proteome"], descending=[True, False])
        a.specimen = last["proteome"][0]
    i = a.proteomes.index(a.specimen)
    pairs = pl.read_parquet(a.pairs)
    words = pairs["word"].to_numpy().astype(np.uint64)
    masks = pairs["mask"].to_numpy().astype(np.uint64)
    bit = np.uint64(1 << i)
    mine = (masks & bit) != 0
    words, partner_mask = words[mine], masks[mine] ^ bit
    o = np.argsort(words)
    words, partner_mask = words[o], partner_mask[o]
    partner_idx = np.log2(partner_mask.astype(np.float64)).round().astype(np.int64)

    names, seqs = read_fasta(fastas[i])
    w, prot, start = encode_words(seqs, a.alphabet, a.ksize)
    hit = np.isin(w, words)
    w, prot, start = w[hit], prot[hit], start[hit]
    q = partner_idx[np.searchsorted(words, w)]
    out = []
    for p_i, s0, run, j in longest_runs(prot, start, q, a.top):
        span = run + a.ksize - 1
        x_seq = seqs[p_i][s0 : s0 + span]
        # Find the same first word in the partner and read the same span there.
        pn, ps = read_fasta(fastas[j])
        pw, pp, pst = encode_words(ps, a.alphabet, a.ksize)
        first_word = encode_words([x_seq[: a.ksize]], a.alphabet, a.ksize)[0][0]
        at = np.nonzero(pw == first_word)[0]
        table, _ = alphabet_table(a.alphabet)
        letters = list(ALPHABETS[a.alphabet])

        def enc(seq: str) -> str:
            return "".join(letters[c] if c != INVALID else "?" for c in table[np.frombuffer(seq.encode(), np.uint8)])

        x_enc = enc(x_seq)
        # A repeated word can sit at several places in the partner; keep the one whose
        # whole span agrees best with the specimen's.
        cands = [(int(pp[k]), int(pst[k])) for k in at]
        y_prot, y_start = max(
            cands, key=lambda c: sum(u == v for u, v in zip(x_enc, enc(ps[c[0]][c[1] : c[1] + span])))
        )
        y_seq = ps[y_prot][y_start : y_start + span]
        y_enc = enc(y_seq)
        out.append(
            {
                "specimen": a.specimen,
                "specimen_protein": names[p_i],
                "specimen_start": s0 + 1,
                "specimen_end": s0 + span,
                "specimen_length": len(seqs[p_i]),
                "partner": a.proteomes[j],
                "partner_protein": pn[y_prot],
                "partner_start": y_start + 1,
                "partner_end": y_start + span,
                "partner_length": len(ps[y_prot]),
                "n_words": run,
                "n_partner_copies": len(at),
                "length": span,
                "specimen_seq": x_seq,
                "partner_seq": y_seq,
                "aa_match": match_line(x_seq, y_seq),
                "n_identical": sum(c == d for c, d in zip(x_seq, y_seq)),
                "specimen_encoded": x_enc,
                "partner_encoded": y_enc,
                "encoded_match": match_line(x_enc, y_enc),
                "n_encoded_identical": sum(c == d for c, d in zip(x_enc, y_enc)),
            }
        )
    pl.DataFrame(out).with_columns(pl.lit(a.clade).alias("clade")).write_parquet(a.out)


# --------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("null")
    s.add_argument("--fasta", type=Path, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("masks")
    s.add_argument("--fastas", nargs="+", required=True)
    s.add_argument("--alphabet", choices=list(ALPHABETS), required=True)
    s.add_argument("--ksize", type=int, required=True)
    s.add_argument("--scaled", type=int, default=1)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--pairs-out", type=Path)
    s.add_argument("--summary", type=Path, required=True)

    s = sub.add_parser("mmseqs")
    s.add_argument("--clusters", type=Path, required=True)
    s.add_argument("--proteomes", nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("rarefy")
    s.add_argument("--masks", type=Path, required=True)
    s.add_argument("--clade", required=True)
    s.add_argument("--setting", required=True)
    s.add_argument("--proteomes", nargs="+", required=True)
    s.add_argument("--n-splits", type=int, default=200)
    s.add_argument("--curve-out", type=Path, required=True)
    s.add_argument("--last-out", type=Path, required=True)
    s.add_argument("--splits-out", type=Path, required=True)
    s.add_argument("--fit-out", type=Path, required=True)

    s = sub.add_parser("examples")
    s.add_argument("--fastas", nargs="+", required=True)
    s.add_argument("--proteomes", nargs="+", required=True)
    s.add_argument("--clade", required=True)
    s.add_argument("--specimen", required=True, help="proteome id, or 'auto' to read --last")
    s.add_argument("--last", type=Path, help="rarefy --last-out table, for --specimen auto")
    s.add_argument("--pairs", type=Path, required=True)
    s.add_argument("--alphabet", choices=list(ALPHABETS), required=True)
    s.add_argument("--ksize", type=int, required=True)
    s.add_argument("--top", type=int, default=3)
    s.add_argument("--out", type=Path, required=True)

    a = ap.parse_args()
    {"null": cmd_null, "masks": cmd_masks, "mmseqs": cmd_mmseqs, "rarefy": cmd_rarefy,
     "examples": cmd_examples}[a.cmd](a)


if __name__ == "__main__":
    main()
