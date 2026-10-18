"""Linked word pairs (strobemers) counted against ice-binding proteins (notebook 298).

A key is two exact words of ``a`` letters in one reduced alphabet, with ``g`` residues
between the end of the first word and the start of the second. Five ways to build keys:

* ``pair_gap``: every pair with g from 1 to 40; key = (word1, g, word2).
* ``pair_gapbin3``: the same pairs; key = (word1, g // 3, word2).
* ``pair_nogap``: every pair with g from 0 to 32; key = (word1, word2). This is notebook
  294's ``all_a6_g32_none`` key when a = 6 (its gaps also run from 0 to 32).
* ``randstrobe_gap``: one key per position; the second word is the one notebook 294's
  randstrobe rule picks among starts g = 0..32; the key keeps g.
* ``randstrobe_nogap``: the same choice; key = (word1, word2), notebook 294's randstrobe.

``kmer`` is one exact word of length k, used only to reproduce notebook 295 section 9
(scripts/convergence_kmer_reach_ceiling.py) through this code before any pair is counted.

Word codes and randstrobe partners for two-letter alphabets come from notebook 294's
``linked_pairs_utils`` unchanged. For Dayhoff and the 20 amino acids the same formulas are
written for a larger alphabet (``word_codes_any``, ``randstrobe_partner_any``); the tests
check that they equal notebook 294's on two-letter input.

Counting: every sequence's distinct keys are found at once on one concatenated array, then
a matrix of entries by query keys times its transpose for the queries gives, for every
(entry, query), the number of distinct keys they share.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import linked_pairs_utils as lp
from hp_conservation_utils import ALPHABET_CLUSTERS, GAP

# The one alphabet added here: threonine against everything else.
EXTRA_CLUSTERS = {"thr2": ["T", "ACDEFGHIKLMNPQRSVWY"]}
CLUSTERS = {**ALPHABET_CLUSTERS, **EXTRA_CLUSTERS}

FAMILIES = [
    "pair_gap",
    "pair_gapbin3",
    "pair_nogap",
    "randstrobe_gap",
    "randstrobe_nogap",
]
PAIR_GAPS = range(1, 41)  # pair_gap, pair_gapbin3
GAP_BIN = 3
NOGAP_GAPS = range(0, 33)  # pair_nogap, as notebook 294's g32 window
RAND_WINDOW = 32  # randstrobe: second word starts a + 0 .. a + 32 after the first

# Concatenated sequences are separated by this many unknown letters, more than any gap
# plus word length, so no word or randstrobe window crosses from one sequence into the next.
SEPARATOR = 64
_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)


def table(alphabet: str) -> np.ndarray:
    """Byte -> class index (GAP for anything outside the 20 residues)."""
    t = np.full(256, GAP, dtype=np.uint8)
    for i, cl in enumerate(CLUSTERS[alphabet]):
        for r in cl:
            t[ord(r)] = i
            t[ord(r.lower())] = i
    return t


def base(alphabet: str) -> int:
    return len(CLUSTERS[alphabet])


def encode(seq: str, alphabet: str) -> np.ndarray:
    return table(alphabet)[np.frombuffer(seq.encode(), dtype=np.uint8)]


def word_codes_any(cls: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Notebook 294's word_codes for an alphabet of b letters: the base-b integer of the
    a-letter word at each start, and whether the word holds no unknown letter.

    For b = 2 this calls linked_pairs_utils.word_codes itself. Above 2**62 the code would
    overflow, so a word that long gets a 64-bit hash instead (used only for exact k-mers
    up to 40 letters in the 20- and 6-letter alphabets)."""
    if b == 2 and a < 62:
        return lp.word_codes(cls, a)
    return _word_codes_general(cls, a, b)


def _word_codes_general(cls: np.ndarray, a: int, b: int):
    """word_codes_any without the call to notebook 294's function (tested against it)."""
    n = cls.size
    if n < a:
        return np.empty(0, np.int64), np.empty(0, bool)
    bad = cls == GAP
    ok = ~np.lib.stride_tricks.sliding_window_view(bad, a).any(axis=1)
    x = np.where(bad, 0, cls).astype(np.int64)
    if b**a < 2**62:
        w = np.lib.stride_tricks.sliding_window_view(x, a)
        codes = w @ (np.int64(b) ** np.arange(a - 1, -1, -1, dtype=np.int64))
        return codes, ok
    m = n - a + 1
    h = np.zeros(m, np.uint64)
    xu = (x + 1).astype(np.uint64)
    with np.errstate(over="ignore"):
        for i in range(a):
            h = h * _HASH_MULT + xu[i : i + m]
    codes = (lp._mix64(h) >> np.uint64(1)).astype(np.int64)
    return codes, ok


def randstrobe_partner_any(cls: np.ndarray, a: int, gap_max: int, b: int) -> np.ndarray:
    """linked_pairs_utils.randstrobe_partner for an alphabet of b letters. For b = 2 it
    calls notebook 294's function; otherwise the same rule on base-b word codes: the
    partner of first word i is the start in i + a .. i + a + gap_max minimising
    (h(w1) + h(w2)) mod 997, leftmost on ties, -1 if none is valid."""
    if b == 2:
        return lp.randstrobe_partner(cls, a, gap_max)
    return _randstrobe_general(cls, a, gap_max, b)


def _randstrobe_general(cls: np.ndarray, a: int, gap_max: int, b: int) -> np.ndarray:
    """randstrobe_partner_any without the call to notebook 294's function."""
    codes, ok = _word_codes_general(cls, a, b)
    n = codes.size
    sel = np.full(cls.size, -1, dtype=np.int64)
    if n == 0:
        return sel
    hp = (lp._mix64(codes) % np.uint64(lp.RANDSTROBE_PRIME)).astype(np.int64)
    width = gap_max + 1
    big = np.int64(1 << 40)
    pad_h = np.concatenate([hp, np.zeros(a + width, np.int64)])
    pad_ok = np.concatenate([ok, np.zeros(a + width, bool)])
    j = np.arange(n)[:, None] + a + np.arange(width)[None, :]
    score = (hp[:, None] + pad_h[j]) % lp.RANDSTROBE_PRIME
    score = np.where(pad_ok[j], score, big)
    best = score.argmin(axis=1)
    has = score[np.arange(n), best] < big
    sel[:n] = np.where(has & ok, np.arange(n) + a + best, -1)
    return sel


def n_gap_values(family: str) -> int:
    if family == "pair_gap":
        return max(PAIR_GAPS) + 1
    if family == "pair_gapbin3":
        return max(PAIR_GAPS) // GAP_BIN + 1
    if family == "randstrobe_gap":
        return RAND_WINDOW + 1
    return 1


def pair_key(w1, gfield, w2, n_words: int, n_gaps: int):
    """One integer per key: (word1, gap field, word2) in mixed radix."""
    return (w1 * n_gaps + gfield) * n_words + w2


def sequence_keys(seq: str, alphabet: str, a: int, family: str) -> dict[int, list]:
    """Every key of one sequence with where it sits: key -> [(start1, g, start2), ...].

    Written separately from the concatenated counter below, one sequence at a time,
    so the tests can check the counter against it and the notebook can print residues.
    """
    cls = encode(seq, alphabet)
    b = base(alphabet)
    codes, ok = word_codes_any(cls, a, b)
    n_words = b**a if b**a < 2**62 else 2**62
    out: dict[int, list] = {}
    if family == "kmer":
        for i in np.flatnonzero(ok):
            out.setdefault(int(codes[i]), []).append((int(i), 0, int(i)))
        return out
    ng = n_gap_values(family)
    if family.startswith("pair_"):
        gaps = NOGAP_GAPS if family == "pair_nogap" else PAIR_GAPS
        for i in range(codes.size):
            for g in gaps:
                j = i + a + g
                if j >= codes.size:
                    break
                if not (ok[i] and ok[j]):
                    continue
                gf = {"pair_gap": g, "pair_gapbin3": g // GAP_BIN, "pair_nogap": 0}[
                    family
                ]
                k = pair_key(int(codes[i]), gf, int(codes[j]), n_words, ng)
                out.setdefault(k, []).append((i, g, j))
        return out
    sel = randstrobe_partner_any(cls, a, RAND_WINDOW, b)
    for i in range(codes.size):
        j = int(sel[i])
        if j < 0:
            continue
        g = j - i - a
        gf = g if family == "randstrobe_gap" else 0
        k = pair_key(int(codes[i]), gf, int(codes[j]), n_words, ng)
        out.setdefault(k, []).append((i, g, j))
    return out


def all_keys(
    seqs: list[str], alphabet: str, a: int, family: str
) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (sequence index, key) pairs over a list of sequences, sorted.

    Pairs are found on one concatenated array (sequences separated by SEPARATOR unknown
    letters, and a pair is kept only if both words lie in the same sequence). Randstrobe
    partners are chosen one sequence at a time, exactly as notebook 294 does."""
    b = base(alphabet)
    n_words = b**a if b**a < 2**62 else 2**62
    ng = n_gap_values(family)
    tab = table(alphabet)
    sep = np.full(SEPARATOR, GAP, np.uint8)
    parts, ids = [], []
    for s_i, s in enumerate(seqs):
        parts += [tab[np.frombuffer(s.encode(), dtype=np.uint8)], sep]
        ids += [np.full(len(s), s_i, np.int32), np.full(SEPARATOR, -1, np.int32)]
    cls = np.concatenate(parts)
    sid = np.concatenate(ids)
    if family.startswith("randstrobe"):
        sel = np.full(cls.size, -1, np.int64)
        off = 0
        for s in seqs:
            local = randstrobe_partner_any(cls[off : off + len(s)], a, RAND_WINDOW, b)
            sel[off : off + len(s)] = np.where(local >= 0, local + off, -1)
            off += len(s) + SEPARATOR
    codes, ok = word_codes_any(cls, a, b)
    m = codes.size
    seq_parts, key_parts = [], []
    if family == "kmer":
        i = np.flatnonzero(ok)
        seq_parts.append(sid[i])
        key_parts.append(codes[i])
    elif family.startswith("pair_"):
        gaps = NOGAP_GAPS if family == "pair_nogap" else PAIR_GAPS
        for g in gaps:
            off = a + g
            if m <= off:
                break
            i = np.arange(m - off)
            j = i + off
            keep = ok[i] & ok[j] & (sid[i] == sid[j])
            i, j = i[keep], j[keep]
            gf = {"pair_gap": g, "pair_gapbin3": g // GAP_BIN, "pair_nogap": 0}[family]
            seq_parts.append(sid[i])
            key_parts.append(pair_key(codes[i], gf, codes[j], n_words, ng))
    else:
        i = np.flatnonzero(sel[:m] >= 0)
        j = sel[i]
        g = j - i - a
        gf = g if family == "randstrobe_gap" else np.zeros_like(g)
        seq_parts.append(sid[i])
        key_parts.append(pair_key(codes[i], gf, codes[j], n_words, ng))
    s_all = np.concatenate(seq_parts).astype(np.int64)
    k_all = np.concatenate(key_parts).astype(np.int64)
    # Distinct (sequence, key): sort by key within sequence and drop repeats.
    order = np.lexsort((k_all, s_all))
    s_all, k_all = s_all[order], k_all[order]
    first = np.ones(s_all.size, bool)
    first[1:] = (s_all[1:] != s_all[:-1]) | (k_all[1:] != k_all[:-1])
    return s_all[first], k_all[first]


def shared_counts(
    n_entries: int, n_queries: int, s_ids: np.ndarray, keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distinct keys shared by every (entry, query).

    s_ids/keys are all_keys output over entries (ids 0..n_entries-1) followed by queries
    (ids n_entries..). Returns the shared-key matrix (entries x queries) and the number of
    distinct keys of every sequence (entries then queries)."""
    n_keys = np.bincount(s_ids, minlength=n_entries + n_queries)
    is_q = s_ids >= n_entries
    union = np.unique(keys[is_q])
    if union.size == 0:
        return np.zeros((n_entries, n_queries), np.int32), n_keys

    def matrix(sel, row_offset, n_rows):
        k = keys[sel]
        pos = np.searchsorted(union, k)
        hit = (pos < union.size) & (union[np.minimum(pos, union.size - 1)] == k)
        rows = s_ids[sel][hit] - row_offset
        return sp.csr_matrix(
            (np.ones(hit.sum(), np.float32), (rows, pos[hit])),
            shape=(n_rows, union.size),
        )

    E = matrix(~is_q, 0, n_entries)
    Q = matrix(is_q, n_entries, n_queries)
    if union.size * n_entries <= 50_000_000:
        S = E.toarray() @ Q.toarray().T
    else:
        S = (E @ Q.T).toarray()
    return np.rint(S).astype(np.int32), n_keys


def null_c(seq: str, period: int, rng: np.random.Generator) -> str:
    """Shuffle residues only among positions with the same remainder modulo the period,
    so the letters at each phase of the repeat are kept."""
    arr = np.frombuffer(seq.encode(), dtype=np.uint8).copy()
    for r in range(period):
        idx = np.arange(r, arr.size, period)
        arr[idx] = arr[rng.permutation(idx)]
    return arr.tobytes().decode()
