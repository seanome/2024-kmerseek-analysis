"""Shared pieces of the time-split test (notebook 280): decoys, the tune/test split and
one coordinate convention.

Coordinates: everything in this test is stored 1-based with the end included, as Pfam,
MMseqs2 and HMMER report it. kmerseek reports 0-based starts with exclusive ends;
`from_kmerseek` is the one place that converts.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

DECOY_PREFIX = "DECOY_"
DECOY_SEED = 280


def dipeptide_shuffle(seq: str, rng: random.Random) -> str:
    """A random sequence with the same first residue, last residue and count of every
    adjacent residue pair as `seq` (Altschul & Erickson 1985, Mol Biol Evol 2:526).

    Copied unchanged from scripts/make_decoy_queries.py on olgabot/midi-plus-0.4-decoy
    (PR 88, commit da09f1a), which copied it from notebook 256, so notebooks 256, 260,
    270 and 280 use one null.
    """
    if len(seq) < 3:
        return seq
    out = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        out[a].append(b)
    last = seq[-1]
    while True:
        final = {v: rng.choice(ends) for v, ends in out.items() if v != last}
        ok = True
        for v in final:
            seen, u = set(), v
            while u != last:
                if u in seen:
                    ok = False
                    break
                seen.add(u)
                u = final[u]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, ends in out.items():
        rest = list(ends)
        if v in final:
            rest.remove(final[v])
        rng.shuffle(rest)
        order[v] = rest + ([final[v]] if v in final else [])
    walk, u = [seq[0]], seq[0]
    for _ in range(len(seq) - 1):
        u = order[u].pop(0)
        walk.append(u)
    return "".join(walk)


def pair_counts(seq: str) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for a, b in zip(seq, seq[1:]):
        c[a + b] += 1
    return c


def query_half(accession: str) -> str:
    """'tune' when the first byte of SHA-1(accession) is even, else 'test'. A decoy gets
    its source protein's half. The rule of query_split() in notebooks/region_table_260.py
    (PR 88, commit f7e3df2), applied to a bare accession instead of an sp|...| header."""
    src = accession.removeprefix(DECOY_PREFIX)
    return "tune" if hashlib.sha1(src.encode()).digest()[0] % 2 == 0 else "test"


def from_kmerseek(start0: int, end_exclusive: int) -> tuple[int, int]:
    """kmerseek (0-based start, exclusive end) -> (1-based start, inclusive end)."""
    return start0 + 1, end_exclusive


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Residues shared by two 1-based inclusive intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)

SHORT_AA = 60  # "short region": under 60 aa, as the plan defines it
