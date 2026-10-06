"""Seeds that allow mismatches, for a two-letter (H/P) alphabet (notebook 246).

A seed is the rule that decides which query-target diagonals a search looks at further.
kmerseek's seed is an exact k-mer: k consecutive positions in the same class. This module
scores other seed rules on the same footing:

* ``exact``: k consecutive class matches (kmerseek today).
* ``spaced``: class matches at the 1-positions of a pattern such as 110110111; the
  0-positions may mismatch (Ma, Tromp and Li 2002, PatternHunter).
* ``chained``: at least n exact k-mers on one diagonal, not overlapping, all inside a
  window of W positions; the gaps between them may mismatch (the two-hit rule of
  Altschul et al. 1997, generalised to n hits).
* ``window``: at least ``min_match`` class matches among the last L positions on one
  diagonal, anywhere in the window.
* ``extend``: no seed; the best ungapped segment score, +1 per class match and
  -``penalty`` per mismatch (the s = 0 case of notebook 232). An upper bound on what any
  seed followed by ungapped extension can see; it has no index.

``scaled = 2`` keeps the half of the seed keys whose hash is even, the FracMinHash rule
kmerseek uses for ``--scaled``. The key is the query's classes at the seed's 1-positions,
which equal the target's there, so query and target keep or drop the same seeds. It
halves the index. ``window`` and ``extend`` have no key and are only run at scaled 1.

The scan walks every diagonal of a query against a target, one position at a time, and
counts for each scheme the positions where the seed fires. A residue outside the
alphabet (X, U, a gap column in an alignment) ends the diagonal there and starts it again.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numba as nb
import numpy as np

GAP = 255

# Two-letter alphabets, copied from kmerseek src/rust/alphabets.rs (LEHNINGER_HP,
# THOMAS_DILL_HP), the same as hp_conservation_utils.ALPHABET_CLUSTERS. Class 0 = hydrophobic.
ALPHABETS = {
    "hp_thomas_dill2": ["ACFILMVWY", "DEGHKNPQRST"],
    "hp_lehninger2": ["AFGILMPVWY", "CDEHKNQRST"],
}

KIND = {"exact": 0, "spaced": 1, "chained": 2, "window": 3, "extend": 4}

# PatternHunter's default seed, weight 11 over 18 positions (Ma, Tromp and Li 2002).
PATTERNHUNTER = "111010010100110111"


def class_table(alphabet: str) -> np.ndarray:
    """Byte -> class (0 or 1) lookup; every other byte maps to GAP."""
    t = np.full(256, GAP, dtype=np.uint8)
    for i, cl in enumerate(ALPHABETS[alphabet]):
        for r in cl:
            t[ord(r)] = i
            t[ord(r.lower())] = i
    return t


def encode(seq: str, alphabet: str) -> np.ndarray:
    return class_table(alphabet)[np.frombuffer(seq.encode(), dtype=np.uint8)]


# ------------------------------------------------------------------ schemes ------------
@dataclass(frozen=True)
class Scheme:
    """One seed rule. ``pattern`` is written oldest position first, newest last."""

    family: str
    pattern: str = ""  # exact and spaced
    k: int = 0  # chained: length of each exact piece
    n: int = 0  # chained: pieces needed
    window: int = 0  # chained: W; window: L
    min_match: int = 0  # window
    penalty: float = 0.0  # extend
    scaled: int = 1

    @property
    def name(self) -> str:
        f = self.family
        if f == "exact":
            s = f"exact k={len(self.pattern)}"
        elif f == "spaced":
            s = f"spaced {self.pattern}"
        elif f == "chained":
            s = f"chained {self.n} x k={self.k} in {self.window}"
        elif f == "window":
            s = f"window {self.min_match} of {self.window}"
        else:
            s = f"extend +1/-{self.penalty:g}"
        return s if self.scaled == 1 else f"{s}, scaled 2"

    @property
    def weight(self) -> int:
        """Class matches the seed itself requires (the seed's information, in positions)."""
        if self.family in ("exact", "spaced"):
            return self.pattern.count("1")
        if self.family == "chained":
            return self.n * self.k
        if self.family == "window":
            return self.min_match
        return 0

    @property
    def span(self) -> int:
        if self.family in ("exact", "spaced"):
            return len(self.pattern)
        if self.family in ("chained", "window"):
            return self.window
        return 0


def tiled(unit: str, weight: int) -> str:
    """Repeat ``unit`` until it holds ``weight`` ones, then cut after the last one."""
    out, w = "", 0
    while w < weight:
        for c in unit:
            out += c
            w += c == "1"
            if w == weight:
                break
    return out


def default_schemes() -> list[Scheme]:
    out: list[Scheme] = []
    keyed: list[Scheme] = []
    for k in (8, 10, 12, 14, 16, 18, 20, 24, 28, 32):
        keyed.append(Scheme("exact", pattern="1" * k))
    for w in (12, 16, 20, 24):
        keyed.append(Scheme("spaced", pattern=tiled("110", w)))
    for w in (12, 16, 20):
        keyed.append(Scheme("spaced", pattern=tiled("10", w)))
    keyed.append(Scheme("spaced", pattern=PATTERNHUNTER))
    for k, n, w in (
        (5, 2, 40),
        (5, 3, 40),
        (5, 4, 40),
        (8, 2, 40),
        (8, 3, 60),
        (12, 2, 40),
        (12, 2, 60),
    ):
        keyed.append(Scheme("chained", k=k, n=n, window=w))
    for s in keyed:
        out.append(s)
        out.append(Scheme(**{**s.__dict__, "scaled": 2}))
    for L, j in (
        (20, 2),
        (20, 4),
        (30, 3),
        (30, 6),
        (40, 4),
        (40, 8),
        (60, 6),
        (60, 12),
    ):
        out.append(Scheme("window", window=L, min_match=L - j))
    out.append(Scheme("extend", penalty=2.0))
    return out


def scheme_arrays(schemes: list[Scheme]) -> dict[str, np.ndarray]:
    """The schemes as flat arrays for the numba kernel."""
    S = len(schemes)
    a = {
        "kind": np.zeros(S, np.int64),
        "pat": np.zeros(S, np.uint64),
        "span": np.zeros(S, np.int64),
        "k": np.zeros(S, np.int64),
        "n": np.zeros(S, np.int64),
        "win": np.zeros(S, np.int64),
        "minm": np.zeros(S, np.int64),
        "pen": np.zeros(S, np.float64),
        "scaled": np.ones(S, np.uint64),
        "salt": np.zeros(S, np.uint64),
    }
    for i, s in enumerate(schemes):
        a["kind"][i] = KIND[s.family]
        if s.family in ("exact", "spaced"):
            # Bit 0 is the newest position, so the pattern is read right to left.
            assert s.pattern[-1] == "1" and len(s.pattern) <= 64
            a["pat"][i] = np.uint64(int(s.pattern, 2))
            a["span"][i] = len(s.pattern)
        if s.family == "chained":
            assert s.window <= 64 and s.k <= s.window
            a["pat"][i] = np.uint64((1 << s.k) - 1)
        a["k"][i], a["n"][i], a["win"][i] = s.k, s.n, s.window
        a["minm"][i], a["pen"][i], a["scaled"][i] = s.min_match, s.penalty, s.scaled
        # The salt depends on the seed shape only, so the scaled-2 keys are a subset of
        # the scaled-1 keys of the same shape.
        shape = f"{s.family}|{s.pattern}|{s.k}"
        a["salt"][i] = np.uint64(
            zlib.crc32(shape.encode()) * 0x9E3779B97F4A7C15 % 2**64
        )
    return a


# ------------------------------------------------------------------ kernel -------------
@nb.njit(cache=True, inline="always")
def _mix(x):
    """splitmix64 finaliser."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(cache=True, inline="always")
def _kept(key, salt, scaled):
    if scaled == np.uint64(1):
        return True
    return _mix(key ^ salt) % scaled == np.uint64(0)


@nb.njit(cache=True, inline="always")
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@nb.njit(cache=True)
def scan_diagonal(
    q,
    t,
    i0,
    j0,
    length,
    kind,
    pat,
    span,
    k,
    n,
    win,
    minm,
    pen,
    scaled,
    salt,
    fires,
    pieces,
    best_score,
):
    """Walk one diagonal: q[i0 + x] against t[j0 + x] for x in 0..length-1.

    Adds to ``fires[s]`` the positions where scheme s fires, to ``pieces[s]`` the exact
    k-mer pieces a chained scheme finds (what its index returns), and sets
    ``best_score[s]`` to the best extension score for ``extend`` schemes.
    """
    S = kind.shape[0]
    one = np.uint64(1)
    full = np.uint64(0xFFFFFFFFFFFFFFFF)
    reg = np.uint64(0)  # class-match bits, bit 0 = newest position
    qreg = np.uint64(0)  # query class bits, bit 0 = newest position
    valid = 0  # positions since the last reset
    run = 0  # current exact run
    hist = np.zeros(S, np.uint64)  # chained: positions where a kept piece ends
    cur = np.zeros(S, np.float64)  # extend: running segment score
    for x in range(length):
        a = q[i0 + x]
        b = t[j0 + x]
        if a == 255 or b == 255:
            reg = np.uint64(0)
            qreg = np.uint64(0)
            valid = 0
            run = 0
            for s in range(S):
                hist[s] = np.uint64(0)
                cur[s] = 0.0
            continue
        m = a == b
        reg = (reg << one) | np.uint64(m)
        qreg = (qreg << one) | np.uint64(a)
        if valid < 64:
            valid += 1
        run = run + 1 if m else 0
        for s in range(S):
            kd = kind[s]
            if kd == 4:
                c = cur[s] + (1.0 if m else -pen[s])
                if c < 0.0:
                    c = 0.0
                cur[s] = c
                if c > best_score[s]:
                    best_score[s] = c
            elif kd == 2:
                hist[s] = hist[s] << one
                if m and run >= k[s] and _kept(qreg & pat[s], salt[s], scaled[s]):
                    hist[s] |= one
                    pieces[s] += 1
                    # Greedy count of non-overlapping pieces, newest first, whose
                    # starts all lie inside the last win[s] positions.
                    cnt = 0
                    back = 0
                    h = hist[s]
                    while back <= win[s] - k[s]:
                        if (h >> np.uint64(back)) & one:
                            cnt += 1
                            back += k[s]
                        else:
                            back += 1
                    if cnt >= n[s]:
                        fires[s] += 1
            elif not m:
                continue  # exact, spaced and window seeds fire only on a matching position
            elif kd == 0:
                if run >= span[s] and _kept(qreg & pat[s], salt[s], scaled[s]):
                    fires[s] += 1
            elif kd == 1:
                if (
                    valid >= span[s]
                    and (reg & pat[s]) == pat[s]
                    and _kept(qreg & pat[s], salt[s], scaled[s])
                ):
                    fires[s] += 1
            elif kd == 3:
                if valid >= win[s]:
                    mask = full if win[s] == 64 else (one << np.uint64(win[s])) - one
                    if _popcount(reg & mask) >= np.uint64(minm[s]):
                        fires[s] += 1
    return 0


@nb.njit(cache=True)
def scan_pair(q, t, kind, pat, span, k, n, win, minm, pen, scaled, salt):
    """Every diagonal of q against t. Returns per-diagonal fires and extension scores,
    shape (len(q) + len(t) - 1, S); row d is the diagonal t_pos - q_pos = d - (len(q) - 1).
    """
    m, L = q.shape[0], t.shape[0]
    S = kind.shape[0]
    nd = m + L - 1
    out = np.zeros((nd, S), np.float64)
    pieces = np.zeros(S, np.int64)
    for d in range(nd):
        off = d - (m - 1)
        i0 = -off if off < 0 else 0
        j0 = off if off > 0 else 0
        length = min(m - i0, L - j0)
        fires = np.zeros(S, np.int64)
        best = np.zeros(S, np.float64)
        scan_diagonal(
            q,
            t,
            i0,
            j0,
            length,
            kind,
            pat,
            span,
            k,
            n,
            win,
            minm,
            pen,
            scaled,
            salt,
            fires,
            pieces,
            best,
        )
        for s in range(S):
            out[d, s] = best[s] if kind[s] == 4 else fires[s]
    return out


@nb.njit(cache=True, parallel=True)
def scan_database(q, db, starts, kind, pat, span, k, n, win, minm, pen, scaled, salt):
    """q against every protein of a concatenated database (``starts`` has P + 1 entries).

    Returns, per protein and scheme: the score (most fires on one diagonal, or the best
    extension score), the total fires over all diagonals, and the pieces a chained
    scheme's index returns.
    """
    P = starts.shape[0] - 1
    S = kind.shape[0]
    m = q.shape[0]
    score = np.zeros((P, S), np.float64)
    total = np.zeros((P, S), np.int64)
    pieces = np.zeros((P, S), np.int64)
    for p in nb.prange(P):
        t = db[starts[p] : starts[p + 1]]
        L = t.shape[0]
        pc = np.zeros(S, np.int64)
        for d in range(m + L - 1):
            off = d - (m - 1)
            i0 = -off if off < 0 else 0
            j0 = off if off > 0 else 0
            length = min(m - i0, L - j0)
            fires = np.zeros(S, np.int64)
            best = np.zeros(S, np.float64)
            scan_diagonal(
                q,
                t,
                i0,
                j0,
                length,
                kind,
                pat,
                span,
                k,
                n,
                win,
                minm,
                pen,
                scaled,
                salt,
                fires,
                pc,
                best,
            )
            for s in range(S):
                v = best[s] if kind[s] == 4 else float(fires[s])
                if v > score[p, s]:
                    score[p, s] = v
                total[p, s] += fires[s]
        for s in range(S):
            pieces[p, s] = pc[s]
    return score, total, pieces


@nb.njit(cache=True, parallel=True)
def index_entries(db, starts, kind, pat, span, k, scaled, salt):
    """Database positions each keyed scheme would store in its index (k-mer ends whose
    key is kept). Zero for window and extend, which have no index."""
    P = starts.shape[0] - 1
    S = kind.shape[0]
    per = np.zeros((P, S), np.int64)
    one = np.uint64(1)
    for p in nb.prange(P):
        t = db[starts[p] : starts[p + 1]]
        reg = np.uint64(0)
        valid = 0
        for x in range(t.shape[0]):
            a = t[x]
            if a == 255:
                reg = np.uint64(0)
                valid = 0
                continue
            reg = (reg << one) | np.uint64(a)
            if valid < 64:
                valid += 1
            for s in range(S):
                kd = kind[s]
                need = span[s] if kd < 2 else (k[s] if kd == 2 else 1 << 30)
                if valid >= need and _kept(reg & pat[s], salt[s], scaled[s]):
                    per[p, s] += 1
    return per.sum(axis=0)


def run_pair(q: np.ndarray, t: np.ndarray, schemes: list[Scheme]) -> np.ndarray:
    a = scheme_arrays(schemes)
    return scan_pair(
        q,
        t,
        a["kind"],
        a["pat"],
        a["span"],
        a["k"],
        a["n"],
        a["win"],
        a["minm"],
        a["pen"],
        a["scaled"],
        a["salt"],
    )


def run_database(
    q: np.ndarray, db: np.ndarray, starts: np.ndarray, schemes: list[Scheme]
):
    a = scheme_arrays(schemes)
    return scan_database(
        q,
        db,
        starts,
        a["kind"],
        a["pat"],
        a["span"],
        a["k"],
        a["n"],
        a["win"],
        a["minm"],
        a["pen"],
        a["scaled"],
        a["salt"],
    )


def run_index_entries(
    db: np.ndarray, starts: np.ndarray, schemes: list[Scheme]
) -> np.ndarray:
    a = scheme_arrays(schemes)
    return index_entries(
        db, starts, a["kind"], a["pat"], a["span"], a["k"], a["scaled"], a["salt"]
    )


@nb.njit(cache=True, parallel=True)
def scan_alignments(
    qcat, tcat, starts, kind, pat, span, k, n, win, minm, pen, scaled, salt
):
    """Each alignment's own diagonal only (gap columns reset). Row a, scheme s: fires, or
    the best extension score. A pair is reached by a seed when its count is above zero.
    """
    A = starts.shape[0] - 1
    S = kind.shape[0]
    out = np.zeros((A, S), np.float64)
    for r in nb.prange(A):
        fires = np.zeros(S, np.int64)
        pieces = np.zeros(S, np.int64)
        best = np.zeros(S, np.float64)
        scan_diagonal(
            qcat,
            tcat,
            starts[r],
            starts[r],
            starts[r + 1] - starts[r],
            kind,
            pat,
            span,
            k,
            n,
            win,
            minm,
            pen,
            scaled,
            salt,
            fires,
            pieces,
            best,
        )
        for s in range(S):
            out[r, s] = best[s] if kind[s] == 4 else fires[s]
    return out


def run_aligned(
    qaln: list[np.ndarray], taln: list[np.ndarray], schemes: list[Scheme]
) -> np.ndarray:
    """``scan_alignments`` over aligned class arrays of equal length per pair."""
    a = scheme_arrays(schemes)
    starts = np.concatenate(([0], np.cumsum([len(x) for x in qaln]))).astype(np.int64)
    return scan_alignments(
        np.concatenate(qaln),
        np.concatenate(taln),
        starts,
        a["kind"],
        a["pat"],
        a["span"],
        a["k"],
        a["n"],
        a["win"],
        a["minm"],
        a["pen"],
        a["scaled"],
        a["salt"],
    )


def concat_database(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate(([0], np.cumsum([len(x) for x in seqs]))).astype(np.int64)
    return np.concatenate(seqs), starts


def fire_positions(
    q: np.ndarray, t: np.ndarray, diagonal: int, schemes: list[Scheme]
) -> np.ndarray:
    """Where each scheme fires along one diagonal (t_pos - q_pos = ``diagonal``).

    Row x is query position i0 + x. Found by scanning every prefix of the diagonal and
    taking differences, so it reuses the scan unchanged; fine for one diagonal.
    """
    a = scheme_arrays(schemes)
    i0, j0 = max(0, -diagonal), max(0, diagonal)
    length = min(len(q) - i0, len(t) - j0)
    S = len(schemes)
    prev = np.zeros(S, np.int64)
    out = np.zeros((length, S), bool)
    for x in range(length):
        fires = np.zeros(S, np.int64)
        scan_diagonal(
            q,
            t,
            i0,
            j0,
            x + 1,
            a["kind"],
            a["pat"],
            a["span"],
            a["k"],
            a["n"],
            a["win"],
            a["minm"],
            a["pen"],
            a["scaled"],
            a["salt"],
            fires,
            np.zeros(S, np.int64),
            np.zeros(S, np.float64),
        )
        out[x] = fires > prev
        prev = fires
    return out
