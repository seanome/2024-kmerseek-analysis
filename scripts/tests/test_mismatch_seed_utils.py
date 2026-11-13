"""The numba seed scan against a plain-Python reading of each rule, on random H/P strings.

Run with numba available, for example:
    pixi exec --spec numba --spec numpy --spec polars --spec pytest -- pytest scripts/tests/test_mismatch_seed_utils.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import mismatch_seed_utils as ms  # noqa: E402

M64 = 2**64 - 1


def mix(x: int) -> int:
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def key(qcls: list[int], end: int, pattern: str) -> int:
    """Query classes at the pattern's 1-positions, packed with the newest position as bit 0."""
    out = 0
    span = len(pattern)
    for p, c in enumerate(pattern):
        if c == "1":
            out |= qcls[end - (span - 1 - p)] << (span - 1 - p)
    return out


def kept(k: int, salt: int, scaled: int) -> bool:
    return scaled == 1 or mix(k ^ salt) % scaled == 0


def reference(q: list[int], t: list[int], s: ms.Scheme, salt: int) -> float:
    """Fires (or best extension score) along the diagonal q[x] vs t[x]; 255 resets."""
    fires, best, cur = 0, 0.0, 0.0
    seg = 0  # first index of the current reset-free stretch
    hits: list[int] = []  # chained: ends of kept pieces in this stretch
    for x in range(len(q)):
        if q[x] == 255 or t[x] == 255:
            seg, cur, hits = x + 1, 0.0, []
            continue
        m = [q[i] == t[i] for i in range(seg, x + 1)]  # match string of the stretch so far
        if s.family == "extend":
            cur = max(0.0, cur + (1.0 if m[-1] else -s.penalty))
            best = max(best, cur)
        elif s.family in ("exact", "spaced"):
            L = len(s.pattern)
            if len(m) >= L and all(mm for mm, c in zip(m[-L:], s.pattern) if c == "1") \
                    and kept(key(q, x, s.pattern), salt, s.scaled):
                fires += 1
        elif s.family == "chained":
            if len(m) >= s.k and all(m[-s.k:]) and kept(key(q, x, "1" * s.k), salt, s.scaled):
                hits.append(x)
                cnt, limit = 0, x  # newest first; a piece may end at or before limit
                for e in reversed(hits):
                    if e <= limit and e - s.k + 1 >= x - s.window + 1:
                        cnt += 1
                        limit = e - s.k
                if cnt >= s.n:
                    fires += 1
        elif s.family == "window":
            if m[-1] and len(m) >= s.window and sum(m[-s.window:]) >= s.min_match:
                fires += 1
    return best if s.family == "extend" else float(fires)


@pytest.mark.parametrize("seed", range(4))
def test_scan_matches_reference_on_one_diagonal(seed):
    rng = np.random.default_rng(seed)
    n = 400
    q = rng.integers(0, 2, n).astype(np.uint8)
    t = q.copy()
    flip = rng.random(n) < 0.3  # a related pair: 70% of positions keep the class
    t[flip] ^= 1
    t[rng.integers(0, n, 3)] = 255  # three resets
    schemes = ms.default_schemes()
    got = ms.run_aligned([q], [t], schemes)[0]
    salts = ms.scheme_arrays(schemes)["salt"]
    want = [reference(q.tolist(), t.tolist(), s, int(salts[i])) for i, s in enumerate(schemes)]
    assert got.tolist() == want


def test_scan_pair_diagonal_layout():
    """Row d of scan_pair is the diagonal t_pos - q_pos = d - (len(q) - 1)."""
    q = ms.encode("FATVVEELFRDGVNWGRIVAFFEFGGVM", "hp_lehninger2")
    t = ms.encode("WWWWW" + "FATVVEELFRDGVNWGRIVAFFEFGGVM", "hp_lehninger2")
    s = [ms.Scheme("exact", pattern="1" * 28)]
    out = ms.run_pair(q, t, s)
    assert out.shape == (len(q) + len(t) - 1, 1)
    assert np.flatnonzero(out[:, 0]).tolist() == [5 + len(q) - 1]


def test_scaled_two_keeps_a_subset_of_about_half():
    rng = np.random.default_rng(0)
    db = [rng.integers(0, 2, 5_000).astype(np.uint8) for _ in range(4)]
    cat, starts = ms.concat_database(db)
    schemes = [ms.Scheme("exact", pattern="1" * 12), ms.Scheme("exact", pattern="1" * 12, scaled=2)]
    full, half = ms.run_index_entries(cat, starts, schemes)
    assert full == 4 * (5_000 - 11)
    assert 0.45 < half / full < 0.55
