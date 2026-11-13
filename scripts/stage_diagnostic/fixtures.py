"""Small deterministic fixtures that check the implementation, not biology.

One base protein (BASE, 72 residues, two conserved blocks around a linker) and seven
references, each built to make one stage behave in a known way. `EXPECTED` says what each
stage must do with each; evaluate.check_fixtures compares and reports pass/fail per cell.

The sequences depend only on the seed below, so the files are identical on every run.
"""

from __future__ import annotations

import random

from alphabets import CANONICAL, CLUSTERS
from seeds import hashes

SEED = 350


def _rand(n: int, rng: random.Random, letters: str = CANONICAL) -> str:
    return "".join(rng.choice(letters) for _ in range(n))


def build(alphabet: str, k: int, scaled: int, mask: bool) -> tuple[dict, dict]:
    """(queries, references) as {name: sequence} for one kmerseek setting."""
    rng = random.Random(SEED)
    block_a, linker, block_b = _rand(30, rng), _rand(12, rng), _rand(30, rng)
    base = block_a + linker + block_b

    # one substitution in the middle of block A, to a residue of another class
    i = 15
    sub_to = next(r for r in CANONICAL if not _same_class(r, base[i], alphabet))
    substitution = base[:i] + sub_to + base[i + 1 :]

    insertion = block_a + linker + _rand(8, rng) + block_b

    # identical reduced encoding, different residues: every residue swapped for another
    # member of its class where the class has one
    twin = "".join(_class_mate(r, alphabet) for r in base)

    low_complexity = ("AAT" * 24)[: len(base)]
    short = base[: max(1, k - 1)]

    # a sequence of k+3 residues none of whose k-mers survive the sketch at this scaled
    empty = None
    for _ in range(200_000):
        s = _rand(k + 3, rng)
        if hashes(s, alphabet, k, mask, scaled=1) and not hashes(s, alphabet, k, mask, scaled):
            empty = s
            break
    if empty is None:
        raise RuntimeError("no empty-sketch fixture found")

    queries = {"fx_base": base, "fx_short_query": short, "fx_empty_sketch_query": empty}
    refs = {
        "fx_exact_duplicate": base,
        "fx_substitution": substitution,
        "fx_insertion": insertion,
        "fx_reduced_twin": twin,
        "fx_low_complexity": low_complexity,
        "fx_short": short,
        "fx_empty_sketch": empty,
    }
    return queries, refs


def _class_of(r: str, alphabet: str) -> str:
    return next(c for c in CLUSTERS[alphabet] if r in c)


def _same_class(a: str, b: str, alphabet: str) -> bool:
    return _class_of(a, alphabet) == _class_of(b, alphabet)


def _class_mate(r: str, alphabet: str) -> str:
    c = _class_of(r, alphabet)
    return c[(c.index(r) + 1) % len(c)]


# What each stage must do, query fx_base unless named. "hit" = among the candidates /
# scores > 0; "top" = ranked first; "miss" = not a candidate.
# The twin scores exactly as the duplicate under B (same encoding) but lower under A.
EXPECTED = [
    ("fx_base", "fx_exact_duplicate", "A", "top"),
    ("fx_base", "fx_exact_duplicate", "C", "hit"),
    ("fx_base", "fx_substitution", "C", "hit"),
    ("fx_base", "fx_insertion", "A", "hit"),
    ("fx_base", "fx_insertion", "C", "hit"),
    ("fx_base", "fx_reduced_twin", "B", "ties_duplicate"),
    ("fx_base", "fx_reduced_twin", "A", "below_duplicate"),
    ("fx_base", "fx_reduced_twin", "C", "hit"),
    ("fx_base", "fx_short", "C", "miss"),
    ("fx_short_query", "fx_exact_duplicate", "C", "miss"),
    ("fx_empty_sketch_query", "fx_empty_sketch", "C", "hit"),
    ("fx_empty_sketch_query", "fx_empty_sketch", "D", "miss"),
]
