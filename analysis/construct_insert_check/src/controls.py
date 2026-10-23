"""Controls: natural stretches matched to insert composition, and shuffled inserts.

These say how often the reduced-alphabet search matches something that is not an insert.
``fit_l_distinguish`` reads only the development collection, because PLAN.md freezes that
threshold on development data.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from dictionary import DictionaryEntry
from records import Collection
from workflows import REDUCED_SETTINGS, _diagonal_runs, encode


def uniform_shuffle(seq: str, rng: random.Random) -> str:
    letters = list(seq)
    rng.shuffle(letters)
    return "".join(letters)


def neighbour_preserving_shuffle(seq: str, rng: random.Random) -> str:
    """Shuffle keeping the counts of neighbouring residue pairs, where possible.

    Walks the graph whose edges are the adjacent pairs of the original sequence, the
    residue-level version of a doublet-preserving shuffle. Falls back to a uniform
    shuffle when the walk cannot use every edge, so the returned sequence always has the
    original composition.
    """
    if len(seq) < 3:
        return seq
    edges: dict[str, list[str]] = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    for options in edges.values():
        rng.shuffle(options)
    walk = [seq[0]]
    while edges[walk[-1]]:
        walk.append(edges[walk[-1]].pop())
    if len(walk) != len(seq) or Counter(walk) != Counter(seq):
        return uniform_shuffle(seq, rng)
    return "".join(walk)


def composition_matched_stretch(
    insert_seq: str, source: str, rng: random.Random
) -> str:
    """A natural stretch of the same length, chosen to match the insert's composition.

    Draws candidate windows from ``source`` and keeps the one whose residue composition is
    closest to the insert's, measured as total absolute difference in residue fractions.
    """
    length = len(insert_seq)
    if len(source) <= length:
        return source
    target = Counter(insert_seq)
    best = (float("inf"), source[:length])
    positions = range(0, len(source) - length + 1)
    sample = rng.sample(list(positions), min(200, len(positions)))
    for start in sample:
        window = source[start : start + length]
        counts = Counter(window)
        distance = sum(
            abs(counts[r] - target[r]) for r in set(counts) | set(target)
        ) / length
        if distance < best[0]:
            best = (distance, window)
    return best[1]


def count_runs(query: str, inserts: list[DictionaryEntry]) -> list[int]:
    """Lengths of every reduced-alphabet run between ``query`` and the dictionary."""
    lengths = []
    for alphabet, k in REDUCED_SETTINGS:
        query_codes = encode(query, alphabet)
        for entry in inserts:
            insert_codes = encode(entry.insert_seq, alphabet)
            for _, _, length in _diagonal_runs(query_codes, insert_codes, k):
                lengths.append(length)
    return lengths


def fit_l_distinguish(
    development: Collection, seed: int = 0
) -> tuple[int | None, dict[str, float]]:
    """Shortest run length where composition-matched natural stretches give under one
    chance match per construct, in the development collection only."""
    rng = random.Random(seed)
    inserts = [
        DictionaryEntry(d.insert_name, d.insert_seq, d.category, "development")
        for d in development.dictionary
    ]
    if not inserts or not development.constructs:
        return None, {}
    lengths: list[int] = []
    for construct in development.constructs:
        for entry in inserts:
            stretch = composition_matched_stretch(
                entry.insert_seq, construct.intended_protein_seq, rng
            )
            lengths.extend(count_runs(stretch, inserts))
    per_construct = np.array(lengths, dtype=int)
    constructs = len(development.constructs)
    shortest = None
    if per_construct.size:
        for candidate in range(int(per_construct.min()), int(per_construct.max()) + 2):
            rate = float((per_construct >= candidate).sum()) / constructs
            if rate < 1.0:
                shortest = candidate
                break
    else:
        shortest = min(k for _, k in REDUCED_SETTINGS)
    summary = {
        "natural_stretch_runs": float(per_construct.size),
        "development_constructs": float(constructs),
        "median_run_length": float(np.median(per_construct)) if per_construct.size else 0.0,
        "max_run_length": float(per_construct.max()) if per_construct.size else 0.0,
    }
    return shortest, summary


def shuffled_insert_runs(
    collection: Collection, seed: int = 0
) -> dict[str, dict[str, float]]:
    """How often shuffled inserts still match the dictionary, by shuffle kind."""
    rng = random.Random(seed)
    inserts = [
        DictionaryEntry(d.insert_name, d.insert_seq, d.category, "development")
        for d in collection.dictionary
    ]
    out: dict[str, dict[str, float]] = {}
    for kind, shuffle in (
        ("uniform shuffle", uniform_shuffle),
        ("neighbour-preserving shuffle", neighbour_preserving_shuffle),
    ):
        matched = 0
        total = 0
        run_lengths: list[int] = []
        for entry in inserts:
            total += 1
            shuffled = shuffle(entry.insert_seq, rng)
            lengths = count_runs(shuffled, inserts)
            if lengths:
                matched += 1
                run_lengths.extend(lengths)
        out[kind] = {
            "inserts": float(total),
            "still_matching": float(matched),
            "fraction_matching": matched / total if total else 0.0,
            "median_run_length": float(np.median(run_lengths)) if run_lengths else 0.0,
        }
    return out
