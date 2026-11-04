"""Find which protein each probe lands on, with the rule that fits the platform's probe length.

Thresholds are in sketch.PLATFORM_RULES. A probe lands on a protein through protein k-mers, so this
needs no gene annotation. All probes of one length are translated at once, which keeps a platform
with millions of 25-letter probes affordable.
"""

from collections import defaultdict

import numpy as np

from geo2kmers.geo import Geo2KmersError
from geo2kmers.sketch import (
    AMINO,
    AMINO_ACIDS,
    PLATFORM_RULES,
    kmer_numbers,
    letter_codes,
    platform_rule,
    read_fasta,
)

BASE = np.full(256, 4, dtype=np.uint8)
BASE[[ord(c) for c in "TCAG"]] = [0, 1, 2, 3]
STOP, UNKNOWN = 20, 21
CODON_CODE = np.array([AMINO_ACIDS.find(a) if a != "*" else STOP for a in AMINO] + [UNKNOWN], dtype=np.uint8)
COMPLEMENT = np.array([2, 3, 0, 1, 4], dtype=np.uint8)  # T<->A, C<->G


def proteome_owners(paths, k):
    """Return (sorted k-mer numbers, each number's proteins)."""
    table = letter_codes("protein")
    owners = defaultdict(set)
    for path in paths:
        for name, sequence in read_fasta(path).items():
            for number in np.unique(kmer_numbers(sequence, k, "protein", table)):
                owners[int(number)].add(name)
    if not owners:
        raise Geo2KmersError(f"no k-mer of k = {k} in {', '.join(map(str, paths))}")
    return np.array(sorted(owners), dtype=np.uint64), owners


def frames_of(sequences):
    """Yield (frame, n by residues array) for six frames of equal-length sequences, numbered as translate_frames."""
    bases = BASE[np.frombuffer("".join(sequences).encode(), dtype=np.uint8)].reshape(len(sequences), -1)
    for strand_number, strand in enumerate((bases, COMPLEMENT[bases[:, ::-1]])):
        for start in range(3):
            n = (strand.shape[1] - start) // 3
            a, b, c = (strand[:, start + i : start + 3 * n : 3].astype(np.int16) for i in range(3))
            unknown = (a == 4) | (b == 4) | (c == 4)
            codes = np.where(unknown, 64, 16 * a + 4 * b + c)
            yield strand_number * 3 + start, CODON_CODE[codes]


def window_numbers(residues, k):
    """Each window's k-mer number, and whether the window holds only amino acids."""
    windows = np.lib.stride_tricks.sliding_window_view(residues, k, axis=1)
    powers = np.array([20**i for i in range(k - 1, -1, -1)], dtype=np.uint64)
    return windows.astype(np.uint64) @ powers, (windows < STOP).all(axis=2)


def land_probes(probes, proteome, k=None):
    """Return (rule name, k, landing rows, count of probes with a k-mer in the proteome).

    A landing row is (probe, protein, frame, k-mers on that protein, usable k-mers in that frame, their numbers).
    On a short oligo only the longest stop-free frame gives rows, with ties going to the lower frame.
    """
    median = float(np.median([len(s) for s in probes.values()]))
    rule = platform_rule(median)
    k = k or PLATFORM_RULES[rule]["k"]
    known, owners = proteome_owners(proteome, k)
    rows, touched = [], set()
    by_length = defaultdict(list)
    for name, sequence in probes.items():
        by_length[len(sequence)].append(name)
    for length, names in sorted(by_length.items()):
        if length // 3 < k:
            continue
        frames = list(frames_of([probes[n] for n in names]))
        chosen = np.full(len(names), -1)
        if rule == "short_oligo":
            best_length = np.zeros(len(names), dtype=int)
            for frame, residues in frames:
                better = (residues != STOP).all(axis=1) & (residues.shape[1] > best_length)
                chosen[better], best_length[better] = frame, residues.shape[1]
        for frame, residues in frames:
            if residues.shape[1] < k:
                continue
            values, usable = window_numbers(residues, k)
            hits = np.isin(values, known) & usable
            for row in np.flatnonzero(hits.any(axis=1)):
                touched.add(names[row])
                if rule == "short_oligo" and frame != chosen[row]:
                    continue
                on_protein = defaultdict(list)
                for value in values[row][hits[row]]:
                    for protein in owners[int(value)]:
                        on_protein[protein].append(int(value))
                rows.extend(
                    (names[row], protein, frame, len(numbers), int(usable[row].sum()), frozenset(numbers))
                    for protein, numbers in on_protein.items()
                )
    return rule, k, rows, len(touched)


def passing_landings(rule, rows, probe_sets_of=None):
    """Keep the landings the platform's rule accepts.

    Returns {probe or probe set: {protein: how many different k-mers support the landing}}. A probe can
    belong to several probe sets, so probe_sets_of returns a list.
    """
    settings = PLATFORM_RULES[rule]
    passed = defaultdict(dict)
    if rule == "probe_set":
        probes_on, kmers_on = defaultdict(set), defaultdict(set)
        for probe, protein, _, _, _, numbers in rows:
            for probe_set in probe_sets_of(probe):
                probes_on[probe_set, protein].add(probe)
                kmers_on[probe_set, protein] |= numbers
        for (probe_set, protein), members in probes_on.items():
            if len(members) >= settings["min_probes"]:
                passed[probe_set][protein] = len(kmers_on[probe_set, protein])
        return passed
    for probe, protein, _, on_protein, usable, numbers in rows:
        if on_protein / usable >= settings["min_share"]:
            passed[probe][protein] = len(numbers)
    return passed
