"""Low-complexity and transmembrane-helix masks for one protein sequence.

The same two rules as nextflow-runs/invertebrate-dark-set/tools/dark-regions (PR 57), so a
notebook marks a protein the way that tool scores its regions.

low_complexity_mask: the first pass of SEG (Wootton & Federhen 1993, Computers & Chemistry
17:149). A 12-residue window whose Shannon entropy is at most 2.2 bits marks its residues;
any 12-residue window of at most 2.5 bits that touches a marked residue is marked too. SEG's
trimming of each stretch to its lowest-complexity core is not done.

helix_mask: Kyte & Doolittle 1982 (J Mol Biol 157:105). A 19-residue window whose mean
hydropathy is at least 1.6 marks its residues as a likely membrane-spanning helix.
"""

import math

KYTE_DOOLITTLE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
SEG_WINDOW, SEG_LOCUT, SEG_HICUT = 12, 2.2, 2.5
TM_WINDOW, TM_CUT = 19, 1.6


def _window_entropies(seq: str, window: int) -> list[float]:
    """Shannon entropy in bits of every window of `window` residues, in order."""
    out = []
    for s in range(len(seq) - window + 1):
        counts: dict[str, int] = {}
        for a in seq[s : s + window]:
            counts[a] = counts.get(a, 0) + 1
        out.append(-sum(c / window * math.log2(c / window) for c in counts.values()))
    return out


def low_complexity_mask(seq: str) -> list[bool]:
    """True for each residue in a low-complexity stretch (SEG first pass)."""
    mask = [False] * len(seq)
    ents = _window_entropies(seq, SEG_WINDOW)
    for s, e in enumerate(ents):
        if e <= SEG_LOCUT:
            mask[s : s + SEG_WINDOW] = [True] * SEG_WINDOW
    changed = True
    while changed:
        changed = False
        for s, e in enumerate(ents):
            win = mask[s : s + SEG_WINDOW]
            if e <= SEG_HICUT and any(win) and not all(win):
                mask[s : s + SEG_WINDOW] = [True] * SEG_WINDOW
                changed = True
    return mask


def helix_mask(seq: str) -> list[bool]:
    """True for each residue in a 19-residue window of mean Kyte-Doolittle hydropathy >= 1.6."""
    mask = [False] * len(seq)
    vals = [KYTE_DOOLITTLE.get(a, 0.0) for a in seq]
    for s in range(len(seq) - TM_WINDOW + 1):
        if sum(vals[s : s + TM_WINDOW]) / TM_WINDOW >= TM_CUT:
            mask[s : s + TM_WINDOW] = [True] * TM_WINDOW
    return mask


def stretches(mask: list[bool]) -> list[tuple[int, int]]:
    """(start, end) of each run of True, 0-based and end-exclusive."""
    out, start = [], None
    for i, v in enumerate(mask + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out
