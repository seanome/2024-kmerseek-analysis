"""Shuffled copies of protein sequences, used as decoy queries.

A decoy query is a real query with its residues reordered. A hit to a decoy is a chance
hit, so counting decoy hits measures how many chance hits a search lets through.

Notebooks and scripts import this file by adding its folder to `sys.path`. A Nextflow
process stages it as a `path` input and puts the task folder on `PYTHONPATH`, as the QfO
Pfam region benchmark does for flow_diagram.py:

    input:
    path decoys_module, stageAs: 'decoys.py'
    script:
    '''
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    '''

Staging alone is not enough. Python looks for imports next to the real file behind a
staged script, which is in bin/, not in the task folder.
"""

import random
from collections import defaultdict


def dipeptide_shuffle(seq: str, rng: random.Random) -> str:
    """A random sequence with the same first residue, last residue and count of every
    adjacent residue pair as `seq` (Altschul & Erickson 1985, Mol Biol Evol 2:526).

    The pairs are edges of a graph on residues; `seq` is one walk that uses every edge
    once. Pick, for every residue except the last one, which of its outgoing edges is
    used last, such that those edges lead to the last residue from everywhere (else
    the walk strands). Shuffle the other edges, put the chosen one at the end, walk.

    The code is unchanged from notebook 256's copy, so a decoy made with the same seed
    is the same before and after a branch switches to this file.
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


def composition_shuffle(seq: str, rng: random.Random) -> str:
    """The residues of `seq` in a random order. Keeps the count of each amino acid and
    nothing else, so hydrophobic runs and low-complexity stretches are broken up. A
    weaker null than `dipeptide_shuffle`: it lets through fewer chance matches."""
    residues = list(seq)
    rng.shuffle(residues)
    return "".join(residues)


def window_shuffle(seq: str, rng: random.Random, window: int = 10) -> str:
    """`seq` with its residues shuffled inside each non-overlapping `window`-residue
    stretch, counted from the first residue (the last stretch may be shorter).

    Each stretch keeps its residues, so its mean hydrophobicity stays where it was, and a
    low-complexity run longer than `window` stays low-complexity. A stronger null than
    `dipeptide_shuffle`. The code is the `window_shuffle` of
    nextflow-runs/regions-index-pilot/bin/make_window_decoys.py (PR 114), which notebook
    290 (PR 134) also copied.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    out = []
    for i in range(0, len(seq), window):
        w = list(seq[i : i + window])
        rng.shuffle(w)
        out.extend(w)
    return "".join(out)
