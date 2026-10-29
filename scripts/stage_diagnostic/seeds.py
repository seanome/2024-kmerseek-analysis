"""Seed k-mers as kmerseek sees them: how many a sequence has, and how many a sketch keeps.

kmerseek (src/rust/sketch.rs, add_protein, at 05b4131) hashes each window of k residues:
  * protein20, dayhoff6, hp_lehninger2: sourmash's own protein / Dayhoff / HP hash.
  * every table alphabet (cys2, sdm12, ...): the window written in the table's class
    letters (the first residue of each class, lowercase), hashed as a protein k-mer.
  * with --remove-low-complexity: a window is dropped when its residues are one letter
    repeated, or (any reduced alphabet) when its encoded letters are.
A sketch at scaled s keeps the hashes sourmash keeps at scaled s (FracMinHash).

The counts from here are checked against kmerseek's own n_intersecting_hashes on the
exact-duplicate fixture (see `evaluate.check_fixtures`).
"""

from __future__ import annotations

from functools import lru_cache

import sourmash

from alphabets import CLUSTERS, encode

SOURMASH_KIND = {"protein20": "is_protein", "dayhoff6": "dayhoff", "hp_lehninger2": "hp"}


@lru_cache(maxsize=None)
def _kmerseek_table(alphabet: str) -> dict[str, str]:
    """The letters kmerseek hashes. Cluster alphabets (build_reduced in alphabets.rs) use
    each cluster's first residue, lowercase. cys2 is written by hand there: c and x. The
    HP tables (build_hp) use h and p. A wrong letter keeps every count at scaled 1 the same
    (it is a relabelling) but changes which hashes a scaled sketch keeps."""
    if alphabet == "cys2":
        return {r: ("c" if r == "C" else "x") for r in CLUSTERS["protein20"]}
    if alphabet.startswith("hp_"):
        h, p = CLUSTERS[alphabet]
        return {**{r: "h" for r in h}, **{r: "p" for r in p}}
    m = {}
    for members in CLUSTERS[alphabet]:
        for r in members:
            m[r] = members[0].lower()
    return m


def windows(seq: str, k: int) -> list[str]:
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def _homopolymer(s: str) -> bool:
    return len(s) > 0 and s.upper() == s[0].upper() * len(s)


def kept_windows(seq: str, alphabet: str, k: int, mask: bool) -> list[str]:
    """Raw-residue windows kmerseek hashes (after the low-complexity mask, if on)."""
    out = []
    for w in windows(seq, k):
        if mask:
            if _homopolymer(w):
                continue
            if alphabet != "protein20" and _homopolymer(encode(w, alphabet)):
                continue
        out.append(w)
    return out


def reduced_kmers(seq: str, alphabet: str, k: int, mask: bool = False) -> set[str]:
    """Distinct encoded k-mers: the seed features available before any sketching."""
    return {encode(w, alphabet) for w in kept_windows(seq, alphabet, k, mask)}


def hashes(seq: str, alphabet: str, k: int, mask: bool = False, scaled: int = 1) -> set[int]:
    """The hashes a kmerseek sketch of `seq` holds at this scaled value."""
    kind = SOURMASH_KIND.get(alphabet, "is_protein")
    mh = sourmash.MinHash(n=0, ksize=k, scaled=scaled, **{kind: True})
    table = None if alphabet in SOURMASH_KIND else _kmerseek_table(alphabet)
    for w in kept_windows(seq, alphabet, k, mask):
        mh.add_protein("".join(table[r] for r in w) if table else w)
    return set(mh.hashes)
