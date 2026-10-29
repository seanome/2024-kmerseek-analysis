"""Reduced amino-acid alphabets, copied from kmerseek so both sides encode the same way.

Source: seanome/kmerseek, src/rust/alphabets.rs at commit 05b4131 (branch
olgabot/cys-runs-040-ka-bins, the build in image
olgabot/kmerseek:2026-10-17-cys-frameworks-05b4131). dayhoff6 and hp_lehninger2 are encoded
by sourmash inside kmerseek; their tables here are checked against sourmash's hashes in
tests/test_stage_diagnostic.py, so a wrong letter fails the test.

Only the 20 canonical residues are mapped. The panels drop any sequence with another
letter (B, J, O, U, X, Z), so no ambiguity rule is needed.
"""

from __future__ import annotations

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

# class lists, one string of residues per class, in kmerseek's order
CLUSTERS: dict[str, list[str]] = {
    "protein20": list(CANONICAL),
    # sourmash aa_to_dayhoff: a=C, b=AGPST, c=DENQ, d=HKR, e=ILMV, f=FWY
    "dayhoff6": ["C", "AGPST", "DENQ", "HKR", "ILMV", "FWY"],
    # alphabets.rs LEHNINGER_HP: h = AFGILMPVWY, p = CDEHKNQRST
    "hp_lehninger2": ["AFGILMPVWY", "CDEHKNQRST"],
    # alphabets.rs CYS2: c = C, x = the other 19
    "cys2": ["C", "ADEFGHIKLMNPQRSTVWY"],
    # alphabets.rs SDM12_CLUSTERS
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    # alphabets.rs GBMR4_CLUSTERS and FUNCGROUPS8_CLUSTERS (not used in the v1 settings)
    "gbmr4": ["ADKERNTSQ", "YFLIVMCWH", "G", "P"],
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
}

# letters kmerseek writes for each class. Only the sourmash-encoded ones matter for hashing
# (they are hashed as these letters); table-backed alphabets are written with their own
# letters by kmerseek, and for those we hash the class index letter below, which gives the
# same k-mer *sets* (a relabelling), which is all the seed counts need.
SOURMASH_LETTERS = {
    "dayhoff6": "abcdef",
    "hp_lehninger2": "hp",
}


def class_letters(alphabet: str) -> str:
    """One letter per class. Uppercase A.. so Biopython's aligner can use them."""
    n = len(CLUSTERS[alphabet])
    return CANONICAL[:n] if alphabet == "protein20" else "ABCDEFGHIJKLMNOPQRST"[:n]


def residue_map(alphabet: str) -> dict[str, str]:
    """Residue -> class letter. protein20 maps every residue to itself."""
    if alphabet == "protein20":
        return {r: r for r in CANONICAL}
    letters = class_letters(alphabet)
    m = {}
    for letter, members in zip(letters, CLUSTERS[alphabet]):
        for r in members:
            m[r] = letter
    if sorted(m) != sorted(CANONICAL):
        raise ValueError(f"{alphabet} does not cover the 20 residues exactly once")
    return m


def encode(seq: str, alphabet: str) -> str:
    m = residue_map(alphabet)
    return "".join(m[r] for r in seq)


def sourmash_encode(seq: str, alphabet: str) -> str:
    """The string sourmash would hash for this alphabet (dayhoff6 and HP only)."""
    letters = SOURMASH_LETTERS[alphabet]
    m = {}
    for letter, members in zip(letters, CLUSTERS[alphabet]):
        for r in members:
            m[r] = letter
    return "".join(m[r] for r in seq)
