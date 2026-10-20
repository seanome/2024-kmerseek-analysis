"""Residue classes of the three candidate alphabets of notebook 353.

Copied from kmerseek at commit 0a229700 (the build in image
olgabot/kmerseek:2026-10-18-nb344-0a22970):
  sdm12, mmseqs12: src/rust/alphabets.rs, SDM12_CLUSTERS (line 867) and MMSEQS12_CLUSTERS (877).
  dayhoff6: sourmash's Dayhoff encoding, which kmerseek uses for dayhoff6 (alphabets.rs,
  uses_sourmash_encoder); the same table as notebooks/ortholog_analysis_utils.py
  ALPHABET_AA_GROUPS["dayhoff"].
Standard library only, so it runs inside the kmerseek image.
"""

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

CLASSES: dict[str, list[str]] = {
    "dayhoff6": ["C", "AGPST", "DENQ", "HKR", "ILMV", "FWY"],
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
}

# Residues with positive Kyte-Doolittle hydropathy (Kyte and Doolittle 1982): the
# hydrophobic class of kmerseek's hp_kyte_doolittle2.
HYDROPHOBIC = set("ACFILMV")


def class_of(alphabet: str) -> dict[str, str]:
    """Residue -> the string of every residue in its class."""
    out = {}
    for cls in CLASSES[alphabet]:
        for aa in cls:
            out[aa] = cls
    assert sorted(out) == sorted(CANONICAL), alphabet
    return out


def encode(seq: str, alphabet: str) -> str:
    """One letter per class (a, b, c, ... in CLASSES order); '.' for a non-canonical residue."""
    letters = {}
    for i, cls in enumerate(CLASSES[alphabet]):
        for aa in cls:
            letters[aa] = chr(ord("a") + i)
    return "".join(letters.get(aa, ".") for aa in seq)
