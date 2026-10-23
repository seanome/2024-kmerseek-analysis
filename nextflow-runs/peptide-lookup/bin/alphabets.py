"""Residue classes of every kmerseek alphabet, plus the mass-spec alphabet of notebook 300.

Copied from kmerseek `src/rust/alphabets.rs` at seanome/kmerseek main 00395d7 (2026-10-20).
`protein20` keeps all 20 residues. `dayhoff6` is sourmash's Dayhoff table
(`sourmash/src/core/src/encodings.rs`). `tests/test_alphabets.py` checks the cluster lists
against a kmerseek checkout when one is found.

`massspec18` is new: mass spectrometry cannot tell I from L (same mass) and often not Q from
K (0.036 Da apart), so those two pairs merge and the other 16 residues stay apart, 18 classes.
"""

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def _hp(hydrophobic: str, polar: str) -> list[str]:
    """An HP alphabet as two clusters."""
    return [hydrophobic, polar]


CLUSTERS: dict[str, list[str]] = {
    "protein20": list(CANONICAL),
    "dayhoff6": ["C", "AGPST", "DENQ", "HKR", "ILMV", "FWY"],
    "hp_lehninger2": _hp("AFGILMPVWY", "CDEHKNQRST"),
    "hp_thomas_dill2": _hp("ACFILMVWY", "DEGHKNPQRST"),
    "hp_kyte_doolittle2": _hp("ACFILMV", "DEGHKNPQRSTWY"),
    "hp_thomas_dill_no_c2": _hp("AFILMVWY", "CDEGHKNPQRST"),
    "hp_lehninger_c_nonpolar2": _hp("ACFGILMPVWY", "DEHKNQRST"),
    "hp_lehninger_hpc3": ["AFGILMPVWY", "DEHKNQRST", "C"],
    "hp_pbotc_1st_ed2": _hp("ACFILMPVWY", "DEGHKNQRST"),
    "gbmr4": ["ADKERNTSQ", "YFLIVMCWH", "G", "P"],
    "polarity4": ["GAVLIFWMP", "STCYNQ", "DE", "HKR"],
    "wwmj5": ["CMFILVWY", "ATH", "GP", "DE", "SNQRK"],
    "gbmr7": ["DN", "AEFIKLMQRVWY", "CH", "T", "S", "G", "P"],
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    "wass14": [
        "WM",
        "DI",
        "P",
        "C",
        "AV",
        "K",
        "T",
        "RE",
        "G",
        "L",
        "Y",
        "SH",
        "F",
        "NQ",
    ],
    "hsdm17": [
        "A",
        "D",
        "KE",
        "R",
        "N",
        "T",
        "S",
        "Q",
        "Y",
        "F",
        "LIV",
        "M",
        "C",
        "W",
        "H",
        "G",
        "P",
    ],
    "uniprot18": [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "EP",
        "G",
        "HL",
        "I",
        "K",
        "M",
        "F",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ],
    "massspec18": [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "IL",
        "QK",
        "M",
        "N",
        "P",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ],
}

# Alphabets kmerseek can index with at main 00395d7. massspec18 is added by
# seanome/kmerseek PR 156; no index or fit has been built with it yet.
KMERSEEK_ALPHABETS = [a for a in CLUSTERS if a != "massspec18"]


def class_of(alphabet: str) -> dict[str, int]:
    """Residue -> class index for one alphabet."""
    return {aa: i for i, cluster in enumerate(CLUSTERS[alphabet]) for aa in cluster}


def check_partition(alphabet: str) -> None:
    """Raise if an alphabet does not cover each canonical residue exactly once."""
    letters = "".join(CLUSTERS[alphabet])
    if sorted(letters) != sorted(CANONICAL):
        raise ValueError(f"{alphabet} does not partition the 20 residues: {letters}")


for _name in CLUSTERS:
    check_partition(_name)
