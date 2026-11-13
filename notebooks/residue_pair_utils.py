"""Print a kmerseek region as residues: both sequences, a match line, and the alphabet's
encoding underneath.

kmerseek regions are ungapped: query residues region_start..region_end line up one to one
with target residues from target_start. kmerseek reports 0-based, end-exclusive
coordinates; the printed coordinates are 1-based and inclusive, as UniProt writes them.
"""

from __future__ import annotations

from hp_conservation_utils import ALPHABET_CLUSTERS as _SHARED

# The four alphabets of notebook 241's 19 that hp_conservation_utils does not list, copied
# from kmerseek 982a055, src/rust/alphabets.rs (*_CLUSTERS). On 2026-10-13 the six shared
# lists that file also has (gbmr4, gbmr7, polarity4, sdm12, uniprot18, wwmj5) matched it;
# the HP alphabets, dayhoff6 and protein20 are defined elsewhere and were not compared.
ALPHABET_CLUSTERS: dict[str, list[str]] = _SHARED | {
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
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
}


def encode(seq: str, alphabet: str) -> str:
    """Each residue as the first residue of its class in `alphabet` (protein20 unchanged)."""
    rep = {aa: cls[0] for cls in ALPHABET_CLUSTERS[alphabet] for aa in cls}
    return "".join(rep.get(aa, "X") for aa in seq)


def match_line(a: str, b: str) -> str:
    return "".join("|" if x == y else " " for x, y in zip(a, b))


def ungapped_pair_text(
    q_name: str,
    q_seq: str,
    q_start: int,
    t_name: str,
    t_seq: str,
    t_start: int,
    length: int,
    alphabet: str | None = None,
) -> str:
    """The region q_seq[q_start:q_start+length] against t_seq[t_start:t_start+length]
    (0-based starts), as text: residues, identity match line, and, for a reduced
    alphabet, the encoded strings with their own match line.
    """
    q = q_seq[q_start : q_start + length]
    t = t_seq[t_start : t_start + length]
    n_id = sum(x == y for x, y in zip(q, t))
    w = max(len(q_name), len(t_name))
    qc = f"{q_start + 1}-{q_start + len(q)}"
    tc = f"{t_start + 1}-{t_start + len(t)}"
    cw = max(len(qc), len(tc))
    pad = " " * (w + cw + 2)
    lines = [
        f"{q_name:>{w}} {qc:>{cw}} {q}",
        f"{pad}{match_line(q, t)}",
        f"{t_name:>{w}} {tc:>{cw}} {t}",
        f"{n_id} of {len(q)} residues identical",
    ]
    if alphabet and alphabet != "protein20":
        qe, te = encode(q, alphabet), encode(t, alphabet)
        n_cls = sum(x == y for x, y in zip(qe, te))
        cls = " ".join(c for c in ALPHABET_CLUSTERS[alphabet])
        lines += [
            f"{alphabet} classes: {cls} (each written as its first residue)",
            f"{q_name:>{w}} {'':>{cw}} {qe}",
            f"{pad}{match_line(qe, te)}",
            f"{t_name:>{w}} {'':>{cw}} {te}",
            f"{n_cls} of {len(q)} positions in the same {alphabet} class",
        ]
    return "\n".join(lines)
