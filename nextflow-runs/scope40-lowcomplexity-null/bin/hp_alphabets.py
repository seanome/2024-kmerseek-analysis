#!/usr/bin/env python3
"""
HP alphabet tables and compositional-complexity helpers.

The tables are transcribed verbatim from kmerseek's Rust source
(`src/rust/hp_alphabets.rs`, kmerseek 0.3.1) so that the Python side of this
analysis encodes sequences identically to the search itself. This is verified
at pipeline runtime by `verify_encoding` in hit_complexity.py, which re-encodes
kmerseek's own `query_subseq` output and asserts it reproduces `moltype_seq`.

Complexity convention
---------------------
For a two-letter (HP) k-mer with minority count m = min(n_h, n_p), the number of
distinct k-mers sharing that composition is the binomial coefficient C(k, m),
and the Wootton & Federhen (1993) compositional complexity, specialised from
the multinomial to the binomial, is

    K(k, m) = (1/k) * log2 C(k, m)        [bits per residue]

which converges to the binary Shannon entropy H(m/k) as k grows. A complexity
cutoff K* is therefore equivalent to a minority-count floor

    m*(k, K*) = min { m : K(k, m) >= K* }

and, because the criterion converges to H(m/k) >= K*, holds the minority
*fraction* alpha = m/k roughly fixed as k varies. The exact C(k, m) form is
used everywhere here rather than the H(m/k) limit, so small-k behaviour is
correct.
"""

from __future__ import annotations

from math import comb, log2

# --- HP tables, verbatim from src/rust/hp_alphabets.rs -----------------------
# Keyed by the kmerseek CLI flag (clap kebab-case). The value is (h_residues,
# p_residues); every table must cover all 20 canonical amino acids exactly once.
HP_TABLES: dict[str, tuple[str, str]] = {
    # Lehninger five-group collapse (kmerseek default; `hp` is an alias)
    "hp-lehninger": ("AFGILMPVWY", "CDEHKNQRST"),
    # Thomas & Dill 1996 ENERGI 2-class reduction (also PBotC 2nd ed Fig 8.28)
    "hp-thomas-dill": ("ACFILMVWY", "DEGHKNPQRST"),
    # Kyte & Doolittle 1982 hydropathy, binarised at hydropathy > 0
    "hp-kyte-doolittle": ("ACFILMV", "DEGHKNPQRSTWY"),
    # Thomas-Dill with C reassigned to polar
    "hp-thomas-dill-no-c": ("AFILMVWY", "CDEGHKNPQRST"),
    # Lehninger with C reassigned to hydrophobic
    "hp-lehninger-plus-c": ("ACFGILMPVWY", "DEHKNQRST"),
    # Phillips et al., Physical Biology of the Cell 1st ed, Fig 8.30
    "hp-pbotc-1st-ed": ("ACFILMPVWY", "DEGHKNQRST"),
}

# snake_case label used in filenames and in kmerseek's `moltype` column.
def label_for(cli_flag: str) -> str:
    """`hp-thomas-dill` -> `hp_thomas_dill` (kmerseek's moltype naming)."""
    return cli_flag.replace("-", "_")


def flag_for(label: str) -> str:
    """`hp_thomas_dill` -> `hp-thomas-dill` (kmerseek CLI naming)."""
    return label.replace("_", "-")


def _validate() -> None:
    for flag, (h, p) in HP_TABLES.items():
        combined = "".join(sorted(h + p))
        assert combined == "ACDEFGHIKLMNPQRSTVWY", (
            f"HP table {flag} does not partition the 20 canonical amino acids: {combined}"
        )


_validate()


def encode_table(cli_flag: str) -> dict[str, str]:
    """Residue -> 'H'/'P' lookup for one alphabet.

    kmerseek stores lowercase h/p internally but emits uppercase in the
    `moltype_seq` column; uppercase is used throughout here to match the output.
    """
    h, p = HP_TABLES[cli_flag]
    table = {r: "H" for r in h}
    table.update({r: "P" for r in p})
    return table


def encode(seq: str, cli_flag: str) -> str:
    """HP-encode a protein sequence. Non-canonical residues become 'X'.

    kmerseek skips k-mers containing non-canonical residues, so 'X' acts as a
    sentinel that callers must exclude rather than silently count as H or P.
    """
    table = encode_table(cli_flag)
    return "".join(table.get(c, "X") for c in seq.upper())


def h_fraction(cli_flag: str) -> float:
    """Fraction of the 20 residues assigned to H — the alphabet's residue-level
    balance. Ranges from 0.35 (kyte-doolittle, 7/13) to 0.55 (lehninger-plus-c,
    11/9), which is why a single complexity cutoff need not imply the same
    false-positive rate across alphabets."""
    h, _ = HP_TABLES[cli_flag]
    return len(h) / 20.0


# --- complexity helpers ------------------------------------------------------

def complexity_bits(k: int, m: int) -> float:
    """Wootton-Federhen compositional complexity in bits/residue for an HP
    k-mer with minority count m: (1/k) * log2 C(k, m)."""
    if not 0 <= m <= k:
        raise ValueError(f"minority count {m} out of range for k={k}")
    return log2(comb(k, m)) / k


def m_star(k: int, k_star: float) -> int:
    """Smallest minority count whose complexity reaches k_star bits/residue.

    Returns k // 2 + 1 (i.e. unreachable) if even the maximum-complexity
    composition m = k // 2 falls below k_star.
    """
    for m in range(k // 2 + 1):
        if complexity_bits(k, m) >= k_star:
            return m
    return k // 2 + 1


def alpha_star(k: int, k_star: float) -> float:
    """m*(k, K*) expressed as a minority fraction."""
    return m_star(k, k_star) / k


def minority_count(hp_kmer: str) -> int:
    """min(n_H, n_P) for an HP string."""
    n_h = hp_kmer.count("H")
    return min(n_h, len(hp_kmer) - n_h)
