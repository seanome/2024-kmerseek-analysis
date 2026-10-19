"""Mutations a reduced alphabet cannot see (notebook 347).

If two protein variants have the same encoded sequence R(seq), any predictor that reads
only R(seq) must give them the same value. The best single value for a group of such
variants is the group's mean outcome, so the spread of outcomes around group means is a
floor on the squared error of every such predictor, on these measured variants.

This module holds the pieces the run script (scripts/run_nb347_invisible_mutations.py)
puts together, so they can be tested on small hand-made cases:

- the amino-acid groups of each alphabet, taken from ortholog_analysis_utils, which
  copies them from sourmash's encodings.rs (the tables kmerseek uses for dayhoff6 and
  hp_lehninger2); tests/test_invisible_mutations_utils.py checks them against sourmash's
  own hashes
- parsing of HGVS protein notation and rebuilding of each variant's full sequence
- grouping of variants by encoded sequence, and the floor, total variance and expected
  measurement-noise sums of squares, all with per-variant weights so a bootstrap over
  positions can reuse them
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl

from ortholog_analysis_utils import ALPHABET_AA_GROUPS

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}  # fmt: skip

# kmerseek alphabet name -> name of the same grouping in ALPHABET_AA_GROUPS
_GROUP_NAMES = {"dayhoff6": "dayhoff", "hp_lehninger2": "hp_lehninger"}

# Standard genetic code, codons in TCAG order.
_BASES = "TCAG"
_CODE = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {
    a + b + c: _CODE[16 * i + 4 * j + k]
    for i, a in enumerate(_BASES)
    for j, b in enumerate(_BASES)
    for k, c in enumerate(_BASES)
}

HGVS_SUBSTITUTION = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=)$")


def alphabet_mapping(name: str) -> dict[str, str]:
    """Residue -> encoded symbol for one alphabet. Only the 20 standard residues are keys,
    so encoding anything else raises instead of guessing a class."""
    if name == "identity":
        return {aa: aa for aa in STANDARD_AA}
    groups = ALPHABET_AA_GROUPS[_GROUP_NAMES[name]]
    mapping = {aa: symbol for symbol, aas in groups.items() for aa in aas}
    if sorted(mapping) != sorted(STANDARD_AA):
        raise ValueError(f"{name} does not cover exactly the 20 standard residues")
    return mapping


def encode(sequence: str, name: str) -> str:
    """Encode a protein sequence. 'composition' gives the residue counts in A..Y order."""
    unknown = set(sequence) - set(STANDARD_AA)
    if unknown:
        raise ValueError(f"non-standard residues {sorted(unknown)} cannot be encoded")
    if name == "composition":
        counts = Counter(sequence)
        return ",".join(str(counts[aa]) for aa in STANDARD_AA)
    mapping = alphabet_mapping(name)
    return "".join(mapping[aa] for aa in sequence)


def translate(dna: str) -> str:
    """Translate a coding DNA sequence; stops at the first stop codon (not included)."""
    dna = dna.upper()
    if len(dna) % 3:
        raise ValueError("DNA length is not a multiple of 3")
    protein = []
    for i in range(0, len(dna), 3):
        aa = CODON_TABLE[dna[i : i + 3]]
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


@dataclass(frozen=True)
class ParsedVariant:
    kind: str  # missense, synonymous, stop, malformed, nonstandard_residue,
    # reference_mismatch, wild_type_anchor
    position: int | None = None
    ref: str | None = None
    alt: str | None = None


def parse_hgvs_pro(hgvs: str | None, reference: str) -> ParsedVariant:
    """Classify one HGVS protein string against the reference sequence (1-based)."""
    if hgvs == "_wt":
        # MaveDB's wild-type row; in a min-max normalised score set it is the anchor
        # (score 1 in every replicate by definition), not a measurement.
        return ParsedVariant("wild_type_anchor")
    m = HGVS_SUBSTITUTION.match(hgvs or "")
    if m is None:
        return ParsedVariant("malformed")
    ref3, pos, alt3 = m.group(1), int(m.group(2)), m.group(3)
    if alt3 == "Ter":
        return ParsedVariant("stop", pos)
    ref = AA3_TO_1.get(ref3)
    alt = ref if alt3 == "=" else AA3_TO_1.get(alt3)
    if ref is None or alt is None:
        return ParsedVariant("nonstandard_residue", pos)
    if not 1 <= pos <= len(reference) or reference[pos - 1] != ref:
        return ParsedVariant("reference_mismatch", pos, ref, alt)
    if alt == ref:
        return ParsedVariant("synonymous", pos, ref, alt)
    return ParsedVariant("missense", pos, ref, alt)


def apply_substitution(reference: str, position: int, alt: str) -> str:
    return reference[: position - 1] + alt + reference[position:]


def replicate_matrix(df: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return df.select(columns).to_numpy().astype(float)


def build_variants(
    scores: pl.DataFrame,
    reference: str,
    *,
    assay: str,
    condition: str,
    background: str,
    replicate_columns: list[str],
    outcome: str = "score",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """One row per distinct full protein sequence, plus exclusions and synonymous rows.

    Rows giving the same protein sequence are pooled: their replicate scores are joined,
    and the variant's outcome becomes the mean of all of them, so a sequence measured
    twice still counts once. Every row not kept as a variant is listed with its reason.
    """
    reps = replicate_matrix(scores, replicate_columns)
    kept, excluded, synonymous = [], [], []
    for row, rep in zip(scores.iter_rows(named=True), reps):
        p = parse_hgvs_pro(row["hgvs_pro"], reference)
        base = {"accession": row["accession"], "hgvs_pro": row["hgvs_pro"]}
        if p.kind != "missense" and p.kind != "synonymous":
            excluded.append({**base, "reason": p.kind})
            continue
        if row[outcome] is None:
            excluded.append({**base, "reason": "missing_score"})
            continue
        values = rep[~np.isnan(rep)]
        record = {
            **base,
            "position": p.position,
            "ref": p.ref,
            "alt": p.alt,
            "replicates": values.tolist(),
            "reported_score": row[outcome],
            "abundance_class": row.get("abundance_class"),
        }
        (synonymous if p.kind == "synonymous" else kept).append(record)

    by_sequence: dict[str, list[dict]] = {}
    for r in kept:
        seq = apply_substitution(reference, r["position"], r["alt"])
        by_sequence.setdefault(seq, []).append(r)

    rows = []
    for seq, records in by_sequence.items():
        first = records[0]
        for dup in records[1:]:
            excluded.append(
                {"accession": dup["accession"], "hgvs_pro": dup["hgvs_pro"],
                 "reason": f"duplicate_sequence (pooled into {first['accession']})"}
            )  # fmt: skip
        values = np.concatenate([np.asarray(r["replicates"]) for r in records])
        n = len(values)
        sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        rows.append(
            {
                "assay": assay,
                "condition": condition,
                "background": background,
                "variant_id": first["accession"],
                "hgvs_pro": first["hgvs_pro"],
                "position": first["position"],
                "ref": first["ref"],
                "alt": first["alt"],
                "sequence": seq,
                "y": float(np.mean(values)),
                "reported_score": first["reported_score"],
                "n_replicates": n,
                "sd": sd,
                "se": sd / np.sqrt(n) if n > 1 else float("nan"),
                "replicates": values.tolist(),
                "abundance_class": first["abundance_class"],
                "n_rows_pooled": len(records),
            }
        )
    variants = pl.DataFrame(rows, schema_overrides={"replicates": pl.List(pl.Float64)})
    excl = pl.DataFrame(
        excluded, schema={"accession": pl.Utf8, "hgvs_pro": pl.Utf8, "reason": pl.Utf8}
    )
    syn = pl.DataFrame(synonymous) if synonymous else pl.DataFrame()
    return variants, excl, syn


def group_ids(keys: list[tuple]) -> np.ndarray:
    """Integer group id per record; records share an id only if their keys are equal.
    Dict lookup compares keys exactly after hashing, so a hash clash cannot merge groups.
    """
    index: dict[tuple, int] = {}
    return np.array([index.setdefault(k, len(index)) for k in keys], dtype=np.int64)


def floor_sums(
    y: np.ndarray, groups: np.ndarray, v: np.ndarray, w: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-group weighted sums for the floor and the noise term.

    y: outcome per variant; groups: group id per variant; v: variance of the variant's
    measured mean (se^2); w: weight per variant (1 = counted once, 0 = left out).

    Returns per group: W (total weight), sse (weighted sum of squares around the weighted
    group mean), noise (its expected value if all members shared one true outcome and
    errors were independent between variants: sum w v - sum w^2 v / W), and n_distinct
    (members with w > 0).
    """
    if w is None:
        w = np.ones_like(y)
    G = int(groups.max()) + 1 if len(groups) else 0
    W = np.bincount(groups, w, G)
    S1 = np.bincount(groups, w * y, G)
    S2 = np.bincount(groups, w * y * y, G)
    V1 = np.bincount(groups, w * v, G)
    V2 = np.bincount(groups, w * w * v, G)
    nd = np.bincount(groups, (w > 0).astype(float), G)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(W > 0, S1 / W, 0.0)
        sse = np.where(W > 0, S2 - W * mean * mean, 0.0)
        noise = np.where(W > 0, V1 - V2 / W, 0.0)
    # S2 - W mean^2 can come out as -1e-16 for groups with equal values.
    sse = np.maximum(sse, 0.0)
    return {"W": W, "sse": sse, "noise": noise, "n_distinct": nd, "mean": mean}


def floor_summary(
    y: np.ndarray,
    groups: np.ndarray,
    v: np.ndarray,
    w: np.ndarray | None = None,
    subset: np.ndarray | None = None,
) -> dict[str, float]:
    """Floor, total variance and noise numbers for one grouping.

    subset: boolean mask over groups; only those groups' sums enter the "part" numbers,
    while the whole-dataset denominator N stays the total weight of all variants.
    """
    if w is None:
        w = np.ones_like(y)
    s = floor_sums(y, groups, v, w)
    collided = s["n_distinct"] >= 2
    part = collided if subset is None else (collided & subset)
    N = float(w.sum())
    mean_all = float((w * y).sum() / N)
    total_var = float((w * (y - mean_all) ** 2).sum() / N)
    sse = float(s["sse"][part].sum())
    noise = float(s["noise"][part].sum())
    n_coll = float(s["W"][part].sum())
    floor_mse = sse / N
    return {
        "N": N,
        "n_groups_collided": int(part.sum()),
        "n_variants_collided": n_coll,
        "fraction_collided": n_coll / N,
        "total_variance": total_var,
        "floor_mse": floor_mse,
        "floor_rmse": float(np.sqrt(floor_mse)),
        "collided_mse": sse / n_coll if n_coll else float("nan"),
        "collided_rmse": float(np.sqrt(sse / n_coll)) if n_coll else float("nan"),
        "floor_fraction": floor_mse / total_var if total_var > 0 else float("nan"),
        "expected_noise_mse": noise / N,
        "noise_ratio": sse / noise if noise > 0 else float("nan"),
        "noise_adjusted_floor_mse": (sse - noise) / N,
    }


def kmer_set(encoded: str, k: int) -> set[str]:
    return {encoded[i : i + k] for i in range(len(encoded) - k + 1)}
