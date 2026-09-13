"""Composition controls for the Swiss-Prot feature-type benchmark (notebook 231).

Two no-search arms that see only the human query sequence and label TRANSMEM:

* ``kd_transmem_calls``: Kyte & Doolittle 1982 hydropathy, 19-residue window, mean above
  1.6 marks a membrane-spanning segment. The classic single-sequence TM scan.
* ``hp_run_calls``: the longest exact runs of the hydrophobic class in one of kmerseek's
  own HP alphabets. This is what a 2-letter k-mer index can carry about a TM helix.

Both produce call tables in the pipeline's schema (query_acc, pfam_id, qstart, qend,
score) so evaluate_domain_calls.score_calls scores them exactly as it scores a search arm.

Coordinates are 1-based like the truth tables; the scorer treats (start, end) as
half-open when it subtracts them, and the truth was built the same way, so no shift.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import polars as pl

HUMAN_FASTA = Path(
    "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota/UP000005640_9606.fasta"
)

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

#: The feature types whose truth intervals a hydrophobicity scan could reproduce with no
#: homology at all, plus the ones that are sequence-composition or geometry rather than
#: a family. Everything else is a family-bearing or site feature.
COMPOSITION_TYPES = ["TRANSMEM", "INTRAMEM", "COILED", "REGION", "REPEAT"]


def read_fasta(path: Path, keep: set[str] | None = None) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    acc = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                acc = name.split("|")[1] if "|" in name else name
                if keep is not None and acc not in keep:
                    acc = None
                    continue
                seqs[acc] = []
            elif acc is not None:
                seqs[acc].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) 0-based half-open runs of True."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def kd_transmem_calls(
    seqs: dict[str, str], window: int = 19, threshold: float = 1.6
) -> pl.DataFrame:
    """Windows whose mean hydropathy exceeds `threshold`; merged into segments spanning
    the full windows. Score is the segment's best window mean."""
    rows = []
    half = window // 2
    for acc, s in seqs.items():
        h = np.array([KYTE_DOOLITTLE.get(c, 0.0) for c in s.upper()])
        if h.size < window:
            continue
        means = np.convolve(
            h, np.ones(window) / window, mode="valid"
        )  # centre index i -> residues i..i+window-1
        hot = means > threshold
        for a, b in _segments(hot):
            # window centres a..b-1 -> residues a .. b-1+window-1 (0-based), 1-based inclusive below
            rows.append(
                {
                    "query_acc": acc,
                    "pfam_id": "TRANSMEM",
                    "qstart": a + 1,
                    "qend": b - 1 + window,
                    "score": float(means[a:b].max()),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "query_acc": pl.String,
            "pfam_id": pl.String,
            "qstart": pl.Int64,
            "qend": pl.Int64,
            "score": pl.Float64,
        },
    )


def hp_run_calls(
    seqs: dict[str, str],
    h_residues: str = "ACFILMVWY",
    min_run: int = 12,
    max_polar: int = 0,
) -> pl.DataFrame:
    """Runs of hydrophobic-class residues of length >= `min_run`, allowing up to
    `max_polar` polar residues inside the run (0 = exact run). Score is the run length.
    """
    hset = set(h_residues)
    rows = []
    for acc, s in seqs.items():
        is_h = np.array([c in hset for c in s.upper()])
        if max_polar == 0:
            segs = _segments(is_h)
        else:
            # Greedy: extend while the count of polar residues in the run stays <= max_polar.
            segs = []
            i, n = 0, len(is_h)
            while i < n:
                if not is_h[i]:
                    i += 1
                    continue
                j, polar = i, 0
                last_h = i
                while j < n:
                    if not is_h[j]:
                        polar += 1
                        if polar > max_polar:
                            break
                    else:
                        last_h = j
                    j += 1
                segs.append((i, last_h + 1))
                i = last_h + 1
        for a, b in segs:
            if b - a >= min_run:
                rows.append(
                    {
                        "query_acc": acc,
                        "pfam_id": "TRANSMEM",
                        "qstart": a + 1,
                        "qend": b,
                        "score": float(b - a),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "query_acc": pl.String,
            "pfam_id": pl.String,
            "qstart": pl.Int64,
            "qend": pl.Int64,
            "score": pl.Float64,
        },
    )


def parse_arm(filename: str) -> dict:
    """swissprot.<tool>.<variant>.<species>[.dedup].calls.parquet -> fields."""
    parts = filename.split(".")
    assert parts[0] == "swissprot" and parts[-2] == "calls"
    dedup = parts[-3] == "dedup"
    core = parts[1:-3] if dedup else parts[1:-2]
    tool, species = core[0], core[-1]
    variant = ".".join(core[1:-1])
    return {
        "tool": tool,
        "variant": variant,
        "species": species,
        "dedup": dedup,
        "arm": f"{tool}.{variant}",
    }


def fmax_for(calls: pl.DataFrame, truth: pl.DataFrame, ic: pl.DataFrame, cm) -> dict:
    """Interval-level Fmax with the pipeline's own curve; the same call the pipeline makes."""
    if truth.height == 0:
        return {"fmax": np.nan, "fmax_precision": np.nan, "fmax_recall": np.nan}
    curve = cm.protein_centric_curve(calls, truth, ic)
    sc = cm.cafa_scalars(curve)
    return {k: sc[k] for k in ("fmax", "fmax_precision", "fmax_recall")}
