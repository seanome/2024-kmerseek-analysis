#!/usr/bin/env python3
"""What is the 36-residue P66 197-232 / CD47 109-144 similarity?

The prespecified test said this segment is specific: no human protein matched to
CD47 on length and acidic content reaches its score. The loop-versus-flank split
said the loop is its weakest part. Neither says what the segment is.

Four questions, each with a comparison that can answer it:

  1. What are these two stretches, structurally? Secondary structure on both
     sides, from the same CA-geometry call used elsewhere in this notebook, plus
     where they sit in each protein's architecture.
  2. Is it a compositional artefact? Composition of both sides against their own
     proteins, and a low-complexity check.
  3. Is it CD47-specific, or does P66 197-232 like immunoglobulin domains in
     general? The same setting run against the 20 SCOPe b.1.1 domains' sequences.
  4. Is P66 197-232 special within P66, or does any 36-residue window of P66 hit
     CD47 about as well? Every window of P66 scored against CD47.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_characterise_segment.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    OUTDIR,
    SCOPE_ASTRAL40,
    ca_secondary_structure,
    resolve_uniprot,
)
from nb276_seed_extend_loop import XDROP, encode, extend_from_seed  # noqa: E402

#: The segment, in each protein's UniProt numbering, from the prespecified test.
P66_SEGMENT = (197, 232)
CD47_SEGMENT = (109, 144)
#: The prespecified setting that found it.
ALPHABET, SEED_LENGTH, PENALTY = "hsdm17", 4, 0.5
#: Window length for question 4, the length of the segment itself.
WINDOW = P66_SEGMENT[1] - P66_SEGMENT[0] + 1


def best_segment_anywhere(
    query: np.ndarray, target: np.ndarray, q_range: range, seed_len: int, penalty: float
) -> float:
    """Best extended score for any seed whose query side starts inside q_range."""
    index: dict[bytes, list[int]] = {}
    for i in range(len(target) - seed_len + 1):
        window = target[i : i + seed_len]
        if window.min() < 0:
            continue
        index.setdefault(window.tobytes(), []).append(i)
    best = 0.0
    for qi in q_range:
        if qi + seed_len > len(query):
            continue
        window = query[qi : qi + seed_len]
        if window.min() < 0:
            continue
        for ti in index.get(window.tobytes(), ()):
            _, _, _, score = extend_from_seed(
                query, target, qi, ti, seed_len, penalty, XDROP
            )
            best = max(best, score)
    return best


def longest_repeat(sequence: str) -> int:
    """Longest run of one residue, a crude low-complexity check."""
    return max(len(m.group()) for m in re.finditer(r"(.)\1*", sequence))


def composition_row(label: str, segment: str, whole: str) -> dict:
    counts = Counter(segment)
    return {
        "which": label,
        "length": len(segment),
        "sequence": segment,
        "acidic_DE_percent": round(
            100 * (counts["D"] + counts["E"]) / len(segment), 1
        ),
        "basic_KR_percent": round(100 * (counts["K"] + counts["R"]) / len(segment), 1),
        "hydrophobic_AILMFWV_percent": round(
            100 * sum(counts[r] for r in "AILMFWV") / len(segment), 1
        ),
        "whole_protein_acidic_percent": round(
            100 * (whole.count("D") + whole.count("E")) / len(whole), 1
        ),
        "n_distinct_residues": len(counts),
        "longest_single_residue_run": longest_repeat(segment),
        "most_common": ", ".join(f"{r}{n}" for r, n in counts.most_common(5)),
    }


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    p66_seq, cd47_seq = p66["sequence"], cd47["sequence"]
    p66_part = p66_seq[P66_SEGMENT[0] - 1 : P66_SEGMENT[1]]
    cd47_part = cd47_seq[CD47_SEGMENT[0] - 1 : CD47_SEGMENT[1]]

    print("=" * 76)
    print(f"P66  {P66_SEGMENT[0]}-{P66_SEGMENT[1]}  {p66_part}")
    print(f"CD47 {CD47_SEGMENT[0]}-{CD47_SEGMENT[1]}  {cd47_part}")
    print("identities:   " + "".join(
        "|" if a == b else " " for a, b in zip(p66_part, cd47_part)
    ))
    n_identical = sum(a == b for a, b in zip(p66_part, cd47_part))
    print(f"{n_identical} identical of {len(p66_part)} "
          f"({100 * n_identical / len(p66_part):.0f}%)")
    print("=" * 76)

    # --- 1. structure ---------------------------------------------------
    p66_ss = pl.read_csv(OUTDIR / "07_p66_secondary_structure.csv")
    in_segment = p66_ss.filter(
        pl.col("full_position").is_between(*P66_SEGMENT)
    )
    af = pl.read_csv(OUTDIR / "02_alphafold_models.csv")
    cd47_model = Path(
        af.filter(pl.col("protein") == "CD47").row(0, named=True)["pdb_path"]
    )
    cd47_ss_all = ca_secondary_structure(cd47_model)
    cd47_ss = cd47_ss_all.filter(pl.col("full_position").is_between(*CD47_SEGMENT))
    print("\n1. SECONDARY STRUCTURE over the segment")
    print(f"P66 {P66_SEGMENT[0]}-{P66_SEGMENT[1]}:  "
          + "".join("E" if s == "strand" else "H" if s == "helix" else "-"
                    for s in in_segment["secondary_structure"]))
    print(f"CD47 {CD47_SEGMENT[0]}-{CD47_SEGMENT[1]}: "
          + "".join("E" if s == "strand" else "H" if s == "helix" else "-"
                    for s in cd47_ss["secondary_structure"]))
    print("  (E strand, H helix, - coil; P66 from its AlphaFold model, which has "
          "mean pLDDT 29)")
    print(in_segment.group_by("secondary_structure").len().sort("len", descending=True))
    print(cd47_ss.group_by("secondary_structure").len().sort("len", descending=True))

    # --- 2. composition -------------------------------------------------
    comp = pl.DataFrame(
        [
            composition_row("P66 197-232", p66_part, p66_seq),
            composition_row("CD47 109-144", cd47_part, cd47_seq),
        ]
    )
    comp.write_csv(OUTDIR / "21_segment_composition.csv")
    pl.Config.set_tbl_width_chars(200)
    print("\n2. COMPOSITION")
    print(comp.drop("sequence"))

    # --- 3. is it CD47 or immunoglobulins in general --------------------
    clusters = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}[ALPHABET]
    query = encode(p66_seq, clusters)
    q_range = range(P66_SEGMENT[0] - 1, P66_SEGMENT[1] - SEED_LENGTH + 1)
    cd47_best = best_segment_anywhere(
        query, encode(cd47_seq, clusters), q_range, SEED_LENGTH, PENALTY
    )

    ig_rows = []
    sequences: dict[str, str] = {}
    name = None
    for line in SCOPE_ASTRAL40.read_text().splitlines():
        if line.startswith(">"):
            match = re.match(r">(\S+)\s+(\S+)", line)
            name = match.group(1) if match and match.group(2).startswith("b.1.1.") \
                else None
            if name:
                sequences[name] = ""
        elif name:
            sequences[name] += line.strip()
    pool = pl.read_csv(OUTDIR / "05_ig_superfamily_pool.csv")["scope_id"].to_list()
    for scope_id in pool:
        seq = sequences.get(scope_id, "").upper()
        if len(seq) < 40:
            continue
        ig_rows.append(
            {
                "scope_id": scope_id,
                "length": len(seq),
                "best_score": best_segment_anywhere(
                    query, encode(seq, clusters), q_range, SEED_LENGTH, PENALTY
                ),
            }
        )
    ig = pl.DataFrame(ig_rows)
    ig.write_csv(OUTDIR / "22_segment_vs_ig_domains.csv")
    n_ge = int((ig["best_score"] >= cd47_best).sum())
    print(f"\n3. IS IT CD47, OR IMMUNOGLOBULINS IN GENERAL")
    print(f"P66 {P66_SEGMENT[0]}-{P66_SEGMENT[1]} against CD47: best score {cd47_best}")
    print(f"against {ig.height} SCOPe b.1.1 immunoglobulin domains: median "
          f"{ig['best_score'].median()}, max {ig['best_score'].max()}")
    print(f"domains reaching CD47's score: {n_ge} of {ig.height}")
    print(ig.sort("best_score", descending=True).head(8))

    # --- 4. is this window special within P66 ---------------------------
    cd47_code = encode(cd47_seq, clusters)
    window_rows = []
    for start in range(1, len(p66_seq) - WINDOW + 2):
        window_rows.append(
            {
                "p66_start": start,
                "p66_end": start + WINDOW - 1,
                "best_score": best_segment_anywhere(
                    query,
                    cd47_code,
                    range(start - 1, start - 1 + WINDOW - SEED_LENGTH + 1),
                    SEED_LENGTH,
                    PENALTY,
                ),
            }
        )
    windows = pl.DataFrame(window_rows)
    windows.write_csv(OUTDIR / "23_p66_windows_vs_cd47.csv")
    observed = windows.filter(pl.col("p66_start") == P66_SEGMENT[0])["best_score"][0]
    n_better = int((windows["best_score"] > observed).sum())
    print(f"\n4. IS THIS WINDOW SPECIAL WITHIN P66")
    print(f"{windows.height} windows of {WINDOW} residues across P66, each scored "
          f"against CD47.")
    print(f"The window starting at {P66_SEGMENT[0]}: {observed}")
    print(f"median {windows['best_score'].median()}, max {windows['best_score'].max()}")
    print(f"windows scoring higher: {n_better} of {windows.height} "
          f"({100 * n_better / windows.height:.1f}%)")
    print(windows.sort("best_score", descending=True).head(8))

    pl.DataFrame(
        [
            {
                "segment_identities": n_identical,
                "segment_length": len(p66_part),
                "cd47_best_score": cd47_best,
                "n_ig_domains_tested": ig.height,
                "n_ig_domains_reaching_cd47": n_ge,
                "ig_median_score": float(ig["best_score"].median()),
                "n_p66_windows": windows.height,
                "p66_window_rank": n_better + 1,
                "p66_windows_scoring_higher": n_better,
                "p66_window_median": float(windows["best_score"].median()),
            }
        ]
    ).write_csv(OUTDIR / "24_segment_characterisation.csv")
    print(f"\nWrote {OUTDIR / '24_segment_characterisation.csv'}")


if __name__ == "__main__":
    main()
