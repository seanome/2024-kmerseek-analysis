#!/usr/bin/env python3
"""Is the hsdm17 segment specific to CD47 because of the loop, or despite it?

scripts/nb276_hsdm17_prespecified_test.py passed: no human protein matched to
CD47 on length and acidic content reaches CD47's score of 6.0 at the prespecified
setting. But that score is the whole 36-residue segment, and
scripts/nb276_seed_extend_loop_contribution.py showed the loop contributes only
1.0 of it, scoring worse per column than its own flanks.

So the passing test may be about the window P66 197-232 rather than about the loop
the claim names. This splits the same background by column: CD47's loop columns
against every matched protein's loop columns, and CD47's flanking columns against
theirs.

**These two statistics are post-hoc.** The prespecified test is the other script;
this one was written after seeing its result, and its p-values are described as
such. It is here to say what the passing test is evidence about, not to add a
second claim.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_hsdm17_loop_vs_flanks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_hsdm17_prespecified_test import (  # noqa: E402
    ACIDIC_TOLERANCE_PP,
    ALPHABET,
    HUMAN_PROTEOME,
    LENGTH_TOLERANCE,
    N_BACKGROUND,
    PENALTY,
    SEED_LENGTH,
    acidic_fraction,
    read_fasta,
)
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    SEED,
    resolve_uniprot,
)
from nb276_seed_extend_loop import best_loop_segment, encode  # noqa: E402


def split_score(
    query: np.ndarray,
    target: np.ndarray,
    segment: dict,
    loop: tuple[int, int],
    penalty: float,
) -> dict:
    """Split one segment's score into the loop's columns and the rest."""
    loop_score = other_score = 0.0
    loop_cols = other_cols = 0
    for i in range(segment["q_end"] - segment["q_start"]):
        qi = segment["q_start"] + i
        ti = segment["t_start"] + i
        if not (0 <= qi < len(query) and 0 <= ti < len(target)):
            continue
        column = 1.0 if query[qi] == target[ti] and query[qi] >= 0 else -penalty
        if loop[0] - 1 <= qi <= loop[1] - 1:
            loop_score += column
            loop_cols += 1
        else:
            other_score += column
            other_cols += 1
    return {
        "loop_score": loop_score,
        "loop_columns": loop_cols,
        "flank_score": other_score,
        "flank_columns": other_cols,
        "loop_per_column": loop_score / loop_cols if loop_cols else None,
        "flank_per_column": other_score / other_cols if other_cols else None,
    }


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)
    loop0 = range(loop[0] - 1, loop[1])

    clusters = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}[ALPHABET]
    query = encode(p66["sequence"], clusters)
    cd47_code = encode(cd47["sequence"], clusters)
    observed = best_loop_segment(query, cd47_code, loop0, SEED_LENGTH, PENALTY)
    observed_split = split_score(query, cd47_code, observed, loop, PENALTY)
    print(f"CD47, {ALPHABET} seed {SEED_LENGTH} penalty {PENALTY}:")
    print(f"  total {observed['score']}  loop {observed_split['loop_score']} over "
          f"{observed_split['loop_columns']} columns  flanks "
          f"{observed_split['flank_score']} over {observed_split['flank_columns']}")
    print(f"  per column: loop {observed_split['loop_per_column']:.3f}, flanks "
          f"{observed_split['flank_per_column']:.3f}\n")

    cd47_acidic = acidic_fraction(cd47["sequence"])
    cd47_length = len(cd47["sequence"])
    human = read_fasta(HUMAN_PROTEOME)
    low = int(cd47_length * (1 - LENGTH_TOLERANCE))
    high = int(cd47_length * (1 + LENGTH_TOLERANCE))
    eligible = [
        (name, seq)
        for name, seq in human.items()
        if low <= len(seq) <= high
        and seq.isalpha()
        and abs(acidic_fraction(seq) - cd47_acidic) * 100 <= ACIDIC_TOLERANCE_PP
        and name != "CD47"
        and seq != cd47["sequence"]
    ]
    rng = np.random.default_rng(SEED)
    picked = [
        eligible[i]
        for i in sorted(
            rng.choice(
                len(eligible), size=min(N_BACKGROUND, len(eligible)), replace=False
            )
        )
    ]
    print(f"Same {len(picked)} matched proteins as the prespecified test.\n")

    rows = []
    for i, (name, sequence) in enumerate(picked, start=1):
        if i % 100 == 0:
            print(f"  {i}/{len(picked)}...", flush=True)
        code = encode(sequence, clusters)
        found = best_loop_segment(query, code, loop0, SEED_LENGTH, PENALTY)
        if found is None:
            rows.append({"gene": name, "reaches_loop": False, "total_score": 0.0,
                         "loop_score": None, "flank_score": None,
                         "loop_per_column": None, "flank_per_column": None})
            continue
        split = split_score(query, code, found, loop, PENALTY)
        rows.append(
            {
                "gene": name,
                "reaches_loop": True,
                "total_score": found["score"],
                "loop_score": split["loop_score"],
                "flank_score": split["flank_score"],
                "loop_per_column": split["loop_per_column"],
                "flank_per_column": split["flank_per_column"],
            }
        )

    background = pl.DataFrame(rows)
    background.write_csv(OUTDIR / "17_hsdm17_loop_vs_flanks_background.csv")
    reached = background.filter(pl.col("reaches_loop"))

    # Proteins that never reach the loop have no loop score. Scoring them as 0 is
    # the conservative choice for CD47: it can only make CD47 look more unusual,
    # so both conventions are reported.
    loop_scores_zero = background["loop_score"].fill_null(0.0)
    n_loop_ge_all = int((loop_scores_zero >= observed_split["loop_score"]).sum())
    n_loop_ge_reached = int(
        (reached["loop_score"] >= observed_split["loop_score"]).sum()
    )
    n_flank_ge = int((reached["flank_score"] >= observed_split["flank_score"]).sum())

    summary = pl.DataFrame(
        [
            {
                "statistic": "loop columns only",
                "observed": observed_split["loop_score"],
                "n_background_ge_counting_non_reaching_as_zero": n_loop_ge_all,
                "n_background": background.height,
                "p_counting_all": n_loop_ge_all / background.height,
                "n_background_ge_among_those_reaching": n_loop_ge_reached,
                "n_reaching": reached.height,
                "p_among_reaching": n_loop_ge_reached / reached.height
                if reached.height
                else None,
            },
            {
                "statistic": "flanking columns only",
                "observed": observed_split["flank_score"],
                "n_background_ge_counting_non_reaching_as_zero": n_flank_ge,
                "n_background": background.height,
                "p_counting_all": n_flank_ge / background.height,
                "n_background_ge_among_those_reaching": n_flank_ge,
                "n_reaching": reached.height,
                "p_among_reaching": n_flank_ge / reached.height
                if reached.height
                else None,
            },
        ]
    )
    summary.write_csv(OUTDIR / "18_hsdm17_loop_vs_flanks_result.csv")

    pl.Config.set_tbl_width_chars(200)
    print(f"\n{'=' * 72}")
    print("POST-HOC, not prespecified.\n")
    print(summary)
    print(f"\n{reached.height} of {background.height} matched proteins reach the loop.")
    print(f"Among those, loop-column score: median "
          f"{reached['loop_score'].median()}, max {reached['loop_score'].max()}; "
          f"CD47 is {observed_split['loop_score']}.")
    print(f"Among those, flank-column score: median "
          f"{reached['flank_score'].median()}, max {reached['flank_score'].max()}; "
          f"CD47 is {observed_split['flank_score']}.")
    print(f"{'=' * 72}")
    print(f"\nWrote {OUTDIR / '18_hsdm17_loop_vs_flanks_result.csv'}")


if __name__ == "__main__":
    main()
