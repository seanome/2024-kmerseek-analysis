#!/usr/bin/env python3
"""The prespecified test of the one seed-and-extend lead, hsdm17 seed 4 penalty 0.5.

scripts/nb276_seed_extend_loop.py swept 132 settings and one came close: hsdm17
with a 4-class seed and mismatch penalty 0.5 puts a 36-residue ungapped segment
across the P66 loop and onto 9 residues of CD47's SIRPalpha face, at p = 0.0004
against shuffled CD47. That p-value is not quotable, because it is the best of 132
tries and shuffling CD47 is a weak null: it destroys the protein, so almost any
real protein beats it.

This is the test that would make the lead quotable or kill it. Everything below is
fixed before the test runs, and nothing is swept:

  alphabet            hsdm17
  seed length         4
  mismatch penalty    0.5
  X-drop              10
  statistic           the score of the best seed-and-extend segment whose P66 side
                      covers any of the loop, P66 202-208
  observed            whatever that statistic is for CD47
  null                real human proteins matched to CD47 on length and on acidic
                      fraction, which is what the loop is made of
  p                   the share of matched proteins scoring at least as high
  decision            the lead survives only if p < 0.05 on the primary statistic

A second statistic is reported alongside, also fixed in advance: the share of
matched proteins that reach the observed score with a segment at least as long as
CD47's. A short high-scoring segment is a different thing from a 36-residue one.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_hsdm17_prespecified_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    SEED,
    resolve_uniprot,
)
from nb276_seed_extend_loop import best_loop_segment, encode  # noqa: E402

# ---------------------------------------------------------------------------
# Everything here is fixed before the test. None of it is swept.
# ---------------------------------------------------------------------------
ALPHABET = "hsdm17"
SEED_LENGTH = 4
PENALTY = 0.5
#: Matched proteins must be within this fraction of CD47's length.
LENGTH_TOLERANCE = 0.10
#: and within this many percentage points of CD47's D+E share.
ACIDIC_TOLERANCE_PP = 1.5
#: How many matched proteins to draw. 500 resolves p to 0.002.
N_BACKGROUND = 500
#: The decision threshold, on the primary statistic.
ALPHA = 0.05

HUMAN_PROTEOME = Path(
    "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
)


#: Pipe-separated field holding the gene symbol in a GENCODE translation header.
#: Field 5 is the transcript name (CD47-202) and field 6 is the gene symbol (CD47);
#: reading field 5 let CD47's own transcript into the background set as "CD47-202"
#: and it then came back as the one protein matching CD47's score.
GENCODE_GENE_SYMBOL_FIELD = 6


def read_fasta(path: Path) -> dict[str, str]:
    """Gene symbol to sequence, taking the first entry seen for each symbol."""
    sequences: dict[str, str] = {}
    name, chunks = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name and name not in sequences:
                sequences[name] = "".join(chunks)
            fields = line[1:].split("|")
            name = (
                fields[GENCODE_GENE_SYMBOL_FIELD]
                if len(fields) > GENCODE_GENE_SYMBOL_FIELD
                else fields[0]
            )
            chunks = []
        else:
            chunks.append(line.strip())
    if name and name not in sequences:
        sequences[name] = "".join(chunks)
    return sequences


def acidic_fraction(sequence: str) -> float:
    return (sequence.count("D") + sequence.count("E")) / len(sequence)


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)
    loop0 = range(loop[0] - 1, loop[1])

    clusters = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}[ALPHABET]
    query = encode(p66["sequence"], clusters)

    # --- the observed value ---------------------------------------------
    observed = best_loop_segment(
        query, encode(cd47["sequence"], clusters), loop0, SEED_LENGTH, PENALTY
    )
    if observed is None:
        raise SystemExit("The prespecified setting does not reach the loop on CD47.")
    observed_score = observed["score"]
    observed_length = observed["q_end"] - observed["q_start"]
    cd47_length = len(cd47["sequence"])
    cd47_acidic = acidic_fraction(cd47["sequence"])
    print(f"Setting fixed in advance: {ALPHABET}, seed {SEED_LENGTH}, penalty "
          f"{PENALTY}.")
    print(f"Observed on CD47: score {observed_score}, segment {observed_length} "
          f"residues, P66 {observed['q_start'] + 1}-{observed['q_end']}.")
    print(f"CD47: {cd47_length} aa, acidic (D+E) fraction "
          f"{cd47_acidic:.3f}.\n")

    # --- the matched background -----------------------------------------
    if not HUMAN_PROTEOME.exists():
        raise SystemExit(f"Missing {HUMAN_PROTEOME}")
    human = read_fasta(HUMAN_PROTEOME)
    low = int(cd47_length * (1 - LENGTH_TOLERANCE))
    high = int(cd47_length * (1 + LENGTH_TOLERANCE))
    eligible = [
        (name, seq)
        for name, seq in human.items()
        if low <= len(seq) <= high
        and seq.isalpha()
        and abs(acidic_fraction(seq) - cd47_acidic) * 100 <= ACIDIC_TOLERANCE_PP
        # Exclude CD47 itself, and anything whose sequence is CD47's, so the
        # background cannot contain the protein being tested.
        and name != "CD47"
        and seq != cd47["sequence"]
    ]
    print(f"{len(human)} canonical human proteins; {len(eligible)} match CD47 on "
          f"length ({low}-{high} aa) and acidic fraction "
          f"(within {ACIDIC_TOLERANCE_PP} points of {cd47_acidic * 100:.1f}%).")
    if len(eligible) < 50:
        raise SystemExit("Too few matched proteins to run the test.")

    rng = np.random.default_rng(SEED)
    picked = [
        eligible[i]
        for i in sorted(
            rng.choice(
                len(eligible), size=min(N_BACKGROUND, len(eligible)), replace=False
            )
        )
    ]
    print(f"Testing {len(picked)} of them.\n")

    rows = []
    for i, (name, sequence) in enumerate(picked, start=1):
        if i % 100 == 0:
            print(f"  {i}/{len(picked)}...", flush=True)
        found = best_loop_segment(
            query, encode(sequence, clusters), loop0, SEED_LENGTH, PENALTY
        )
        rows.append(
            {
                "gene": name,
                "length": len(sequence),
                "acidic_fraction": round(acidic_fraction(sequence), 4),
                "reaches_loop": found is not None,
                "score": found["score"] if found else 0.0,
                "segment_length": (found["q_end"] - found["q_start"]) if found else 0,
            }
        )

    background = pl.DataFrame(rows)
    background.write_csv(OUTDIR / "15_hsdm17_prespecified_background.csv")

    n_ge = int((background["score"] >= observed_score).sum())
    p_primary = n_ge / background.height
    both = background.filter(
        (pl.col("score") >= observed_score)
        & (pl.col("segment_length") >= observed_length)
    )
    p_secondary = both.height / background.height

    summary = pl.DataFrame(
        [
            {
                "alphabet": ALPHABET,
                "seed_length": SEED_LENGTH,
                "penalty": PENALTY,
                "observed_score": observed_score,
                "observed_segment_length": observed_length,
                "n_background": background.height,
                "background_reaching_loop": int(background["reaches_loop"].sum()),
                "background_median_score": float(background["score"].median()),
                "background_max_score": float(background["score"].max()),
                "n_background_score_ge_observed": n_ge,
                "p_primary": p_primary,
                "n_background_score_and_length_ge": both.height,
                "p_secondary": p_secondary,
                "alpha": ALPHA,
                "lead_survives": p_primary < ALPHA,
            }
        ]
    )
    summary.write_csv(OUTDIR / "16_hsdm17_prespecified_result.csv")

    pl.Config.set_tbl_width_chars(200)
    print(f"\n{'=' * 72}")
    print(f"Background: {int(background['reaches_loop'].sum())} of "
          f"{background.height} matched human proteins produce a segment covering "
          f"the loop at all.")
    print(f"  score median {background['score'].median()}, "
          f"max {background['score'].max()}")
    print(f"\nPRIMARY: {n_ge} of {background.height} match or beat CD47's "
          f"{observed_score}  ->  p = {p_primary:.3f}")
    print(f"SECONDARY (also at least {observed_length} residues long): "
          f"{both.height} of {background.height}  ->  p = {p_secondary:.3f}")
    print(f"\nDecision rule was p < {ALPHA} on the primary statistic.")
    print(f"VERDICT: the lead {'SURVIVES' if p_primary < ALPHA else 'DOES NOT SURVIVE'}.")
    print(f"{'=' * 72}")
    print("\nThe matched proteins that beat CD47, if any:")
    print(background.filter(pl.col("score") >= observed_score)
          .sort("score", descending=True)
          .select("gene", "length", "acidic_fraction", "score", "segment_length")
          .head(20))
    print(f"\nWrote {OUTDIR / '16_hsdm17_prespecified_result.csv'}")


if __name__ == "__main__":
    main()
