#!/usr/bin/env python3
"""The held-out test: choose P66's query window without CD47, then rank CD47.

Section 6l found the one setting of 76 that makes CD47 the sole best match in the
human proteome for a P66 query, and section 6m had to discard it: the query, P66
197-232, is the window seed-and-extend produced *because* it aligns to CD47.
Section 6m then showed the result does not survive centring the window on the
loop instead.

This is the remaining test, and the only one that could still rescue the result.
Rather than centring the window on the loop, it lets the data choose the window
while CD47 is held out:

  1. Hold CD47 out. The discovery set is the 437 human proteins matched to CD47
     on length and acidic fraction, which is the background every earlier test
     used, and which never contained CD47.
  2. For every one of P66's 36-residue windows, score it against that discovery
     set and keep the best score each window reaches. Choose the window with the
     highest such score. CD47 contributes nothing to this choice.
  3. Rank all 19_705 canonical human proteins for that window and report where
     CD47 lands.

If CD47 still comes first for a window chosen this way, the similarity is
findable without knowing the answer and the circularity objection falls. If the
chosen window is somewhere else in P66, or CD47 ranks poorly for it, then P66
197-232 is only special once CD47 has been used to find it.

Fixed before running: alphabet hsdm17, window length 36, metric best ungapped
match count, discovery set the same 437 proteins, selection rule the single
highest-scoring window with ties broken by the lower start position.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_heldout_window.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_can_cd47_rank_first import (  # noqa: E402
    best_match_counts,
    encode_all,
    rank_of,
)
from nb276_hsdm17_prespecified_test import (  # noqa: E402
    ACIDIC_TOLERANCE_PP,
    HUMAN_PROTEOME,
    LENGTH_TOLERANCE,
    N_BACKGROUND,
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
from nb276_why_combining_hurts import P66_SEGMENT  # noqa: E402

ALPHABET = "hsdm17"
WINDOW = 36
#: How many of the leading windows to print, so the choice is visibly not a
#: knife-edge between two nearly equal options.
N_SHOWN = 12


def best_score_per_window(
    query: np.ndarray, targets: list[np.ndarray], window: int
) -> np.ndarray:
    """For every window of the query, the best score it reaches on any target.

    Scoring every window against every offset of every target is one array
    operation per diagonal shift: the score of window `a` at target offset `b` is
    the sum over u of (query[a+u] == target[b+u]), which is the sum of `window`
    shifted copies of the equality matrix.
    """
    n_windows = len(query) - window + 1
    best = np.zeros(n_windows, dtype=np.int16)
    for code in targets:
        if len(code) < window:
            continue
        equal = (query[:, None] == code[None, :]).astype(np.int16)
        n_offsets = len(code) - window + 1
        scores = np.zeros((n_windows, n_offsets), dtype=np.int16)
        for u in range(window):
            scores += equal[u : u + n_windows, u : u + n_offsets]
        np.maximum(best, scores.max(axis=1), out=best)
    return best


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)

    human = read_fasta(HUMAN_PROTEOME)
    names = [n for n, s in human.items() if s.isalpha()]
    sequences = [human[n] for n in names]
    cd47_index = names.index("CD47")

    cd47_acidic = acidic_fraction(cd47["sequence"])
    low = int(len(cd47["sequence"]) * (1 - LENGTH_TOLERANCE))
    high = int(len(cd47["sequence"]) * (1 + LENGTH_TOLERANCE))
    eligible = [
        (n, s)
        for n, s in human.items()
        if low <= len(s) <= high
        and s.isalpha()
        and abs(acidic_fraction(s) - cd47_acidic) * 100 <= ACIDIC_TOLERANCE_PP
        and n != "CD47"
        and s != cd47["sequence"]
    ]
    rng = np.random.default_rng(SEED)
    discovery = [
        eligible[i]
        for i in sorted(
            rng.choice(len(eligible), size=min(N_BACKGROUND, len(eligible)),
                       replace=False)
        )
    ]
    assert all(name != "CD47" for name, _ in discovery), "CD47 leaked into discovery"
    assert all(seq != cd47["sequence"] for _, seq in discovery), "CD47 sequence leaked"
    print(f"Discovery set: {len(discovery)} human proteins matched to CD47 on "
          f"length ({low}-{high} aa) and acidic fraction, CD47 held out.")
    print(f"Choosing among {len(p66['sequence']) - WINDOW + 1} windows of "
          f"{WINDOW} residues in P66, in {ALPHABET}.\n")

    clusters = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}[ALPHABET]
    p66_code = encode_all([p66["sequence"]], clusters)[0]
    discovery_codes = encode_all([s for _, s in discovery], clusters)
    scores = best_score_per_window(p66_code, discovery_codes, WINDOW)

    order = np.lexsort((np.arange(len(scores)), -scores))
    chosen = int(order[0])
    chosen_span = (chosen + 1, chosen + WINDOW)
    print(f"The {N_SHOWN} best windows against the held-out set:")
    for rank, index in enumerate(order[:N_SHOWN], start=1):
        start, end = index + 1, index + WINDOW
        overlap_loop = not (end < loop[0] or start > loop[1])
        overlap_circ = not (end < P66_SEGMENT[0] or start > P66_SEGMENT[1])
        print(f"  {rank:>2}. P66 {start}-{end}  score {scores[index]}"
              f"{'   contains the loop' if overlap_loop else ''}"
              f"{'   overlaps 197-232' if overlap_circ else ''}")

    print(f"\nChosen window: P66 {chosen_span[0]}-{chosen_span[1]}, held-out score "
          f"{scores[chosen]}.")
    print(f"  contains the loop ({loop[0]}-{loop[1]}): "
          f"{not (chosen_span[1] < loop[0] or chosen_span[0] > loop[1])}")
    print(f"  overlaps the circular window (197-232): "
          f"{not (chosen_span[1] < P66_SEGMENT[0] or chosen_span[0] > P66_SEGMENT[1])}")
    print(f"  residues: {p66['sequence'][chosen:chosen + WINDOW]}")

    all_codes = encode_all(sequences, clusters)
    rows = []
    for label, (start, end) in (
        ("held-out choice", chosen_span),
        ("circular window", (P66_SEGMENT[0], P66_SEGMENT[1])),
    ):
        query = p66_code[start - 1 : end]
        counts = best_match_counts(query, all_codes)
        rank, tied, top = rank_of(counts, cd47_index)
        best_other = int(np.sort(counts)[-1])
        rows.append(
            {
                "window": label,
                "p66_span": f"{start}-{end}",
                "chosen_using_cd47": label == "circular window",
                "held_out_score": int(scores[start - 1])
                if start - 1 < len(scores)
                else None,
                "cd47_score": int(counts[cd47_index]),
                "best_score_in_proteome": best_other,
                "cd47_rank": rank,
                "n_tied_with_cd47": tied,
                "n_at_rank_1": top,
                "cd47_sole_first": rank == 1 and top == 1,
            }
        )
        print(f"\n{label}, P66 {start}-{end}: CD47 scores {counts[cd47_index]}, "
              f"the proteome's best is {best_other}, CD47's rank is {rank} of "
              f"{len(names)}, {top} protein(s) at rank 1, sole first "
              f"{rank == 1 and top == 1}.")

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "40_heldout_window_test.csv")
    pl.Config.set_tbl_width_chars(220)
    print()
    print(table)
    held, circ = rows[0], rows[1]
    print(f"\n{'=' * 72}")
    if held["cd47_sole_first"]:
        print("CD47 is the sole top hit for a window chosen without it. The "
              "similarity\nis findable without knowing the answer, and the "
              "circularity objection falls.")
    else:
        print(f"CD47 is NOT the sole top hit for a window chosen without it: rank "
              f"{held['cd47_rank']} of {len(names)}.\nThe window that puts CD47 "
              f"first is only found by using CD47 to find it.")
    print(f"{'=' * 72}")
    print(f"\nWrote {OUTDIR / '40_heldout_window_test.csv'}")


if __name__ == "__main__":
    main()
