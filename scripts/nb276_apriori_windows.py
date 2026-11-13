#!/usr/bin/env python3
"""A query defined only from the loop, never from CD47: does CD47 still rank first?

Section 6l found one setting where CD47 is the unique best human match for a P66
query, and had to throw it away: the query P66 197-232 was the window
seed-and-extend produced *because* it aligns to CD47, so CD47 topping a search
for it is circular.

The loop query, P66 202-208, is not circular. The paper chose it. But it is 7
residues in a database of 19_705 proteins, so it ties most of the proteome and
can rank nothing. One query is contaminated, the other is powerless.

This runs the query that is neither. The window is defined by a rule that uses
only the loop's own coordinates, which came from the paper, and never looks at
CD47: **take the loop's midpoint and extend a fixed number of residues each
way.** Several half-widths are tried, and all of them are reported, because
picking the best afterwards is how the circularity got in the first time.

If some loop-centred window makes CD47 the unique top hit, the hsdm17 result
survives the circularity objection: the similarity would be findable from the
loop alone. If none does, the hsdm17 result was an artefact of where the window
was placed, and the placement came from CD47.

The same 200 arbitrary human proteins are run through the same settings, so
"CD47 reaches rank 1 somewhere" can be read against how often anything does.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_apriori_windows.py
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
    N_PRETEND_TARGETS,
    best_match_counts,
    encode_all,
    rank_of,
)
from nb276_hsdm17_prespecified_test import HUMAN_PROTEOME, read_fasta  # noqa: E402
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    SEED,
    resolve_uniprot,
)

#: Half-widths each side of the loop's midpoint. 3 reproduces the loop itself and
#: 17 gives a 35-residue window, the length of the circular one, so the two are
#: comparable on everything except where they sit.
HALF_WIDTHS = (3, 5, 7, 10, 14, 17, 20, 25)
#: The circular window, for comparison only. Nothing here is derived from it.
CIRCULAR_WINDOW = (197, 232)
#: A rank shared by more than this many proteins is not a rank.
TIE_LIMIT = 100


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)
    midpoint = (loop[0] + loop[1]) // 2

    human = read_fasta(HUMAN_PROTEOME)
    names = [n for n, s in human.items() if s.isalpha()]
    sequences = [human[n] for n in names]
    cd47_index = names.index("CD47")
    rng = np.random.default_rng(SEED)
    pretend = sorted(
        rng.choice(len(names), size=N_PRETEND_TARGETS, replace=False).tolist()
    )
    print(f"Loop is P66 {loop[0]}-{loop[1]}, midpoint {midpoint}.")
    print(f"Windows are the midpoint plus or minus {HALF_WIDTHS}, so none of them "
          f"uses CD47.")
    print(f"The circular window for comparison is P66 "
          f"{CIRCULAR_WINDOW[0]}-{CIRCULAR_WINDOW[1]}, which contains the loop at "
          f"positions {loop[0] - CIRCULAR_WINDOW[0] + 1} to "
          f"{loop[1] - CIRCULAR_WINDOW[0] + 1} of "
          f"{CIRCULAR_WINDOW[1] - CIRCULAR_WINDOW[0] + 1}: off centre.\n")

    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    rows = []
    best_pretend = {i: len(names) for i in pretend}
    best_pretend_sole = {i: False for i in pretend}
    for alphabet_name in sorted(alphabets):
        clusters = alphabets[alphabet_name]
        codes = encode_all(sequences, clusters)
        p66_code = encode_all([p66["sequence"]], clusters)[0]
        print(f"  {alphabet_name} ({len(clusters)} classes)...", flush=True)
        windows = [
            (
                f"loop midpoint +/- {half}",
                max(1, midpoint - half),
                min(len(p66["sequence"]), midpoint + half),
            )
            for half in HALF_WIDTHS
        ] + [("circular window 197-232", *CIRCULAR_WINDOW)]
        for label, start, end in windows:
            query = p66_code[start - 1 : end]
            counts = best_match_counts(query, codes)
            rank, tied, top = rank_of(counts, cd47_index)
            rows.append(
                {
                    "alphabet": alphabet_name,
                    "n_classes": len(clusters),
                    "window": label,
                    "p66_span": f"{start}-{end}",
                    "window_length": end - start + 1,
                    "defined_from_cd47": label.startswith("circular"),
                    "cd47_score": int(counts[cd47_index]),
                    "best_score_in_proteome": int(counts.max()),
                    "cd47_rank": rank,
                    "n_tied_with_cd47": tied,
                    "n_at_rank_1": top,
                    "cd47_sole_first": rank == 1 and top == 1,
                }
            )
            if label.startswith("circular"):
                continue
            for i in pretend:
                r, _, t = rank_of(counts, i)
                best_pretend[i] = min(best_pretend[i], r)
                if r == 1 and t == 1:
                    best_pretend_sole[i] = True

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "37_apriori_window_ranks.csv")
    control = pl.DataFrame(
        [
            {
                "protein": names[i],
                "best_rank": best_pretend[i],
                "ever_sole_first": best_pretend_sole[i],
            }
            for i in pretend
        ]
    ).sort("best_rank")
    control.write_csv(OUTDIR / "38_apriori_window_control.csv")

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(220)
    apriori = table.filter(~pl.col("defined_from_cd47"))
    circular = table.filter(pl.col("defined_from_cd47"))
    print(f"\n{apriori.height} settings from windows defined only by the loop.\n")
    print("Best of them for CD47:")
    print(apriori.sort("cd47_rank").select(
        "alphabet", "n_classes", "window", "window_length", "cd47_score",
        "best_score_in_proteome", "cd47_rank", "n_at_rank_1", "cd47_sole_first",
    ).head(15))

    sole = apriori.filter(pl.col("cd47_sole_first"))
    print(f"\nLoop-centred windows making CD47 the SOLE top hit: {sole.height} of "
          f"{apriori.height}")
    if not sole.is_empty():
        print(sole.select("alphabet", "window", "window_length", "cd47_score"))
    usable = apriori.filter(pl.col("n_at_rank_1") <= TIE_LIMIT)
    print(f"\nAmong the {usable.height} settings that do not tie more than "
          f"{TIE_LIMIT} proteins at the top, CD47's best rank is "
          f"{usable['cd47_rank'].min()}.")
    print("\nBy window length, CD47's best rank over the 19 alphabets:")
    print(apriori.group_by("window", "window_length").agg(
        pl.col("cd47_rank").min().alias("best_rank"),
        pl.col("n_at_rank_1").min().alias("fewest_tied_at_rank_1"),
        pl.col("cd47_sole_first").any().alias("ever_sole_first"),
    ).sort("window_length"))
    print("\nThe circular window, for comparison:")
    print(circular.sort("cd47_rank").select(
        "alphabet", "cd47_score", "cd47_rank", "n_at_rank_1", "cd47_sole_first"
    ).head(4))

    print(f"\nCONTROL, {control.height} arbitrary human proteins, loop-centred "
          f"windows only:")
    print(f"  ever the sole top hit: {control['ever_sole_first'].sum()}")
    print(f"  reaching rank 1 at all: "
          f"{control.filter(pl.col('best_rank') == 1).height}")
    print(f"  median best rank: {control['best_rank'].median()}")
    print(f"\nWrote {OUTDIR / '37_apriori_window_ranks.csv'}")


if __name__ == "__main__":
    main()
