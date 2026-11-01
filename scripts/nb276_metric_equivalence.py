#!/usr/bin/env python3
"""Do match count, enrichment, an ungapped score and an E-value rank differently?

scripts/nb276_can_cd47_rank_first.py sweeps one number per alphabet and query and
treats it as standing for four metrics. That is a claim, so it is checked here
rather than asserted.

For a query of fixed length L in a fixed alphabet with chance rate p, a target
window matching m of the L columns gives:

    match count        m
    enrichment         m / (p L)
    ungapped score     m - c (L - m) = m (1 + c) - c L
    Karlin-Altschul E  K n L' exp(-lambda S), with S the ungapped score

Each is a strictly monotone function of m, increasing for the first three and
decreasing for the E-value, so all four induce the same ordering of targets and
the same ties. Monotone transforms cannot reorder anything.

The proof is one line, but a proof of the algebra is not a proof that this code
implements the algebra, so the four are computed over the real human proteome for
one setting and the rank vectors compared element by element.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_metric_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_can_cd47_rank_first import best_match_counts, encode_all  # noqa: E402
from nb276_hsdm17_prespecified_test import HUMAN_PROTEOME, read_fasta  # noqa: E402
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    OUTDIR,
    resolve_uniprot,
)
from nb276_why_combining_hurts import P66_SEGMENT, collision_probability  # noqa: E402
from nb276_combined_alphabets import class_tables  # noqa: E402

#: The setting the equivalence is checked on: the alphabet and query that gave
#: the strongest result anywhere in this notebook.
ALPHABET = "hsdm17"
#: Mismatch penalty for the ungapped score, as used in the seed-and-extend runs.
PENALTY = 0.5
#: Karlin-Altschul constants. Their values do not matter to the ranking, which is
#: the point, so any positive pair demonstrates it.
KA_K, KA_LAMBDA = 0.1, 0.3


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    human = read_fasta(HUMAN_PROTEOME)
    names = [n for n, s in human.items() if s.isalpha()]
    sequences = [human[n] for n in names]

    clusters = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}[ALPHABET]
    codes = encode_all(sequences, clusters)
    p66_code = encode_all([p66["sequence"]], clusters)[0]
    query = p66_code[P66_SEGMENT[0] - 1 : P66_SEGMENT[1]]
    length = len(query)
    chance = collision_probability(class_tables([ALPHABET])[0])

    print(f"{ALPHABET}, query P66 {P66_SEGMENT[0]}-{P66_SEGMENT[1]} ({length} aa), "
          f"chance rate {chance:.3f}, over {len(names)} human proteins.")
    matches = best_match_counts(query, codes).astype(float)

    enrichment = matches / (chance * length)
    ungapped = matches * (1 + PENALTY) - PENALTY * length
    # An E-value falls as the score rises, so it is ranked ascending.
    evalue = KA_K * len(names) * length * np.exp(-KA_LAMBDA * ungapped)

    ranks = {
        "match count": rankdata(-matches, method="min"),
        "enrichment": rankdata(-enrichment, method="min"),
        "ungapped score": rankdata(-ungapped, method="min"),
        "E-value": rankdata(evalue, method="min"),
    }
    reference = ranks["match count"]
    rows = []
    for name, values in ranks.items():
        identical = bool(np.array_equal(values, reference))
        rho = float(spearmanr(values, reference).statistic)
        rows.append(
            {
                "metric": name,
                "rank_vector_identical_to_match_count": identical,
                "spearman_with_match_count": round(rho, 6),
                "n_proteins_ranked_differently": int((values != reference).sum()),
                "best_score": round(float(np.max(matches if name == "match count"
                                                 else enrichment if name == "enrichment"
                                                 else ungapped if name == "ungapped score"
                                                 else -evalue)), 3),
            }
        )
    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "36_metric_equivalence.csv")
    pl.Config.set_tbl_width_chars(200)
    print()
    print(table)
    all_same = all(r["rank_vector_identical_to_match_count"] for r in rows)
    print(f"\nAll four give the identical ranking: {all_same}")
    print(f"Distinct scores among {len(names)} proteins: "
          f"{len(np.unique(matches))}, so the ranking has that many levels and "
          f"every metric above has the same ones.")
    print("\nThis is why the sweep carries one column for the four of them, and why"
          "\nonly a summed-IDF weighting is reported as a separate metric: it is the"
          "\nonly one of the five that is not a monotone function of the match count.")
    print(f"\nWrote {OUTDIR / '36_metric_equivalence.csv'}")


if __name__ == "__main__":
    main()
