#!/usr/bin/env python3
"""Would an E-value, an enrichment or a TF-IDF weighting see any of this better?

Every score so far counts a matched column as +1 whatever it is. kmerseek's own
values do not: it ranks by E-value, by enrichment and by summed inverse document
frequency, so a k-mer that is rare across the corpus counts for more than a common
one. This asks whether any of that changes the two verdicts.

  **Enrichment** is already here under another name. Observed matched columns
  against the columns chance gives is exactly observed over expected, and
  scripts/nb276_which_merges_carry_the_segment.py prints it per alphabet. It is
  repeated in the output below so the three sit together.

  **An E-value** would add nothing. A Karlin-Altschul E-value is a parametric
  stand-in for the null this notebook already has empirically: 437 real human
  proteins matched to CD47 on length and acidic fraction. Where the empirical
  null exists, fitting a distribution to approximate it can only lose accuracy,
  and this project has twice refused a KA fit on data this thin. So the E-value
  is not computed and the empirical p-value is used instead, which is the same
  quantity measured rather than modelled.

  **TF-IDF is the one that could genuinely change the answer**, and it is tested
  here. Each matched k-mer is weighted by how rare it is across the 19_705
  canonical human proteins rather than counted as 1. If the segment's matches are
  rare k-mers, weighting should sharpen it; if they are common, weighting should
  wash it out. The same weighting is applied to the loop, which answers the
  separate question of whether a rarity-weighted metric would find the loop that
  plain counting cannot.

Six alphabets are tested, fixed in advance to span the range rather than chosen
after looking: protein20, hsdm17, sdm12, funcgroups8, polarity4 and
hp_pbotc_1st_ed2. Nineteen would take an hour for an answer these six give.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_idf_weighted_segment.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
from nb276_seed_extend_loop import encode  # noqa: E402
from nb276_why_combining_hurts import CD47_SEGMENT, P66_SEGMENT  # noqa: E402

#: Alphabets tested, fixed in advance to span 2 to 20 classes.
ALPHABETS = (
    "protein20",
    "hsdm17",
    "sdm12",
    "funcgroups8",
    "polarity4",
    "hp_pbotc_1st_ed2",
)
#: k-mer length the weighting is computed on. 5 is the seed length the
#: seed-and-extend runs used and short enough that a k-mer still occurs in the
#: corpus often enough for a document frequency to mean something.
KSIZE = 5


def document_frequency(
    sequences: list[str], clusters: list[str], k: int
) -> tuple[dict[bytes, int], int]:
    """How many proteins each encoded k-mer appears in, and how many proteins."""
    counts: dict[bytes, int] = {}
    for sequence in sequences:
        code = encode(sequence, clusters)
        if len(code) < k:
            continue
        seen = {
            code[i : i + k].tobytes()
            for i in range(len(code) - k + 1)
            if code[i : i + k].min() >= 0
        }
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    return counts, len(sequences)


def idf(key: bytes, counts: dict[bytes, int], n_documents: int) -> float:
    """Inverse document frequency, the same form kmerseek's sum-IDF value uses."""
    return math.log(n_documents / (1 + counts.get(key, 0)))


def matched_kmer_idf(
    query_code: np.ndarray,
    target_code: np.ndarray,
    q_start: int,
    t_start: int,
    length: int,
    k: int,
    counts: dict[bytes, int],
    n_documents: int,
) -> tuple[float, int]:
    """Summed IDF of every k-mer that matches between the two, at a fixed offset."""
    total, n = 0.0, 0
    for i in range(length - k + 1):
        qi, ti = q_start + i, t_start + i
        if qi + k > len(query_code) or ti + k > len(target_code):
            break
        q_kmer = query_code[qi : qi + k]
        t_kmer = target_code[ti : ti + k]
        if q_kmer.min() < 0 or not np.array_equal(q_kmer, t_kmer):
            continue
        total += idf(q_kmer.tobytes(), counts, n_documents)
        n += 1
    return total, n


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)
    seg_len = P66_SEGMENT[1] - P66_SEGMENT[0] + 1

    print("Reading the human proteome...", flush=True)
    human = read_fasta(HUMAN_PROTEOME)
    corpus = [s for s in human.values() if s.isalpha()]
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
    picked = [
        eligible[i]
        for i in sorted(
            rng.choice(len(eligible), size=min(N_BACKGROUND, len(eligible)),
                       replace=False)
        )
    ]
    print(f"{len(corpus)} proteins in the corpus, {len(picked)} in the matched "
          f"background.\n")

    merges = pl.read_csv(OUTDIR / "30_segment_merges_by_alphabet.csv")
    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    rows, loop_rows = [], []
    for name in ALPHABETS:
        clusters = alphabets[name]
        print(f"  {name} ({len(clusters)} classes): counting k-mers...", flush=True)
        counts, n_documents = document_frequency(corpus, clusters, KSIZE)
        query = encode(p66["sequence"], clusters)
        cd47_code = encode(cd47["sequence"], clusters)

        observed_idf, observed_n = matched_kmer_idf(
            query, cd47_code, P66_SEGMENT[0] - 1, CD47_SEGMENT[0] - 1, seg_len,
            KSIZE, counts, n_documents,
        )
        # The background: the best IDF-weighted sum this segment reaches against
        # any offset of each matched protein.
        background = []
        for _, sequence in picked:
            code = encode(sequence, clusters)
            best = 0.0
            for start in range(max(0, len(code) - seg_len + 1)):
                value, _ = matched_kmer_idf(
                    query, code, P66_SEGMENT[0] - 1, start, seg_len, KSIZE,
                    counts, n_documents,
                )
                best = max(best, value)
            background.append(best)
        background = np.array(background)
        n_ge = int((background >= observed_idf).sum())
        row = merges.filter(pl.col("alphabet") == name).row(0, named=True)
        rows.append(
            {
                "alphabet": name,
                "n_classes": len(clusters),
                "unweighted_p_value": row["p_value"],
                "enrichment_observed_matched_columns": row["columns_called_a_match"],
                "enrichment_expected": row["expected_by_chance"],
                "enrichment_ratio": round(
                    row["columns_called_a_match"] / row["expected_by_chance"], 2
                ),
                "n_matched_kmers_in_segment": observed_n,
                "summed_idf_on_cd47": round(observed_idf, 2),
                "background_median_summed_idf": round(float(np.median(background)), 2),
                "background_max_summed_idf": round(float(background.max()), 2),
                "n_background_ge": n_ge,
                "idf_weighted_p_value": n_ge / len(picked),
            }
        )

        # And the loop on its own: how rare are the k-mers sitting on it?
        loop_kmers = [
            query[i : i + KSIZE]
            for i in range(loop[0] - 1 - KSIZE + 1, loop[1])
            if i >= 0 and i + KSIZE <= len(query) and query[i : i + KSIZE].min() >= 0
        ]
        seg_kmers = [
            query[i : i + KSIZE]
            for i in range(P66_SEGMENT[0] - 1, P66_SEGMENT[1] - KSIZE + 1)
            if query[i : i + KSIZE].min() >= 0
        ]
        flank_kmers = [
            km for km in seg_kmers
            if not any(np.array_equal(km, lk) for lk in loop_kmers)
        ]
        loop_rows.append(
            {
                "alphabet": name,
                "n_classes": len(clusters),
                "loop_kmers": len(loop_kmers),
                "loop_median_idf": round(
                    float(np.median([idf(k.tobytes(), counts, n_documents)
                                     for k in loop_kmers])), 2
                ),
                "flank_kmers": len(flank_kmers),
                "flank_median_idf": round(
                    float(np.median([idf(k.tobytes(), counts, n_documents)
                                     for k in flank_kmers])), 2
                ) if flank_kmers else None,
                "max_possible_idf": round(math.log(n_documents / 1), 2),
                "loop_kmers_absent_from_cd47": sum(
                    1
                    for km in loop_kmers
                    if not any(
                        np.array_equal(km, cd47_code[j : j + KSIZE])
                        for j in range(len(cd47_code) - KSIZE + 1)
                    )
                ),
            }
        )

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "32_idf_weighted_segment.csv")
    loops = pl.DataFrame(loop_rows)
    loops.write_csv(OUTDIR / "33_loop_kmer_idf.csv")

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(240)
    print(f"\nThree metrics on the segment, k = {KSIZE}:")
    print(table)
    print(f"\nHow rare the loop's own k-mers are, against the segment's flanks:")
    print(loops)
    print(f"\nAlphabets where IDF weighting changes the verdict at 0.05: "
          f"{table.filter((pl.col('unweighted_p_value') < 0.05) != (pl.col('idf_weighted_p_value') < 0.05)).height}"
          f" of {table.height}")
    print(f"\nWrote {OUTDIR / '32_idf_weighted_segment.csv'}")


if __name__ == "__main__":
    main()
