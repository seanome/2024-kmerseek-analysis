#!/usr/bin/env python3
"""Is there any alphabet, k-size or metric that puts CD47 first for P66?

Every test so far has been pairwise: P66 against CD47, with a background of
matched proteins to say whether the score is unusual. That is not the question a
search tool answers. A search tool ranks a whole database, so the question is
whether CD47 ever comes out on top of the human proteome for a P66 query, under
any setting.

This searches the settings for one, and then asks the only question that makes a
hit worth anything: **how often does the same search put an arbitrary protein
first?** If sweeping settings can promote any protein to rank 1 somewhere, then
CD47 reaching rank 1 somewhere says nothing about CD47.

Two queries, both favourable to the claim by construction:

  * **the segment**, P66 197-232, the stretch the earlier tests selected because
    it already scores well against CD47;
  * **the loop**, P66 202-208, the stretch the claim is actually about.

Two metrics that are genuinely different, and a note on why the others are not:

  * **best ungapped match count** over the query's own length. Within one
    alphabet and one query length, match count, enrichment (observed over
    expected), an ungapped +1/-penalty score and a Karlin-Altschul E-value are
    all monotone functions of each other, so they produce the *same ranking*
    and the same ties. One number stands for all four;
    scripts/nb276_metric_equivalence.py computes the four over the real proteome
    and compares the rank vectors element by element, so this is checked rather
    than assumed.
  * **summed IDF of the matching k-mers**, which reorders the ranking because it
    weights a rare k-mer above a common one.

19 alphabets x 2 queries x 2 metrics = 76 settings. For each, CD47's rank among
the 19_705 canonical human proteins is reported, together with how many proteins
tie with it and how many sit at rank 1, because a metric that ties half the
proteome has no rank 1 worth having.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_can_cd47_rank_first.py
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
from nb276_hsdm17_prespecified_test import HUMAN_PROTEOME, read_fasta  # noqa: E402
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    SEED,
    resolve_uniprot,
)
from nb276_why_combining_hurts import P66_SEGMENT  # noqa: E402

#: k for the IDF-weighted metric. 4 is the longest run of matching columns the
#: segment has, so it is the only k at which an exact k-mer can be placed in it
#: at all; a larger k scores every protein zero and ranks nothing.
IDF_KSIZE = 4
#: Pretend targets for the control: how many arbitrary human proteins get the
#: same sweep, to see how often any protein can be promoted to rank 1.
N_PRETEND_TARGETS = 200


def encode_all(sequences: list[str], clusters: list[str]) -> list[np.ndarray]:
    table = np.full(128, -1, dtype=np.int16)
    for i, cl in enumerate(clusters):
        for residue in cl:
            table[ord(residue)] = i
    return [
        table[np.frombuffer(s.encode(), dtype=np.uint8)] for s in sequences
    ]


def best_match_counts(query: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    """For each target, the most columns any offset of the query matches."""
    length = len(query)
    out = np.zeros(len(targets), dtype=np.int32)
    for i, code in enumerate(targets):
        if len(code) < length:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(code, length)
        out[i] = int((windows == query).sum(axis=1).max())
    return out


def best_summed_idf(
    query: np.ndarray,
    targets: list[np.ndarray],
    k: int,
    weights: np.ndarray,
) -> np.ndarray:
    """For each target, the best summed IDF of the query's matching k-mers.

    A k-mer starting at query position j matches at a given offset only if all k
    of its columns match there, so the k-mer indicator is a sliding "all" over
    the column-match matrix and the score is that indicator times the weights.
    Written as array operations because the obvious nested loop is three orders
    of magnitude too slow over 19_705 proteins.
    """
    length = len(query)
    out = np.zeros(len(targets))
    for i, code in enumerate(targets):
        if len(code) < length:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(code, length)
        columns = windows == query  # (offsets, length)
        kmers = np.lib.stride_tricks.sliding_window_view(
            columns, k, axis=1
        ).all(axis=2)  # (offsets, length - k + 1)
        out[i] = float((kmers * weights).sum(axis=1).max())
    return out


def document_frequency_weights(
    targets: list[np.ndarray], query: np.ndarray, k: int
) -> np.ndarray:
    """IDF of each of the query's own k-mers, in query order.

    Only the query's k-mers are ever needed, so each one is counted with a
    single array pass per protein instead of hashing every k-mer in the proteome.
    """
    n_kmers = len(query) - k + 1
    query_kmers = np.lib.stride_tricks.sliding_window_view(query, k)
    valid = query_kmers.min(axis=1) >= 0
    counts = np.zeros(n_kmers, dtype=np.int64)
    for code in targets:
        if len(code) < k:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(code, k)
        # present[j] is whether the query's j-th k-mer occurs in this protein.
        present = (windows[:, None, :] == query_kmers[None, :, :]).all(axis=2).any(
            axis=0
        )
        counts += present
    n = len(targets)
    weights = np.log(n / (1 + counts))
    weights[~valid] = 0.0
    return weights


def rank_of(scores: np.ndarray, index: int) -> tuple[int, int, int]:
    """Rank of one entry, how many tie with it, and how many are at rank 1."""
    value = scores[index]
    better = int((scores > value).sum())
    tied = int((scores == value).sum())
    top = int((scores == scores.max()).sum())
    return better + 1, tied, top


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)

    human = read_fasta(HUMAN_PROTEOME)
    names = [n for n, s in human.items() if s.isalpha()]
    sequences = [human[n] for n in names]
    if "CD47" not in names:
        raise SystemExit("CD47 is not in the proteome under that gene symbol.")
    cd47_index = names.index("CD47")
    print(f"{len(names)} canonical human proteins; CD47 is entry {cd47_index}.")
    print(f"CD47 in the file is {len(sequences[cd47_index])} aa; the UniProt entry "
          f"is {len(cd47['sequence'])} aa.\n")

    queries = {
        "segment P66 197-232": (P66_SEGMENT[0], P66_SEGMENT[1]),
        f"loop P66 {loop[0]}-{loop[1]}": loop,
    }
    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    rng = np.random.default_rng(SEED)
    pretend = sorted(
        rng.choice(len(names), size=N_PRETEND_TARGETS, replace=False).tolist()
    )

    rows = []
    best_rank_per_pretend = {i: len(names) for i in pretend}
    for alphabet_name in sorted(alphabets):
        clusters = alphabets[alphabet_name]
        codes = encode_all(sequences, clusters)
        p66_code = encode_all([p66["sequence"]], clusters)[0]
        print(f"  {alphabet_name} ({len(clusters)} classes)...", flush=True)
        for query_name, (start, end) in queries.items():
            query = p66_code[start - 1 : end]
            counts = best_match_counts(query, codes)
            rank, tied, top = rank_of(counts, cd47_index)
            rows.append(
                {
                    "alphabet": alphabet_name,
                    "n_classes": len(clusters),
                    "query": query_name,
                    "query_length": len(query),
                    "metric": "best ungapped match count",
                    "cd47_score": float(counts[cd47_index]),
                    "best_score_in_proteome": float(counts.max()),
                    "cd47_rank": rank,
                    "n_tied_with_cd47": tied,
                    "n_at_rank_1": top,
                    "cd47_percentile": round(
                        100 * (1 - (rank - 1) / len(names)), 2
                    ),
                }
            )
            for i in pretend:
                best_rank_per_pretend[i] = min(
                    best_rank_per_pretend[i], rank_of(counts, i)[0]
                )

            if len(query) >= IDF_KSIZE:
                weights = document_frequency_weights(codes, query, IDF_KSIZE)
                idf_scores = best_summed_idf(query, codes, IDF_KSIZE, weights)
                rank, tied, top = rank_of(idf_scores, cd47_index)
                rows.append(
                    {
                        "alphabet": alphabet_name,
                        "n_classes": len(clusters),
                        "query": query_name,
                        "query_length": len(query),
                        "metric": f"best summed IDF, k={IDF_KSIZE}",
                        "cd47_score": round(float(idf_scores[cd47_index]), 3),
                        "best_score_in_proteome": round(float(idf_scores.max()), 3),
                        "cd47_rank": rank,
                        "n_tied_with_cd47": tied,
                        "n_at_rank_1": top,
                        "cd47_percentile": round(
                            100 * (1 - (rank - 1) / len(names)), 2
                        ),
                    }
                )
                for i in pretend:
                    best_rank_per_pretend[i] = min(
                        best_rank_per_pretend[i], rank_of(idf_scores, i)[0]
                    )

    table = pl.DataFrame(rows)
    table.write_csv(OUTDIR / "34_cd47_rank_across_settings.csv")
    control = pl.DataFrame(
        [
            {"protein": names[i], "best_rank_over_all_settings": r}
            for i, r in best_rank_per_pretend.items()
        ]
    ).sort("best_rank_over_all_settings")
    control.write_csv(OUTDIR / "35_pretend_target_best_ranks.csv")

    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(230)
    print(f"\n{table.height} settings tried.\n")
    print("CD47's rank among 19_705 human proteins, best settings first:")
    print(table.sort("cd47_rank").select(
        "alphabet", "n_classes", "query", "metric", "cd47_score",
        "best_score_in_proteome", "cd47_rank", "n_tied_with_cd47", "n_at_rank_1",
    ).head(25))

    best = table.sort("cd47_rank").row(0, named=True)
    print(f"\nBest setting for CD47: {best['alphabet']}, {best['query']}, "
          f"{best['metric']} -> rank {best['cd47_rank']}, tied with "
          f"{best['n_tied_with_cd47']}, {best['n_at_rank_1']} proteins at rank 1.")
    reached_1 = table.filter(pl.col("cd47_rank") == 1)
    print(f"Settings where CD47 is rank 1: {reached_1.height} of {table.height}")
    if not reached_1.is_empty():
        print(reached_1.select("alphabet", "query", "metric", "n_at_rank_1"))

    print(f"\nCONTROL: the same {table.height} settings for "
          f"{N_PRETEND_TARGETS} arbitrary human proteins.")
    print(f"  reaching rank 1 in at least one setting: "
          f"{control.filter(pl.col('best_rank_over_all_settings') == 1).height} "
          f"of {control.height}")
    print(f"  reaching the top 10 in at least one setting: "
          f"{control.filter(pl.col('best_rank_over_all_settings') <= 10).height} "
          f"of {control.height}")
    print(f"  median best rank over all settings: "
          f"{control['best_rank_over_all_settings'].median()}")
    print(f"  CD47's best rank over all settings: {best['cd47_rank']}")
    print(f"\nWrote {OUTDIR / '34_cd47_rank_across_settings.csv'}")


if __name__ == "__main__":
    main()
