#!/usr/bin/env python3
"""One row per alphabet: everything this notebook found for P66 against CD47.

The results are spread over several tables because they were computed by
different tests. This gathers them into one table so every alphabet can be read
at once, and fills the one gap: the eight alphabets that write the loop as a
single repeated class were left out of the seed-and-extend sweep, because at
penalty 0.5 a seed there matches every polar stretch and the run does not finish.
They are run here at penalties 1.0 and 2.0, where extension dies fast enough to
be tractable, so the table has no blanks.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_all_alphabets_summary.py
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
from nb276_seed_extend_loop import (  # noqa: E402
    SEED_LENGTHS,
    UNIFORM_ON_THE_LOOP,
    best_loop_segment,
    encode,
)

#: Penalties used for the uniform alphabets. 0.5 is left out: there a seed matches
#: every polar stretch and the extension runs through whole proteins.
UNIFORM_PENALTIES = (1.0, 2.0)
#: Shuffles for those, enough to say whether the score is ordinary.
N_SHUFFLE = 50
#: Settings in the original sweep, for the Bonferroni threshold it is judged at.
N_SETTINGS_IN_SWEEP = 132


def run_uniform_alphabets() -> pl.DataFrame:
    """Seed-and-extend for the eight alphabets the sweep left out."""
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop0 = range(LOOP_MATURE_START + offset - 1, LOOP_MATURE_END + offset)

    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}
    rng = np.random.default_rng(SEED)
    rows = []
    for name in sorted(UNIFORM_ON_THE_LOOP):
        clusters = alphabets[name]
        query = encode(p66["sequence"], clusters)
        target = encode(cd47["sequence"], clusters)
        print(f"  {name} ({len(clusters)} classes)...", flush=True)
        shuffled = []
        for _ in range(N_SHUFFLE):
            copy = target.copy()
            rng.shuffle(copy)
            shuffled.append(copy)
        for seed_len in SEED_LENGTHS:
            for penalty in UNIFORM_PENALTIES:
                real = best_loop_segment(query, target, loop0, seed_len, penalty)
                nulls = np.array(
                    [
                        (
                            best_loop_segment(query, c, loop0, seed_len, penalty)
                            or {"score": 0.0}
                        )["score"]
                        for c in shuffled
                    ]
                )
                rows.append(
                    {
                        "alphabet": name,
                        "n_classes": len(clusters),
                        "seed_length": seed_len,
                        "penalty": penalty,
                        "reaches_loop": real is not None,
                        "score": real["score"] if real else None,
                        "null_median": float(np.median(nulls)),
                        "p_value_vs_shuffled_cd47": float(
                            (nulls >= real["score"]).mean()
                        )
                        if real
                        else None,
                        "segment_length": (real["q_end"] - real["q_start"])
                        if real
                        else None,
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    uniform_path = OUTDIR / "19_seed_extend_uniform_alphabets.csv"
    if uniform_path.exists():
        uniform = pl.read_csv(uniform_path)
        print(f"Reusing {uniform_path}")
    else:
        print("Running seed-and-extend for the eight uniform alphabets:")
        uniform = run_uniform_alphabets()
        uniform.write_csv(uniform_path)

    reach = pl.read_csv(OUTDIR / "09_loop_kmer_reach_all_alphabets.csv")
    sweep = pl.read_csv(OUTDIR / "11_seed_extend_loop.csv")
    face = pl.read_csv(OUTDIR / "13_seed_extend_face_overlap.csv")
    pairs = pl.read_csv(OUTDIR / "10_loop_kmer_pairs.csv")

    # Exact-k-mer columns, per alphabet.
    exact = reach.group_by("alphabet", "n_classes").agg(
        pl.col("loop_class_string").first(),
        pl.col("n_distinct_classes_in_loop").first(),
        pl.col("n_covering_kmers_found_in_cd47").sum().alias("loop_kmers_in_cd47"),
    )
    highest_k = (
        reach.filter(pl.col("n_covering_kmers_found_in_cd47") > 0)
        .group_by("alphabet")
        .agg(pl.col("ksize").max().alias("highest_k_with_a_loop_kmer"))
    )
    best_pair = (
        pairs.sort("n_identical_residues", descending=True)
        .group_by("alphabet")
        .agg(
            pl.col("p66_residues").first().alias("best_pair_p66"),
            pl.col("cd47_residues").first().alias("best_pair_cd47"),
            pl.col("ksize").first().alias("best_pair_k"),
            pl.col("n_identical_residues").first().alias("best_pair_identities"),
        )
    )

    # Seed-and-extend columns, per alphabet: the strongest setting it has.
    all_se = pl.concat(
        [
            sweep.select("alphabet", "n_classes", "seed_length", "penalty",
                         "reaches_loop", "score", "p_value_vs_shuffled_cd47"),
            uniform.select("alphabet", "n_classes", "seed_length", "penalty",
                           "reaches_loop", "score", "p_value_vs_shuffled_cd47"),
        ],
        how="vertical",
    )
    se = (
        all_se.filter(pl.col("reaches_loop"))
        .sort("p_value_vs_shuffled_cd47")
        .group_by("alphabet")
        .agg(
            pl.len().alias("seed_extend_settings_reaching_loop"),
            pl.col("score").max().alias("seed_extend_best_score"),
            pl.col("p_value_vs_shuffled_cd47").min().alias("seed_extend_best_p"),
        )
    )
    face_best = (
        face.sort("n_face_residues_touched", descending=True)
        .group_by("alphabet")
        .agg(pl.col("n_face_residues_touched").first().alias("max_face_residues_hit"))
    )

    table = (
        exact.join(highest_k, on="alphabet", how="left")
        .join(best_pair, on="alphabet", how="left")
        .join(se, on="alphabet", how="left")
        .join(face_best, on="alphabet", how="left")
        .with_columns(
            pl.col("loop_kmers_in_cd47").fill_null(0),
            pl.col("seed_extend_settings_reaching_loop").fill_null(0),
            pl.col("max_face_residues_hit").fill_null(0),
            pl.col("alphabet").is_in(list(UNIFORM_ON_THE_LOOP))
            .alias("loop_is_one_repeated_class"),
        )
        .sort("n_classes", "alphabet")
    )
    table.write_csv(OUTDIR / "20_all_alphabets_summary.csv")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(260)
    print("\n=== Exact k-mers on the loop, per alphabet ===")
    print(table.select(
        "alphabet", "n_classes", "loop_class_string", "n_distinct_classes_in_loop",
        "loop_kmers_in_cd47", "highest_k_with_a_loop_kmer",
        "best_pair_k", "best_pair_p66", "best_pair_cd47", "best_pair_identities",
    ))
    print("\n=== Seed-and-extend on the loop, per alphabet ===")
    print(table.select(
        "alphabet", "n_classes", "loop_is_one_repeated_class",
        "seed_extend_settings_reaching_loop", "seed_extend_best_score",
        "seed_extend_best_p", "max_face_residues_hit",
    ))
    print(f"\nBonferroni threshold the sweep is judged at: "
          f"p < {0.05 / N_SETTINGS_IN_SWEEP:.5f}")
    print(f"Alphabets with any exact loop k-mer in CD47: "
          f"{(table['loop_kmers_in_cd47'] > 0).sum()} of {table.height}")
    print(f"Alphabets where seed-and-extend reaches the loop at some setting: "
          f"{(table['seed_extend_settings_reaching_loop'] > 0).sum()} of {table.height}")
    print(f"\nWrote {OUTDIR / '20_all_alphabets_summary.csv'}")


if __name__ == "__main__":
    main()
