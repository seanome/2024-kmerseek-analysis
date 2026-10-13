#!/usr/bin/env python3
"""Collect the numbers the notebook 276 explainer page shows, as one JSON file.

The page recomputes the match columns itself from the alphabet definitions, so
what it needs from here is the two sequences, the alphabets, and the results no
browser can derive: the ranks from the window sweeps and the structural
comparisons.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_explainer_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import (  # noqa: E402
    EXTRA_ALPHABET_CLUSTERS,
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    resolve_uniprot,
)
from nb276_why_combining_hurts import CD47_SEGMENT, P66_SEGMENT  # noqa: E402

OUT = Path(__file__).resolve().parent / "nb276_explainer_data.json"


def main() -> None:
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    alphabets = {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}

    windows = pl.read_csv(OUTDIR / "37_apriori_window_ranks.csv")
    apriori = windows.filter(~pl.col("defined_from_cd47"))
    by_length = (
        apriori.sort("cd47_rank")
        .group_by("window_length")
        .agg(
            pl.col("cd47_rank").first().alias("best_rank"),
            pl.col("alphabet").first().alias("alphabet"),
            pl.col("n_at_rank_1").first().alias("n_at_rank_1"),
            pl.col("p66_span").first().alias("p66_span"),
        )
        .sort("window_length")
    )
    circular = (
        windows.filter(
            pl.col("defined_from_cd47") & (pl.col("alphabet") == "hsdm17")
        )
        .select("cd47_score", "cd47_rank", "n_at_rank_1")
        .row(0, named=True)
    )

    face = pl.read_csv(OUTDIR / "04_cd47_sirpalpha_face.csv")
    fold = pl.read_csv(OUTDIR / "05_wholefold_claimed.csv").row(0, named=True)
    fold_bg = pl.read_csv(OUTDIR / "05_wholefold_background.csv")
    loop_rmsd = pl.read_csv(OUTDIR / "07_claimed_loop_vs_face.csv").sort(
        "backbone_rmsd_angstrom"
    ).row(0, named=True)
    loop_bg = pl.read_csv(OUTDIR / "07_background_loops_vs_face.csv")
    confidence = pl.read_csv(OUTDIR / "02_model_confidence_summary.csv").row(
        0, named=True
    )
    prespec = pl.read_csv(OUTDIR / "16_hsdm17_prespecified_result.csv").row(
        0, named=True
    )
    split = pl.read_csv(OUTDIR / "18_hsdm17_loop_vs_flanks_result.csv")
    merges = pl.read_csv(OUTDIR / "30_segment_merges_by_alphabet.csv")

    # Both models' per-residue confidence, so the page can draw them on one axis.
    # P66's is already tabulated; CD47's is read from its own model the same way.
    from nb276_p66_cd47_structure import plddt_per_residue

    p66_plddt = pl.read_csv(OUTDIR / "02_p66_plddt.csv")
    models = pl.read_csv(OUTDIR / "02_alphafold_models.csv")
    cd47_model = Path(
        models.filter(pl.col("protein") == "CD47").row(0, named=True)["pdb_path"]
    )
    cd47_plddt = plddt_per_residue(cd47_model)

    payload = {
        "plddt": {
            "p66": [round(v, 1) for v in p66_plddt["plddt"].to_list()],
            "cd47": [round(v, 1) for v in cd47_plddt["plddt"].to_list()],
            "p66_mean": round(float(p66_plddt["plddt"].mean()), 1),
            "cd47_mean": round(float(cd47_plddt["plddt"].mean()), 1),
            "p66_max": round(float(p66_plddt["plddt"].max()), 1),
            "cd47_max": round(float(cd47_plddt["plddt"].max()), 1),
            "p66_above_50": int((p66_plddt["plddt"] >= 50).sum()),
            "cd47_above_50": int((cd47_plddt["plddt"] >= 50).sum()),
            "cd47_above_70": int((cd47_plddt["plddt"] >= 70).sum()),
        },
        "heldout": pl.read_csv(OUTDIR / "40_heldout_window_test.csv").to_dicts(),
        "face_runs": pl.read_csv(OUTDIR / "39_cd47_face_runs.csv").to_dicts(),
        "p66": {
            "accession": p66["accession"],
            "length": p66["length"],
            "signal_peptide_end": offset,
            "segment_start": P66_SEGMENT[0],
            "segment": p66["sequence"][P66_SEGMENT[0] - 1 : P66_SEGMENT[1]],
            "loop_start": LOOP_MATURE_START + offset,
            "loop_end": LOOP_MATURE_END + offset,
            "loop_mature": [LOOP_MATURE_START, LOOP_MATURE_END],
        },
        "cd47": {
            "accession": cd47["accession"],
            "length": cd47["length"],
            "signal_peptide_end": cd47["signal_peptide_end"],
            "segment_start": CD47_SEGMENT[0],
            "segment": cd47["sequence"][CD47_SEGMENT[0] - 1 : CD47_SEGMENT[1]],
            "face_positions": face["cd47_uniprot_position"].to_list(),
        },
        "alphabets": [
            {
                "name": name,
                "classes": alphabets[name],
                "n_classes": len(alphabets[name]),
                "chance": float(
                    merges.filter(pl.col("alphabet") == name)[
                        "chance_a_random_pair_matches"
                    ][0]
                ),
                "corroborates": bool(
                    merges.filter(pl.col("alphabet") == name)[
                        "corroborates_at_0_05"
                    ][0]
                ),
                "p_value": float(
                    merges.filter(pl.col("alphabet") == name)["p_value"][0]
                ),
            }
            for name in sorted(alphabets, key=lambda n: (len(alphabets[n]), n))
        ],
        "windows": by_length.to_dicts(),
        "circular": circular,
        "structure": {
            "model_mean_plddt": float(confidence["p66_model_mean_plddt"]),
            "fraction_below_50": float(confidence["p66_fraction_below_50_plddt"]),
            "cd47_model_mean_plddt": float(confidence["cd47_model_mean_plddt"]),
            "tm_score": float(fold["tm_score_norm_target"]),
            "tm_aligned": int(fold["aligned_length"]),
            "tm_rmsd": float(fold["rmsd_angstrom"]),
            "ig_min": float(fold_bg["tm_score_norm_target"].min()),
            "ig_max": float(fold_bg["tm_score_norm_target"].max()),
            "ig_median": float(fold_bg["tm_score_norm_target"].median()),
            "ig_n": fold_bg.height,
            "loop_rmsd": float(loop_rmsd["backbone_rmsd_angstrom"]),
            "loop_bg_median": float(loop_bg["backbone_rmsd_angstrom"].median()),
            "loop_bg_best": float(loop_bg["backbone_rmsd_angstrom"].min()),
            "loop_bg_n": loop_bg.height,
            "n_face": face.height,
        },
        "prespecified": {
            "observed": float(prespec["observed_score"]),
            "n_background": int(prespec["n_background"]),
            "n_reaching_loop": int(prespec["background_reaching_loop"]),
            "n_ge": int(prespec["n_background_score_ge_observed"]),
            "loop_observed": float(
                split.filter(pl.col("statistic") == "loop columns only")["observed"][0]
            ),
            "loop_bg_beating": int(
                split.filter(pl.col("statistic") == "loop columns only")[
                    "n_background_ge_among_those_reaching"
                ][0]
            ),
            "flank_observed": float(
                split.filter(pl.col("statistic") == "flanking columns only")[
                    "observed"
                ][0]
            ),
        },
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"Wrote {OUT} ({len(OUT.read_text())} bytes)")
    print(f"  {len(payload['alphabets'])} alphabets, "
          f"{len(payload['windows'])} window lengths")
    print(f"  circular window: {payload['circular']}")
    for row in payload["windows"]:
        print(f"  {row['window_length']:>3} aa ({row['p66_span']}): best rank "
              f"{row['best_rank']} in {row['alphabet']}, "
              f"{row['n_at_rank_1']} at rank 1")


if __name__ == "__main__":
    main()
