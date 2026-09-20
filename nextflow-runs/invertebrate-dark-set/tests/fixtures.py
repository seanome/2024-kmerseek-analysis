"""A small synthetic run shaped like the Botryllus one, for the report tests.

Every file the report reads is written here in the shape the pipeline scripts write it,
with the Botryllus headline numbers (45_339 proteins, 20_448 dark, hp_thomas_dill2 k23
reaching everything, protein20 k10 reaching 169) and a smaller per-protein table.
"""
import json
import random
from pathlib import Path

import polars as pl

BHF = "FUN008084_FUN008084"


def registry(path: Path, *, with_focus: bool = True) -> Path:
    row = {"taxon_id": "30301", "name": "Botryllus schlosseri", "swissprot_reviewed": 0,
           "annotate_clade": "Ascidiacea", "annotate_query": True}
    if with_focus:
        row["focus_proteins"] = {BHF: "BHF, the Botryllus histocompatibility factor"}
    path.write_text(json.dumps({"botryllus": row}))
    return path


def dark_summary(path: Path, *, with_focus: bool = True) -> Path:
    s = {"species": "botryllus", "proteins_in_proteome": 45_339,
         "proteins_placed_by_any_arm": 24_891, "proteins_dark": 20_448,
         "fraction_dark": 0.451, "evalue_call": 0.001,
         "proteins_placed_per_arm": {"phmmer": 22_100, "jackhmmer": 24_300, "mmseqs2": 23_900},
         "raw_hit_rows": 3_412_009}
    if with_focus:
        s["focus_proteins"] = {BHF: {
            "what": "BHF, the Botryllus histocompatibility factor", "in_proteome": True,
            "placed": False,
            "per_arm": {"phmmer": {"best_evalue": 0.42, "placed": False},
                        "jackhmmer": {"best_evalue": 0.011, "placed": False},
                        "mmseqs2": {"best_evalue": None, "placed": False}}}}
    path.write_text(json.dumps(s))
    return path


def reference_summary(path: Path) -> Path:
    path.write_text(json.dumps({"excluded_clade": "Ascidiacea", "entries_kept": 572_580,
                                "entries_excluded": 120}))
    return path


def run_params(path: Path) -> Path:
    path.write_text(json.dumps({"query_chunk_size": 2000, "evalue_report": 10.0,
                                "jackhmmer_iterations": 3, "mmseqs2_sensitivity": 7,
                                "min_region_score": 1.3, "max_query_pvalue": 0.05,
                                "min_shared_kmers": 2}))
    return path


def gain_json(path: Path, *, scores: bool = True, shuffled: bool = True,
              with_focus: bool = True) -> Path:
    dark_n, placed_n = 20_448, 24_891
    rows = []
    for alphabet, k, on, off, p_on, p_off, sh_on, sh_off, at_on, at_placed in [
        ("hp_thomas_dill2", 23, 20_448, 20_448, 24_891, 24_891, 20_448, 20_448,
         {"1.3": 20_448, "3.0": 9_800, "10.0": 610}, {"1.3": 24_891, "3.0": 19_900, "10.0": 8_100}),
        ("protein20", 10, 169, 173, 21_300, 21_350, 12, 14,
         {"1.3": 169, "3.0": 101, "10.0": 20}, {"1.3": 21_300, "3.0": 19_800, "10.0": 12_400}),
    ]:
        for mask, d, p, sh in ((True, on, p_on, sh_on), (False, off, p_off, sh_off)):
            row = {"species": "botryllus", "alphabet": alphabet, "ksize": k,
                   "low_complexity_mask": mask, "dark_proteins": dark_n, "dark_reached": d,
                   "fraction_dark_reached": round(d / dark_n, 4), "placed_proteins": placed_n,
                   "placed_reached": p, "fraction_placed_reached": round(p / placed_n, 4),
                   "queries_with_any_region": d + p}
            if scores:
                row["dark_reached_at"] = at_on
                row["placed_reached_at"] = at_placed
            if shuffled:
                row["shuffled_dark_reached"] = sh
                row["fraction_shuffled_dark_reached"] = round(sh / dark_n, 4)
            rows.append(row)
    pairs = [{"alphabet": "hp_thomas_dill2", "ksize": 23, "dark_reached_mask_on": 20_448,
              "dark_reached_mask_off": 20_448, "lost_to_masking": 0},
             {"alphabet": "protein20", "ksize": 10, "dark_reached_mask_on": 169,
              "dark_reached_mask_off": 173, "lost_to_masking": 4}]
    s = {"species": "botryllus", "dark_proteins": dark_n, "placed_proteins": placed_n,
         "thresholds": [1.3, 3.0, 10.0] if scores else [],
         "has_shuffled_control": shuffled, "by_combo": rows, "mask_pairs": pairs}
    if with_focus:
        s["focus_proteins"] = {BHF: {
            "what": "BHF, the Botryllus histocompatibility factor", "dark": True,
            "per_combo": {
                "hp_thomas_dill2 k23 lctrue": {"alphabet": "hp_thomas_dill2", "ksize": 23,
                                               "low_complexity_mask": True, "reached": True,
                                               "best_region_score": 4.7, "best_region": [812, 1_004],
                                               "shuffled_reached": True if shuffled else None},
                "hp_thomas_dill2 k23 lcfalse": {"alphabet": "hp_thomas_dill2", "ksize": 23,
                                                "low_complexity_mask": False, "reached": True,
                                                "best_region_score": 5.1, "best_region": [812, 1_004],
                                                "shuffled_reached": True if shuffled else None},
                "protein20 k10 lctrue": {"alphabet": "protein20", "ksize": 10,
                                         "low_complexity_mask": True, "reached": False,
                                         "best_region_score": None, "best_region": None,
                                         "shuffled_reached": False if shuffled else None},
                "protein20 k10 lcfalse": {"alphabet": "protein20", "ksize": 10,
                                          "low_complexity_mask": False, "reached": False,
                                          "best_region_score": None, "best_region": None,
                                          "shuffled_reached": False if shuffled else None},
            }}}
    path.write_text(json.dumps(s))
    return path


def length_products(parquet: Path, summary: Path, n: int = 2_000) -> tuple[Path, Path]:
    rng = random.Random(7)
    rows = []
    for i in range(n):
        dark = i % 100 < 45
        length = int(rng.lognormvariate(5.0 if dark else 5.9, 0.6))
        rows.append({"accession": f"FUN{i:06d}_FUN{i:06d}", "species": "botryllus",
                     "group": "dark" if dark else "placed", "length": max(length, 30)})
    rows.append({"accession": BHF, "species": "botryllus", "group": "dark", "length": 1_366})
    pl.DataFrame(rows).write_parquet(parquet)
    summary.write_text(json.dumps({
        "species": "botryllus", "proteins_in_proteome": 45_339, "proteins_dark": 20_448,
        "proteins_placed": 24_891, "fraction_dark": 0.451,
        "dark": {"n": 20_448, "median": 151.0, "p25": 107.0, "p75": 245.0,
                 "fraction_under_50aa": 0.0, "fraction_under_100aa": 0.18},
        "placed": {"n": 24_891, "median": 377.0, "p25": 235.0, "p75": 588.0,
                   "fraction_under_50aa": 0.0, "fraction_under_100aa": 0.02},
        "median_ratio_dark_over_placed": 0.4005,
        "mann_whitney_u": {"u_statistic": 99_000_000.0, "p_value": 0.0,
                           "alternative": "two-sided",
                           "common_language_effect_size": 0.195,
                           "rank_biserial_correlation": -0.61}}))
    return parquet, summary


def disorder_products(parquet: Path, summary: Path, n: int = 2_000) -> tuple[Path, Path]:
    rng = random.Random(11)
    rows = []
    for i in range(n):
        dark = i % 100 < 45
        v = min(0.999, max(0.0, rng.gauss(0.37 if dark else 0.21, 0.15)))
        rows.append({"accession": f"FUN{i:06d}_FUN{i:06d}", "species": "botryllus",
                     "is_dark": dark, "mean_disorder_metapredict": v})
    rows.append({"accession": BHF, "species": "botryllus", "is_dark": True,
                 "mean_disorder_metapredict": 0.512})
    pl.DataFrame(rows).write_parquet(parquet)
    summary.write_text(json.dumps({
        "species": "botryllus", "metric": "mean_disorder_metapredict",
        "dark": {"n": 20_448, "median": 0.367, "q25": 0.25, "q75": 0.51, "mean": 0.38},
        "placed": {"n": 24_891, "median": 0.206, "q25": 0.13, "q75": 0.31, "mean": 0.23},
        "mannwhitneyu_u": 300_000_000.0, "mannwhitneyu_p": 0.0, "rank_biserial": 0.42,
        "note": None}))
    return parquet, summary


def run_dir(tmp: Path, **kw) -> dict[str, Path]:
    """Every input, keyed by the builder's argument name."""
    tmp.mkdir(parents=True, exist_ok=True)
    extra = tmp / "extras"
    extra.mkdir(exist_ok=True)
    length_products(extra / "botryllus_length_comparison.parquet", extra / "botryllus_length_summary.json")
    disorder_products(extra / "botryllus_disorder.parquet", extra / "botryllus_disorder_summary.json")
    return {
        "registry": registry(tmp / "species_metadata.json", with_focus=kw.get("with_focus", True)),
        "dark_summary": dark_summary(tmp / "botryllus_dark_summary.json", with_focus=kw.get("with_focus", True)),
        "reference_summary": reference_summary(tmp / "reference_summary.json"),
        "run_params": run_params(tmp / "run_params.json"),
        "gain_json": gain_json(extra / "botryllus_kmerseek_dark_gain.json",
                               scores=kw.get("scores", True), shuffled=kw.get("shuffled", True),
                               with_focus=kw.get("with_focus", True)),
        "extra_dir": extra,
    }
