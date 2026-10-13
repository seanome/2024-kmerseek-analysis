#!/usr/bin/env python3
"""Score the notebook-254 searches with the landing reduction notebook 244 used.

Input: the region tables scripts/run_254_extension_searches.py wrote, one per (condition,
species, arm), condition exact or extended. Every table goes through the same pipeline code
that notebook 244's numbers came from, with nothing changed:

  reduce_swissprot_instance_landing.reduce_one
      evaluate_domain_calls.load_regions  Bonferroni p < 0.05, ranked by region_enrichment
      transfer_keep_target                a region takes the type of every target feature
                                          it covers by at least half of that feature
      inside, cover, IoU per human instance, "landed" = inside >= 0.8 and cover >= 0.3

It is run twice per case and condition:

  all targets     the whole region table. land_iou is notebook 244's kmerseek_iou (the best
                  landed call over every target protein), so the new-build exact value is
                  compared with the old one on the same definition.
  case target     the rows of the case's own query and target protein only. best_iou is
                  the best call of any kind on that pair, landed or not, the way the
                  comparison tools' IoU is taken. A per-row filter commutes with
                  load_regions, so this is the same set of calls restricted to one pair.

Output: one row per case and condition, written to tables/254_extension_cases.csv.

    python scripts/score_254_extension.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin"))
import evaluate_domain_calls as ev  # noqa: E402
import hero_example_utils as he  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "reduce_landing", REPO / "scripts" / "reduce_swissprot_instance_landing.py"
)
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)

RUN_DIR = Path("/Users/olga/data/hero-244-extension")
CASES = REPO / "tables" / "244_hero_candidates.csv"
OUT = REPO / "tables" / "254_extension_cases.csv"
CONDITIONS = ("exact", "extended")
MIN_OVERLAP = 0.5
KEY = ["query", "feature_type", "feature_start", "feature_end", "species", "target"]
CALL_COLS = ["qstart", "qend", "target_acc", "tstart", "tend", "iou", "inside", "cover"]


def arm_parts(arm: str) -> tuple[str, int]:
    body = arm.removeprefix("kmerseek.").removesuffix("_lcTrue")
    alphabet, k = body.rsplit("_k", 1)
    return alphabet, int(k)


def region_path(cond: str, species: str, alphabet: str, k: int) -> Path:
    return RUN_DIR / cond / f"human_vs_{species}.{alphabet}.k{k}.lctrue.regions.parquet"


def instance_row(out: pl.DataFrame, case: dict) -> dict | None:
    hit = out.filter(
        (pl.col("query_acc") == case["query"])
        & (pl.col("pfam_id") == case["feature_type"])
        & (pl.col("true_start") == case["feature_start"])
        & (pl.col("true_end") == case["feature_end"])
    )
    if hit.height > 1:
        raise RuntimeError(f"{case['gene']}: {hit.height} instance rows")
    return hit.row(0, named=True) if hit.height else None


def call_stats(path: Path, case: dict, call: dict) -> dict:
    """region_evalue, region_ka_evalue and mismatches of one call, read off its region row."""
    t = case["target"]
    rows = (
        pl.scan_parquet(path)
        .filter(
            pl.col("query_name").str.contains(f"|{case['query']}|", literal=True)
            & pl.col("target_name").str.contains(f"|{t}|", literal=True)
            & (pl.col("region_start") == call["qstart"])
            & (pl.col("region_end") == call["qend"])
            & (pl.col("target_start") == call["tstart"])
            & (pl.col("target_end") == call["tend"])
        )
        .select(
            "region_evalue",
            "region_ka_evalue",
            "region_evalue_source",
            "region_n_mismatches",
        )
        .collect()
        .sort("region_evalue")
    )
    return rows.row(0, named=True) if rows.height else {}


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--only", default=None, help="score one species and print, for testing"
    )
    args = ap.parse_args()
    cases = pl.read_csv(CASES).filter(pl.col("admitted_under") == "all criteria")
    if cases.height != 87:
        raise SystemExit(f"expected 87 cases, found {cases.height}")
    if args.only:
        cases = cases.filter(pl.col("species") == args.only)
    truth = pl.read_parquet(he.TRUTH).filter(~pl.col("is_point"))
    maps = {
        sp: pl.read_parquet(
            he.MIDI / "truth_swissprot" / f"{sp}_domain_map.parquet"
        ).lazy()
        for sp in cases["species"].unique()
    }
    target_len = {
        sp: {a: len(s) for a, s in he.sequences(sp, set(g["target"])).items()}
        for (sp,), g in cases.group_by("species")
    }

    rows = []
    reduced: dict[tuple, pl.DataFrame] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case in cases.iter_rows(named=True):
            alphabet, k = arm_parts(case["kmerseek_chosen_arm"])
            for cond in CONDITIONS:
                path = region_path(cond, case["species"], alphabet, k)
                if not path.exists():
                    raise SystemExit(f"missing {path}")
                job = dict(
                    path=path,
                    tool="kmerseek",
                    arm=case["kmerseek_chosen_arm"],
                    species=case["species"],
                )
                key = (cond, case["species"], case["kmerseek_chosen_arm"])
                if key not in reduced:
                    reduced[key], _ = rl.reduce_one(
                        job, truth, maps[case["species"]], ev, MIN_OVERLAP
                    )
                all_t = instance_row(reduced[key], case)

                # The same table cut to the case's query and target protein.
                pair = Path(tmp) / f"{cond}.{case['query']}.{case['target']}.parquet"
                (
                    pl.scan_parquet(path)
                    .filter(
                        pl.col("query_name").str.contains(
                            f"|{case['query']}|", literal=True
                        )
                        & pl.col("target_name").str.contains(
                            f"|{case['target']}|", literal=True
                        )
                    )
                    .sink_parquet(pair)
                )
                pair_out, _ = rl.reduce_one(
                    dict(job, path=pair), truth, maps[case["species"]], ev, MIN_OVERLAP
                )
                on_t = instance_row(pair_out, case) if pair_out.height else None

                row = {k_: case[k_] for k_ in KEY}
                row["condition"] = cond
                row["all_targets_land_iou"] = all_t["land_iou"] if all_t else None
                row["all_targets_land_target"] = (
                    all_t["land_target_acc"] if all_t else None
                )
                row["all_targets_best_iou"] = all_t["best_iou"] if all_t else None
                for c in CALL_COLS:
                    row[f"call_{c}"] = on_t[f"best_{c}"] if on_t else None
                row["call_landed"] = on_t["landed"] if on_t else None
                row["target_length"] = target_len[case["species"]][case["target"]]
                if on_t:
                    call = {
                        c: on_t[f"best_{c}"]
                        for c in ("qstart", "qend", "tstart", "tend")
                    }
                    row.update(call_stats(path, case, call))
                rows.append(row)
            print(f"{case['gene']} {case['species']}: done", flush=True)

    out = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        call_length_aa=pl.col("call_qend") - pl.col("call_qstart"),
        spans_whole_target=(pl.col("call_tstart") == 0)
        & (pl.col("call_tend") == pl.col("target_length")),
    )
    if args.only:
        print(out)
        return
    out.write_csv(OUT)
    print(f"wrote {out.height} rows to {OUT}")


if __name__ == "__main__":
    main()
