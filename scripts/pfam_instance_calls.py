#!/usr/bin/env python3
"""Stage 1 of the Pfam candidate table: every tool's best call on every human Pfam instance.

For one (setting or tool, species) input file of the midi-plus run, this writes one row per
human Pfam-A domain instance that at least one call of the same family overlaps: the best such
call, its IoU, and whether it lands. Instances no call touches get no row here; stage 2
(scripts/pfam_fair_table.py) adds them back with IoU 0, so nothing is filtered on which tool
found what.

What is reused from the pipeline and from notebook 244, unchanged:

  list_jobs, transfer_keep_target   scripts/reduce_swissprot_instance_landing.py (notebook 244)
  load_regions, dedup_fragment_regions
                                    nextflow-runs/qfo-pfam-region-benchmark/bin/
                                    evaluate_domain_calls.py (kmerseek: Bonferroni p < 0.05,
                                    ranked by region_enrichment; Foldseek and Reseek fragment
                                    dedup at IoU 0.9)

Transfer is the pipeline's rule: a call takes the Pfam family of every target domain it covers
by at least half of that domain (the pipeline's own overlap arithmetic, unchanged, because that
decides the label and is the run's definition of a Pfam call).

What is different from notebook 244, on purpose:

  truth         results/truth/human_domain_truth.parquet (Pfam-A, with the pipeline's
                selection/heldout split by family), not the Swiss-Prot feature key.
  IoU           closed, 1-based, the same for every tool: residues in both / residues in
                either. Pfam envelopes and aligner calls are 1-based closed already; kmerseek
                region_start is 0-based, so 1 is added (as notebook 244's export does).
                The pipeline's overlap_expr takes a length as end - start, which drops a
                residue from Pfam envelopes and aligner calls but not from kmerseek calls;
                that is why the IoU is recomputed here.
  best call     the overlapping call with the highest closed IoU (ties: score, then target
                accession, then coordinates), so the choice does not depend on row order.

Runs as a SLURM array: task i of n takes every n-th input file. A finished (arm, species) is
skipped, so resubmitting after a failure redoes only what is missing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reduce_swissprot_instance_landing as land  # noqa: E402

INSIDE_MIN = land.INSIDE_MIN  # 0.8: share of the call inside the domain
COVER_MIN = land.COVER_MIN  # 0.3: share of the domain the call covers


def closed_iou_columns(qs: str, qe: str, fs: str, fe: str) -> list[pl.Expr]:
    """Overlap, inside, cover and IoU on closed 1-based intervals."""
    ov = (pl.min_horizontal(qe, fe) - pl.max_horizontal(qs, fs) + 1).clip(lower_bound=0)
    call_len = pl.col(qe) - pl.col(qs) + 1
    dom_len = pl.col(fe) - pl.col(fs) + 1
    return [
        ov.alias("overlap"),
        (ov / call_len).alias("inside"),
        (ov / dom_len).alias("cover"),
        (ov / (call_len + dom_len - ov)).alias("iou"),
    ]


def with_is_point(df: pl.DataFrame | pl.LazyFrame):
    """Pfam tables have no is_point column; the transfer helper filters on it."""
    names = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    return df if "is_point" in names else df.with_columns(is_point=pl.lit(False))


OUT_SCHEMA = {
    "query_acc": pl.Utf8, "pfam_id": pl.Utf8, "true_start": pl.Int64, "true_end": pl.Int64,
    "n_overlapping_calls": pl.UInt32, "n_overlapping_targets": pl.UInt32, "any_landed": pl.Boolean,
    "best_qstart": pl.Int64, "best_qend": pl.Int64, "best_target_acc": pl.Utf8,
    "best_tstart": pl.Int64, "best_tend": pl.Int64, "best_score": pl.Float64,
    "best_iou": pl.Float64, "best_inside": pl.Float64, "best_cover": pl.Float64,
    "best_landed": pl.Boolean, "arm": pl.Utf8, "tool": pl.Utf8, "species": pl.Utf8,
}


def reduce_one(job: dict, truth: pl.DataFrame, domain_map: pl.LazyFrame, ev,
               min_overlap: float) -> pl.DataFrame:
    regions = ev.load_regions(job["path"], False, rank_by="region_enrichment",
                              max_bonferroni_p=0.05)
    if regions is None:  # an empty result file: a real outcome, written with the full schema
        return pl.DataFrame(schema=OUT_SCHEMA)
    if job["tool"] in land.FRAGMENT_TOOLS:
        regions = ev.dedup_fragment_regions(regions.collect(), 0.9).lazy()
    calls = land.transfer_keep_target(regions, domain_map, min_overlap, ev)
    # kmerseek region_start is 0-based (half-open); every other tool is 1-based closed.
    if job["tool"] == "kmerseek":
        calls = calls.with_columns(pl.col("qstart") + 1, pl.col("tstart") + 1)
    calls = calls.collect(engine="streaming")

    inst = truth.select(
        pl.col("accession").alias("query_acc"), "pfam_id",
        pl.col("domain_start").alias("true_start"), pl.col("domain_end").alias("true_end"),
    )
    m = (
        calls.join(inst, on=["query_acc", "pfam_id"], how="inner")
        .with_columns(closed_iou_columns("qstart", "qend", "true_start", "true_end"))
        .filter(pl.col("overlap") > 0)
        .with_columns(landed=(pl.col("inside") >= INSIDE_MIN) & (pl.col("cover") >= COVER_MIN))
    )
    key = ["query_acc", "pfam_id", "true_start", "true_end"]
    order = ["iou", "score", "target_acc", "tstart", "qstart"]
    desc = [True, True, False, False, False]
    call_cols = ["qstart", "qend", "target_acc", "tstart", "tend", "score", "iou", "inside",
                 "cover", "landed"]
    out = (
        m.sort(order, descending=desc)
        .group_by(key, maintain_order=True)
        .agg(
            n_overlapping_calls=pl.len(),
            n_overlapping_targets=pl.col("target_acc").n_unique(),
            any_landed=pl.col("landed").any(),
            *[pl.col(c).first().alias(f"best_{c}") for c in call_cols],
        )
    )
    out = out.with_columns(arm=pl.lit(job["arm"]), tool=pl.lit(job["tool"]),
                           species=pl.lit(job["species"]))
    return out.select([pl.col(c).cast(d) for c, d in OUT_SCHEMA.items()])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="midi-plus results/ directory")
    ap.add_argument("--qfo-bin", type=Path, required=True,
                    help="pipeline bin/ with evaluate_domain_calls.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--only", default=None, help="regex on the input file name, for testing")
    ap.add_argument("--list", action="store_true", help="print the job count and exit")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--include-mask-off", action="store_true",
                    help="also read the mask-off kmerseek settings (stage 2 uses mask on only)")
    args = ap.parse_args()

    sys.path.insert(0, str(args.qfo_bin))
    import evaluate_domain_calls as ev  # noqa: E402

    jobs = land.list_jobs(args.results)
    if not args.include_mask_off:
        jobs = [j for j in jobs if not j["arm"].endswith("_lcFalse")]
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["path"].name)]
    if args.list:
        n_km = sum(j["tool"] == "kmerseek" for j in jobs)
        print(f"{len(jobs)} input files: {n_km} kmerseek, {len(jobs) - n_km} comparison tools")
        return
    mine = jobs[args.task:: args.n_tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    truth = pl.read_parquet(args.results / "truth" / "human_domain_truth.parquet")
    maps: dict[str, pl.LazyFrame] = {}
    for j in mine:
        stem = f"{j['arm']}.{j['species']}"
        out = args.out_dir / f"{stem}.pfam_calls.parquet"
        if out.exists():
            print(f"skip {stem}: done", flush=True)
            continue
        if j["species"] not in maps:
            maps[j["species"]] = with_is_point(pl.read_parquet(
                args.results / "truth" / f"{j['species']}_domain_map.parquet")).lazy()
        df = reduce_one(j, truth, maps[j["species"]], ev, args.min_overlap)
        ev.release_inflated(j["path"])
        tmp = out.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.rename(out)
        print(f"{stem}: {df.height} instances with an overlapping call", flush=True)


if __name__ == "__main__":
    main()
