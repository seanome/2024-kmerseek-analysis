#!/usr/bin/env python3
"""Reduce the midi-plus region tables to one row per (arm, species, Swiss-Prot instance).

Notebook 244 needs, for every kmerseek arm and every comparison tool, which human
Swiss-Prot feature instance each call landed on and which target protein it came from.
The pipeline's per-call tables (calls_swissprot/) drop the target protein, so this script
redoes the pipeline's own transfer step with the target kept, using the pipeline's own
functions from bin/evaluate_domain_calls.py:

  load_regions            kmerseek Bonferroni filter (p < 0.05) and region_enrichment score,
                          the run's defaults; aligner TSVs parsed the same way
  dedup_fragment_regions  Foldseek and Reseek only, fragment IoU 0.9, as score_one does
  overlap_expr            the overlap arithmetic every IoU and cover in the run uses

Transfer is the pipeline's rule: a region takes the type of every target Swiss-Prot
feature it covers by at least min_overlap (0.5) of that feature's length. The call is the
query-side interval of the region, labelled with that type.

Per call and human instance of the same type on the same query protein:

  inside = overlap / call length      (share of the call inside the instance)
  cover  = overlap / instance length  (share of the instance the call covers)
  iou    = overlap / union

"Landed" is inside >= 0.8 and cover >= 0.3 (notebook 244, criterion 2). Point features
(is_point) are left out: a 2-residue instance cannot hold 80% of any call.

Outputs, one pair of parquets per (arm, species) under --out-dir:

  <arm>.<species>.instances.parquet  one row per human instance any call overlapped
  <arm>.<species>.calls_by_type.parquet  call counts per feature type

Runs as a SLURM array: task i of n takes every n-th input file, sorted.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import polars as pl

SPECIES = ["mouse", "chicken", "zebrafish", "ciona", "fly", "worm", "yeast", "arabidopsis", "ecoli"]
#: Comparison tools and the one variant each ran in midi-plus, named as in calls_swissprot/.
BASELINES = {"foldseek": "3di_aa", "prostt5": "3di_from_seq", "reseek": "verysensitive",
             "hmmer3_phmmer": "default", "mmseqs2_seqseq": "s7", "mmseqs2_iterative": "s7"}
FRAGMENT_TOOLS = {"foldseek", "reseek"}
INSIDE_MIN = 0.8
COVER_MIN = 0.3

KM_RE = re.compile(r"^human_vs_(?P<sp>[a-z]+)\.(?P<alpha>.+)\.k(?P<k>\d+)\.lc(?P<lc>true|false)\.regions\.parquet$")
BL_RE = re.compile(r"^human_vs_(?P<sp>[a-z]+)\.(?P<tool>[a-z0-9_]+)\.tsv\.gz$")


def list_jobs(results: Path) -> list[dict]:
    jobs = []
    for p in sorted((results / "kmerseek").glob("human_vs_*.regions.parquet")):
        m = KM_RE.match(p.name)
        if m and m["sp"] in SPECIES:
            lc = "True" if m["lc"] == "true" else "False"
            jobs.append({"path": p, "species": m["sp"], "tool": "kmerseek",
                         "arm": f"kmerseek.{m['alpha']}_k{m['k']}_lc{lc}"})
    for tool in BASELINES:
        for p in sorted((results / "regions" / tool).glob("human_vs_*.tsv.gz")):
            m = BL_RE.match(p.name)
            if m and m["sp"] in SPECIES and m["tool"] == tool:
                jobs.append({"path": p, "species": m["sp"], "tool": tool,
                             "arm": f"{tool}.{BASELINES[tool]}"})
    return jobs


def transfer_keep_target(regions: pl.LazyFrame, domain_map: pl.LazyFrame,
                         min_overlap: float, ev) -> pl.LazyFrame:
    """evaluate_domain_calls.transfer_domains with the target protein and coordinates kept."""
    joined = regions.join(
        domain_map.filter(~pl.col("is_point")).select(
            pl.col("accession").alias("target_acc"), "pfam_id",
            pl.col("domain_start").alias("t_feat_start"),
            pl.col("domain_end").alias("t_feat_end"),
        ),
        on="target_acc", how="inner",
    )
    return (
        joined.with_columns(
            ev.overlap_expr("tstart", "tend", "t_feat_start", "t_feat_end").alias("t_overlap"),
            (pl.col("t_feat_end") - pl.col("t_feat_start")).alias("t_feat_len"),
        )
        .filter(pl.col("t_overlap") >= min_overlap * pl.col("t_feat_len"))
        .select("query_acc", "pfam_id", "qstart", "qend", "target_acc", "tstart", "tend",
                "t_feat_start", "t_feat_end", "score")
    )


def reduce_one(job: dict, truth: pl.DataFrame, domain_map: pl.LazyFrame, ev,
               min_overlap: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    regions = ev.load_regions(job["path"], False, rank_by="region_enrichment",
                              max_bonferroni_p=0.05)
    if regions is None:
        return pl.DataFrame(), pl.DataFrame()
    if job["tool"] in FRAGMENT_TOOLS:
        regions = ev.dedup_fragment_regions(regions.collect(), 0.9).lazy()
    calls = transfer_keep_target(regions, domain_map, min_overlap, ev).collect(engine="streaming")

    by_type = calls.group_by("pfam_id").agg(
        n_calls=pl.len(),
        n_query_intervals=pl.struct("query_acc", "qstart", "qend").n_unique(),
    )

    inst = truth.select(
        pl.col("accession").alias("query_acc"), "pfam_id",
        pl.col("domain_start").alias("true_start"), pl.col("domain_end").alias("true_end"),
    )
    m = (
        calls.join(inst, on=["query_acc", "pfam_id"], how="inner")
        .with_columns(ov=ev.overlap_expr("qstart", "qend", "true_start", "true_end"))
        .filter(pl.col("ov") > 0)
        .with_columns(
            inside=pl.col("ov") / (pl.col("qend") - pl.col("qstart")).clip(lower_bound=1),
            cover=pl.col("ov") / (pl.col("true_end") - pl.col("true_start")).clip(lower_bound=1),
            iou=pl.col("ov") / (pl.max_horizontal("qend", "true_end")
                                - pl.min_horizontal("qstart", "true_start")).clip(lower_bound=1),
        )
        .with_columns(landed=(pl.col("inside") >= INSIDE_MIN) & (pl.col("cover") >= COVER_MIN))
    )
    key = ["query_acc", "pfam_id", "true_start", "true_end"]
    # Total sort order so the chosen call does not depend on row order: best IoU, then
    # score, then the target accession and coordinates.
    order = ["iou", "score", "target_acc", "tstart", "qstart"]
    desc = [True, True, False, False, False]
    call_cols = ["qstart", "qend", "target_acc", "tstart", "tend", "t_feat_start",
                 "t_feat_end", "score", "iou", "inside", "cover"]
    best_any = (m.sort(order, descending=desc).group_by(key).agg(
        n_overlapping_calls=pl.len(),
        n_overlapping_targets=pl.col("target_acc").n_unique(),
        any_inside_half=(pl.col("inside") >= 0.5).any(),
        max_iou_inside_half=pl.col("iou").filter(pl.col("inside") >= 0.5).max(),
        landed=pl.col("landed").any(),
        n_landed_targets=pl.col("target_acc").filter(pl.col("landed")).n_unique(),
        *[pl.col(c).first().alias(f"best_{c}") for c in call_cols],
    ))
    best_land = (m.filter("landed").sort(order, descending=desc).group_by(key).agg(
        *[pl.col(c).first().alias(f"land_{c}") for c in call_cols],
    ))
    out = best_any.join(best_land, on=key, how="left")
    ident = dict(arm=job["arm"], tool=job["tool"], species=job["species"])
    return (out.with_columns(**{k: pl.lit(v) for k, v in ident.items()}),
            by_type.with_columns(**{k: pl.lit(v) for k, v in ident.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="midi-plus results/ directory")
    ap.add_argument("--qfo-bin", type=Path, required=True, help="pipeline bin/ with evaluate_domain_calls.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--only", default=None, help="regex on the input file name, for testing")
    ap.add_argument("--list", action="store_true", help="print the job count and exit")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args()

    sys.path.insert(0, str(args.qfo_bin))
    import evaluate_domain_calls as ev  # noqa: E402

    jobs = list_jobs(args.results)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["path"].name)]
    if args.list:
        n_km = sum(j["tool"] == "kmerseek" for j in jobs)
        print(f"{len(jobs)} input files: {n_km} kmerseek, {len(jobs) - n_km} comparison tools")
        return
    mine = jobs[args.task::args.n_tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    truth = pl.read_parquet(args.results / "truth_swissprot" / "human_swissprot_truth.parquet")
    truth = truth.filter(~pl.col("is_point"))
    maps: dict[str, pl.LazyFrame] = {}
    for j in mine:
        stem = f"{j['arm']}.{j['species']}"
        out_i = args.out_dir / f"{stem}.instances.parquet"
        out_t = args.out_dir / f"{stem}.calls_by_type.parquet"
        if out_i.exists() and out_t.exists():
            print(f"skip {stem}: done", flush=True)
            continue
        if j["species"] not in maps:
            maps[j["species"]] = pl.read_parquet(
                args.results / "truth_swissprot" / f"{j['species']}_domain_map.parquet").lazy()
        inst, by_type = reduce_one(j, truth, maps[j["species"]], ev, args.min_overlap)
        # load_regions inflates a gzipped TSV into the working directory; drop it now.
        ev.release_inflated(j["path"])
        # Written to a temporary name and renamed, so an interrupted task leaves no file the
        # skip check above would take as finished.
        for df, out in ((inst, out_i), (by_type, out_t)):
            tmp = out.with_suffix(".tmp")
            df.write_parquet(tmp)
            tmp.rename(out)
        print(f"{stem}: {inst.height} instances overlapped, "
              f"{int(inst['landed'].sum()) if inst.height else 0} landed", flush=True)


if __name__ == "__main__":
    main()
