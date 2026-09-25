#!/usr/bin/env python3
"""Score the ELM motif-transfer run: one row per (arm, target set, human ELM instance).

A case is one human ELM instance (the query side), the target set, and the target protein
the tool took the motif label from. Truth is the instance on the human protein.

For every tool's region table this redoes the steps notebook 244 used on Swiss-Prot, with
the pipeline's own functions from bin/evaluate_domain_calls.py:

  load_regions            kmerseek: every region kept (no Bonferroni filter), ranked by
                          region_mean_idf; every other tool: its own score column
  dedup_fragment_regions  Foldseek and Reseek only, fragment IoU 0.9, as score_one does
  overlap_expr            the overlap arithmetic

Coordinates are 0-based and end-exclusive throughout: kmerseek writes them that way, ELM
starts are shifted by one in scripts/fetch_elm.py, and every other tool's 1-based start
is shifted by one here.

Steps per region table:

  1. Rank each query's rows by score, highest first, with a total tie order (target
     accession, then coordinates), and keep the first L = 1000. Every tool is cut at the
     same list length here, not in the tool.
  2. Transfer: a row takes the class of every target ELM instance it covers by at least
     half of that instance's length (min_overlap 0.5, the pipeline's rule).
  3. Per human instance of the same class on the same query protein:
       cover = overlap / motif length     (share of the motif the call covers)
       inside = overlap / call length
       iou    = overlap / union           (the overlap score; recorded, never thresholded)
     "Landed" is cover >= 0.8: the call covers at least 80% of the motif. centre_offset is
     the distance in residues between the centre of the call and the centre of the motif.
  4. Exact-placement null for the chosen landed call: slide a window of the call's length
     to every position of the query protein and count the share of positions that would
     also land on the motif (p_place_query); do the same on the target protein against
     every same-class target instance on it (p_place_target). p_place is their product,
     the chance that a same-length pair of windows placed at random lands on both sides.

The Kyte-Doolittle scan of notebook 231 (swissprot_control_utils.kd_transmem_calls,
window 19, mean hydropathy > 1.6) is scored as one more arm, by position only: it carries no
label, so a scan segment lands on any human instance it covers by half.

Outputs, per (arm, target set) under --out-dir:

  <arm>.<target>.instances.parquet     one row per human instance any call overlapped
  <arm>.<target>.target_ranks.parquet  best rank of each target protein in each query's
                                       list, for the random-query control
  <arm>.<target>.length_checks.parquet kmerseek only: Spearman correlation of every
                                       ranking column with region length

Runs as a SLURM array: task i of n takes every n-th job, sorted.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

HUMAN = "Homo sapiens"
LIST_LENGTH = 1000
#: A call lands when it covers at least this share of the human motif. Scored this way
#: round, not as "80% of the call inside the feature" (notebook 244), because a call is at
#: least k residues long and most motifs are shorter than k.
COVER_MIN = 0.8
FRAGMENT_TOOLS = {"foldseek", "reseek"}
KM_SCORES = [
    "region_mean_idf",
    "region_evalue",
    "region_tfidf",
    "region_enrichment",
    "region_poisson_score",
]

KM_RE = re.compile(
    r"^human_vs_(?P<t>[a-z_]+)\.(?P<alpha>.+)\.k(?P<k>\d+)(?:\.s(?P<s>\d+))?"
    r"\.lc(?P<lc>true|false)\.regions\.parquet$"
)
BL_RE = re.compile(r"^human_vs_(?P<t>[a-z_]+)\.(?P<tool>[a-z0-9_]+)\.tsv\.gz$")


def list_jobs(results: Path, extended: bool = False) -> list[dict]:
    """Every region table under one results directory. An extended run's kmerseek arms get
    the suffix `_ext`; its comparators are not read (the extended run does not run them).
    """
    jobs = []
    ext = "_ext" if extended else ""
    for p in sorted((results / "kmerseek").glob("human_vs_*.regions.parquet")):
        m = KM_RE.match(p.name)
        if not m:
            continue
        lc = "True" if m["lc"] == "true" else "False"
        s = m["s"] or "1"
        jobs.append(
            {
                "path": p,
                "target": m["t"],
                "tool": "kmerseek",
                "arm": f"kmerseek.{m['alpha']}_k{m['k']}_s{s}_lc{lc}{ext}",
            }
        )
    if extended:
        return jobs
    for p in sorted((results / "regions").glob("*/human_vs_*.tsv.gz")):
        m = BL_RE.match(p.name)
        if m and m["tool"] == p.parent.name:
            jobs.append(
                {"path": p, "target": m["t"], "tool": m["tool"], "arm": m["tool"]}
            )
    return jobs


def read_fasta_lengths(path: Path) -> dict[str, int]:
    lengths, acc, n = {}, None, 0
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if acc:
                lengths[acc] = n
            parts = line[1:].split()[0].split("|")
            acc, n = (parts[1] if len(parts) >= 2 else parts[0]), 0
        else:
            n += len(line.strip())
    if acc:
        lengths[acc] = n
    return lengths


def read_fasta(path: Path) -> dict[str, str]:
    seqs, acc, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            parts = line[1:].split()[0].split("|")
            acc, buf = (parts[1] if len(parts) >= 2 else parts[0]), []
        else:
            buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def rank_and_cut(regions: pl.DataFrame, list_length: int) -> pl.DataFrame:
    """Rank rows within each query, best score first, total order; keep the first L."""
    order = ["query_acc", "score", "target_acc", "qstart", "qend", "tstart", "tend"]
    return (
        regions.sort(order, descending=[False, True] + [False] * 5, nulls_last=True)
        .with_columns(
            rank=pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("query_acc")
        )
        .filter(pl.col("rank") <= list_length)
    )


def place_share(
    prot_len: int, width: int, motifs: list[tuple[int, int]]
) -> float | None:
    """Share of window starts 0..prot_len-width whose window covers >= half of any motif."""
    if prot_len is None or width <= 0 or width > prot_len or not motifs:
        return None
    p = np.arange(prot_len - width + 1)
    hit = np.zeros(p.size, dtype=bool)
    for s, e in motifs:
        ov = np.clip(np.minimum(p + width, e) - np.maximum(p, s), 0, None)
        hit |= ov >= COVER_MIN * (e - s)
    return float(hit.mean())


def score_calls(
    calls: pl.DataFrame, human: pl.DataFrame, by_label: bool
) -> pl.DataFrame:
    """Join calls to human instances and measure each overlap."""
    keys = ["query_acc", "elm_class"] if by_label else ["query_acc"]
    inst = human.select(
        pl.col("accession").alias("query_acc"),
        "elm_instance",
        "elm_class",
        pl.col("start").alias("true_start"),
        pl.col("end").alias("true_end"),
    )
    if not by_label:
        calls = calls.drop("elm_class", strict=False)
    m = calls.join(inst, on=keys, how="inner")
    lo = pl.max_horizontal("qstart", "true_start")
    hi = pl.min_horizontal("qend", "true_end")
    return (
        m.with_columns(ov=(hi - lo).clip(lower_bound=0))
        .filter(pl.col("ov") > 0)
        .with_columns(
            cover=pl.col("ov") / (pl.col("true_end") - pl.col("true_start")),
            inside=pl.col("ov")
            / (pl.col("qend") - pl.col("qstart")).clip(lower_bound=1),
            iou=pl.col("ov")
            / (
                pl.max_horizontal("qend", "true_end")
                - pl.min_horizontal("qstart", "true_start")
            ),
        )
        .with_columns(
            landed=pl.col("cover") >= COVER_MIN,
            # Residues between the centre of the call and the centre of the motif.
            centre_offset=(
                (pl.col("qstart") + pl.col("qend")) / 2
                - (pl.col("true_start") + pl.col("true_end")) / 2
            ).abs(),
        )
    )


def per_instance(m: pl.DataFrame) -> pl.DataFrame:
    """One row per human instance: the best call by IoU, and the best landed call."""
    order = ["iou", "score", "target_acc", "tstart", "qstart"]
    desc = [True, True, False, False, False]
    cols = [
        "qstart",
        "qend",
        "target_acc",
        "tstart",
        "tend",
        "t_feat_start",
        "t_feat_end",
        "score",
        "rank",
        "iou",
        "cover",
        "inside",
        "centre_offset",
    ]
    cols = [c for c in cols if c in m.columns]
    best = (
        m.sort(order, descending=desc, nulls_last=True)
        .group_by("elm_instance")
        .agg(
            n_overlapping_calls=pl.len(),
            n_overlapping_targets=pl.col("target_acc").n_unique(),
            landed=pl.col("landed").any(),
            n_landed_targets=pl.col("target_acc").filter(pl.col("landed")).n_unique(),
            *[pl.col(c).first().alias(f"best_{c}") for c in cols],
        )
    )
    land = (
        m.filter("landed")
        .sort(order, descending=desc, nulls_last=True)
        .group_by("elm_instance")
        .agg(*[pl.col(c).first().alias(f"land_{c}") for c in cols])
    )
    return best.join(land, on="elm_instance", how="left")


def add_placement_null(
    out: pl.DataFrame,
    human: pl.DataFrame,
    targets: pl.DataFrame,
    lengths: dict[str, int],
) -> pl.DataFrame:
    if "land_target_acc" not in out.columns:
        return out
    h = {
        r["elm_instance"]: (r["accession"], r["start"], r["end"], r["elm_class"])
        for r in human.iter_rows(named=True)
    }
    t_motifs: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r in targets.iter_rows(named=True):
        t_motifs.setdefault((r["accession"], r["elm_class"]), []).append(
            (r["start"], r["end"])
        )
    pq, pt = [], []
    for r in out.iter_rows(named=True):
        if r["land_qstart"] is None:
            pq.append(None)
            pt.append(None)
            continue
        acc, s, e, cls = h[r["elm_instance"]]
        pq.append(
            place_share(lengths.get(acc), r["land_qend"] - r["land_qstart"], [(s, e)])
        )
        if r.get("land_tstart") is None:  # the KD scan has no target side
            pt.append(None)
        else:
            pt.append(
                place_share(
                    lengths.get(r["land_target_acc"]),
                    r["land_tend"] - r["land_tstart"],
                    t_motifs.get((r["land_target_acc"], cls), []),
                )
            )
    return out.with_columns(
        p_place_query=pl.Series(pq, dtype=pl.Float64),
        p_place_target=pl.Series(pt, dtype=pl.Float64),
    ).with_columns(
        p_place=pl.col("p_place_query") * pl.col("p_place_target").fill_null(1.0)
    )


def length_checks(path: Path) -> pl.DataFrame:
    """Spearman correlation of each kmerseek ranking column with region length."""
    lf = pl.scan_parquet(path)
    names = lf.collect_schema().names()
    present = [c for c in KM_SCORES if c in names]
    if not present:
        return pl.DataFrame()
    df = lf.select(
        (pl.col("region_end") - pl.col("region_start"))
        .cast(pl.Float64)
        .alias("region_length"),
        *[pl.col(c).cast(pl.Float64) for c in present],
    ).collect()
    rows = []
    for c in present:
        sub = df.select("region_length", c).drop_nulls().filter(pl.col(c).is_finite())
        rho = (
            sub.select(pl.corr("region_length", c, method="spearman")).item()
            if sub.height > 2
            else None
        )
        rows.append(
            {
                "metric": c,
                "spearman_vs_length": rho,
                "n_regions": sub.height,
                "n_not_finite": df.height - sub.height,
            }
        )
    return pl.DataFrame(rows)


def reduce_one(
    job: dict,
    human: pl.DataFrame,
    targets: pl.DataFrame,
    lengths: dict[str, int],
    ev,
    min_overlap: float,
    list_length: int,
):
    if job["tool"] == "kd_scan":
        calls = job["calls"].with_columns(
            target_acc=pl.lit(None, pl.String),
            tstart=pl.lit(None, pl.Int64),
            tend=pl.lit(None, pl.Int64),
            rank=pl.lit(None, pl.Int64),
        )
        m = score_calls(calls, human, by_label=False)
        return (
            add_placement_null(per_instance(m), human, targets, lengths),
            pl.DataFrame(),
            pl.DataFrame(),
        )

    if job["tool"] == "kmerseek":
        lf = ev.load_regions(
            job["path"], False, rank_by="region_mean_idf", max_bonferroni_p=None
        )
    else:
        lf = ev.load_regions(job["path"], False)
    if lf is None:
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    regions = lf.collect(engine="streaming")
    if job["tool"] != "kmerseek":
        regions = regions.with_columns(pl.col("qstart") - 1, pl.col("tstart") - 1)
    if job["tool"] in FRAGMENT_TOOLS:
        regions = ev.dedup_fragment_regions(regions, 0.9)
    regions = regions.drop_nulls(["qstart", "qend", "tstart", "tend"])
    ranked = rank_and_cut(regions, list_length)

    target_ranks = ranked.group_by("query_acc", "target_acc").agg(
        best_rank=pl.col("rank").min()
    )

    tmap = targets.select(
        pl.col("accession").alias("target_acc"),
        "elm_class",
        pl.col("start").alias("t_feat_start"),
        pl.col("end").alias("t_feat_end"),
    )
    t_ov = (
        pl.min_horizontal("tend", "t_feat_end")
        - pl.max_horizontal("tstart", "t_feat_start")
    ).clip(lower_bound=0)
    calls = ranked.join(tmap, on="target_acc", how="inner").filter(
        t_ov >= min_overlap * (pl.col("t_feat_end") - pl.col("t_feat_start"))
    )
    m = score_calls(calls, human, by_label=True)
    out = add_placement_null(per_instance(m), human, targets, lengths)
    checks = length_checks(job["path"]) if job["tool"] == "kmerseek" else pl.DataFrame()
    return out, target_ranks, checks


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results", type=Path, required=True, help="the ELM run's results/ directory"
    )
    ap.add_argument(
        "--results-extended",
        type=Path,
        default=None,
        help="the extended run's results/ directory (make run-elm-motif ELM_EXTEND=1)",
    )
    ap.add_argument(
        "--elm-dir",
        type=Path,
        required=True,
        help="holds elm_instances_tp.parquet, elm_instance_checks.parquet, elm_proteins.fasta",
    )
    ap.add_argument(
        "--qfo-bin",
        type=Path,
        required=True,
        help="pipeline bin/ with evaluate_domain_calls.py",
    )
    ap.add_argument(
        "--notebooks",
        type=Path,
        required=True,
        help="notebooks/, for swissprot_control_utils",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    ap.add_argument(
        "--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )
    ap.add_argument(
        "--only", default=None, help="regex on the input file name, for testing"
    )
    ap.add_argument("--list", action="store_true", help="print the job count and exit")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--list-length", type=int, default=LIST_LENGTH)
    args = ap.parse_args()

    sys.path.insert(0, str(args.qfo_bin))
    sys.path.insert(0, str(args.notebooks))
    import evaluate_domain_calls as ev  # noqa: E402

    inst = pl.read_parquet(args.elm_dir / "elm_instances_tp.parquet").join(
        pl.read_parquet(args.elm_dir / "elm_instance_checks.parquet").select(
            "elm_instance", "usable"
        ),
        on="elm_instance",
    )
    inst = inst.filter("usable")
    human = inst.filter(pl.col("organism") == HUMAN)
    targets = inst.filter(pl.col("organism") != HUMAN)
    lengths = read_fasta_lengths(args.elm_dir / "elm_proteins.fasta")

    jobs = list_jobs(args.results)
    if args.results_extended is not None:
        jobs += list_jobs(args.results_extended, extended=True)
    target_sets = sorted({j["target"] for j in jobs})
    for t in target_sets:
        jobs.append(
            {
                "path": Path(f"kd_scan.{t}"),
                "target": t,
                "tool": "kd_scan",
                "arm": "kd_scan.window19",
            }
        )
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["path"].name)]
    if args.list:
        n_km = sum(j["tool"] == "kmerseek" for j in jobs)
        print(
            f"{len(jobs)} jobs: {n_km} kmerseek, {len(jobs) - n_km} other "
            f"(target sets: {', '.join(target_sets)})"
        )
        return
    mine = jobs[args.task :: args.n_tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kd_calls = None
    for j in mine:
        stem = f"{j['arm']}.{j['target']}"
        outs = {
            k: args.out_dir / f"{stem}.{k}.parquet"
            for k in ("instances", "target_ranks", "length_checks")
        }
        if outs["instances"].exists():
            print(f"skip {stem}: done", flush=True)
            continue
        if j["tool"] == "kd_scan":
            if kd_calls is None:
                import swissprot_control_utils as su  # noqa: E402

                seqs = read_fasta(args.elm_dir / "elm_proteins.fasta")
                kd_calls = su.kd_transmem_calls(
                    {a: seqs[a] for a in human["accession"].unique() if a in seqs}
                ).with_columns(pl.col("qstart") - 1, score=pl.col("score"))
            j["calls"] = kd_calls.rename({"pfam_id": "elm_class"})
        out, ranks, checks = reduce_one(
            j, human, targets, lengths, ev, args.min_overlap, args.list_length
        )
        if j["tool"] not in ("kd_scan",):
            ev.release_inflated(j["path"])
        ident = {"arm": j["arm"], "tool": j["tool"], "target_set": j["target"]}
        # Written to a temporary name and renamed, so an interrupted task leaves no file the
        # skip check above would take as finished. instances is written last.
        for key in ("length_checks", "target_ranks", "instances"):
            df = {"instances": out, "target_ranks": ranks, "length_checks": checks}[key]
            df = (
                df.with_columns(**{k: pl.lit(v) for k, v in ident.items()})
                if df.height
                else df
            )
            tmp = outs[key].with_suffix(".tmp")
            df.write_parquet(tmp)
            tmp.rename(outs[key])
        n_land = int(out["landed"].sum()) if out.height else 0
        print(f"{stem}: {out.height} instances overlapped, {n_land} landed", flush=True)


if __name__ == "__main__":
    main()
