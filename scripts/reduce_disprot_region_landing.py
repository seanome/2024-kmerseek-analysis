#!/usr/bin/env python3
"""One row per (arm, target, human DisProt functional region, label rule) for notebook 251.

The task: take the functional-region label that DisProt gives a TARGET protein and put it
on a human QUERY protein. A call is a region the tool reports between the human protein
and a target protein. It carries a label when it covers at least half of a DisProt
functional region on the target (the pipeline's transfer rule, min_overlap 0.5, as
evaluate_domain_calls.transfer_domains and scripts/reduce_swissprot_instance_landing.py
apply it). The call's human interval then says where that function sits on the human
protein, and it is scored against the human protein's own DisProt region.

Two label rules, both scored:

  same_term  the target region has the SAME DisProt function term as the human region
             (flexible linker onto flexible linker)
  any_term   the target region has any DisProt function term: the call found a
             functional disordered region, not necessarily the same function

Two targets, both scored, each named by its label in the pipeline's species table:

  the nine QfO proteomes   only the target proteins DisProt annotated can pass on a
                           label, 4 (zebrafish) to 84 (yeast) proteins with the same
                           sequence as DisProt's, none in ciona
  disprot                  every non-human DisProt protein with a functional region, all
                           organisms pooled (scripts/fetch_disprot.py)

List length. Every tool is compared at the same list length: for each human region, a
tool's calls on that human protein are ranked by the tool's own score, and only calls to
its best L = 1000 target proteins are kept, before transfer. kmerseek has two scores and
uses the one that ranks better at the region's length (MATCHED-PERMISSIVE.md, measured on
the Pfam region benchmark): region_mean_idf for a region under 30 aa, the E-value for 30 aa
and longer. Every kmerseek region is kept first; there is no Bonferroni filter here.

Per call and human region with the same label rule:

  inside = overlap / call length     (share of the call inside the region)
  cover  = overlap / region length   (share of the region the call covers)
  iou    = overlap / union

"Landed" is inside >= 0.8 and cover >= 0.3, the rule of notebook 244.

Coordinates. Everything is put on kmerseek's convention, 0-based with the end excluded:
DisProt and the aligners (phmmer, MMseqs2, Foldseek, ProstT5, Reseek) report 1-based
inclusive ends, so their starts move down by one.

Outputs, one pair per (arm, target) under --out-dir:

  <arm>.<target>.regions.parquet   one row per (human region, label rule): the best call,
                                   the best landing call, their target proteins, and the
                                   number of calls and targets
  <arm>.<target>.hits.parquet      per human region, before any transfer: did the tool
                                   report anything overlapping it at all (used to tell
                                   "no call" from "called, but from an unlabelled target")
  <arm>.<target>.ranks.parquet     for every human query and every target protein that
                                   carries a DisProt label, that target's rank in the
                                   query's list under each ranking score: what the
                                   length-matched random-query control reads
  <arm>.<target>.length_rho.parquet  Spearman correlation of each ranking score with the
                                   call's length, over every call in the lists: the check
                                   that no score is region length in disguise

Runs as a SLURM array: task i of n takes every n-th input file, sorted.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
QFO_BIN = REPO / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin"

TARGETS = [
    "mouse",
    "chicken",
    "zebrafish",
    "ciona",
    "fly",
    "worm",
    "yeast",
    "arabidopsis",
    "ecoli",
    "disprot",
]
#: Comparison tools, named as their output directory under results/regions/.
BASELINES = [
    "foldseek",
    "prostt5",
    "reseek",
    "hmmer3_phmmer",
    "hmmer3_jackhmmer",
    "mmseqs2_seqseq",
    "mmseqs2_iterative",
]
FRAGMENT_TOOLS = {"foldseek", "reseek"}
INSIDE_MIN = 0.8
COVER_MIN = 0.3
#: kmerseek ranks by mean IDF below this region length (aa) and by E-value at or above it.
SHORT_MAX = 30
LIST_LENGTH = 1000
HUMAN_TAXON = 9606

KM_RE = re.compile(
    r"^human_vs_(?P<sp>[a-z]+)\.(?P<alpha>.+)\.k(?P<k>\d+)\.lc(?P<lc>true|false)\.regions\.parquet$"
)
BL_RE = re.compile(r"^human_vs_(?P<sp>[a-z]+)\.(?P<tool>[a-z0-9_]+)\.tsv\.gz$")


def list_jobs(results: Path) -> list[dict]:
    jobs = []
    for p in sorted((results / "kmerseek").glob("human_vs_*.regions.parquet")):
        m = KM_RE.match(p.name)
        if m and m["sp"] in TARGETS:
            lc = "True" if m["lc"] == "true" else "False"
            jobs.append(
                dict(
                    path=p,
                    target=m["sp"],
                    tool="kmerseek",
                    arm=f"kmerseek.{m['alpha']}_k{m['k']}_lc{lc}",
                )
            )
    for tool in BASELINES:
        for p in sorted((results / "regions" / tool).glob("human_vs_*.tsv.gz")):
            m = BL_RE.match(p.name)
            if m and m["sp"] in TARGETS and m["tool"] == tool:
                jobs.append(dict(path=p, target=m["sp"], tool=tool, arm=tool))
    return jobs


# ---------------------------------------------------------------------------
# DisProt tables, from scripts/fetch_disprot.py.
# ---------------------------------------------------------------------------
def human_regions(disprot_dir: Path) -> pl.DataFrame:
    """The human functional regions scored: 0-based start, end excluded."""
    return (
        pl.read_parquet(disprot_dir / "disprot_function_regions.parquet")
        .filter((pl.col("taxon") == HUMAN_TAXON) & pl.col("qfo_same_sequence"))
        .select(
            pl.col("accession").alias("query_acc"),
            "term_id",
            (pl.col("start") - 1).alias("true_start"),
            pl.col("end").alias("true_end"),
        )
        .unique()
    )


def target_map(disprot_dir: Path, target: str) -> pl.DataFrame:
    """The DisProt functional regions that can pass on a label in one target.

    A QfO proteome contributes only its proteins whose QfO sequence is DisProt's, since
    DisProt's coordinates are on DisProt's sequence. The pool contributes every
    non-human protein, whose sequence it is by construction.
    """
    reg = pl.read_parquet(disprot_dir / "disprot_function_regions.parquet")
    if target == "disprot":
        reg = reg.filter(pl.col("taxon") != HUMAN_TAXON)
    else:
        reg = reg.filter((pl.col("qfo_label") == target) & pl.col("qfo_same_sequence"))
    return reg.select(
        pl.col("accession").alias("target_acc"),
        pl.col("term_id").alias("t_term"),
        (pl.col("start") - 1).alias("t_feat_start"),
        pl.col("end").alias("t_feat_end"),
    ).unique()


# ---------------------------------------------------------------------------
# Calls.
# ---------------------------------------------------------------------------
def load_calls(job: dict, ev) -> pl.DataFrame | None:
    """Every call of one arm against one target, 0-based end-excluded, with its scores.

    kmerseek carries two scores, both higher-is-better: mean_idf, and evalue_score =
    -log10(region_evalue). A region with no finite E-value (kmerseek writes inf when the
    region's own composition is past the fit's range, or when the arm is exact) gets
    evalue_score null and falls to the bottom of an E-value ranking.
    """
    p = job["path"]
    if p.stat().st_size == 0:
        return None
    if job["tool"] == "kmerseek":
        lf = pl.scan_parquet(p)
        names = lf.collect_schema().names()
        if not names:
            return None
        lf = ev.drop_self_matches(lf, names)
        ev_col = (
            pl.col("region_evalue").cast(pl.Float64)
            if "region_evalue" in names
            else pl.lit(None, dtype=pl.Float64)
        )
        idf_col = (
            pl.col("region_mean_idf").cast(pl.Float64)
            if "region_mean_idf" in names
            else pl.lit(None, dtype=pl.Float64)
        )
        return (
            lf.select(
                ev.extract_accession(pl.col("query_name")).alias("query_acc"),
                ev.extract_accession(pl.col("target_name")).alias("target_acc"),
                pl.col("region_start").cast(pl.Int64).alias("qstart"),
                pl.col("region_end").cast(pl.Int64).alias("qend"),
                pl.col("target_start").cast(pl.Int64).alias("tstart"),
                pl.col("target_end").cast(pl.Int64).alias("tend"),
                idf_col.alias("mean_idf"),
                pl.when(ev_col.is_finite() & (ev_col > 0))
                .then(-ev_col.log10())
                .when(ev_col == 0)
                .then(pl.lit(400.0))  # below the smallest double; ranks first
                .otherwise(None)
                .alias("evalue_score"),
            )
            .filter(pl.col("query_acc") != pl.col("target_acc"))
            .collect(engine="streaming")
        )
    regions = ev.load_regions(p, False)
    if regions is None:
        return None
    df = regions.collect(engine="streaming")
    ev.release_inflated(p)
    if job["tool"] in FRAGMENT_TOOLS:
        df = ev.dedup_fragment_regions(df, 0.9)
    return df.select(
        "query_acc",
        "target_acc",
        (pl.col("qstart") - 1).alias("qstart"),
        "qend",
        (pl.col("tstart") - 1).alias("tstart"),
        "tend",
        "score",
    )


def truncate(calls: pl.DataFrame, score: str, list_length: int) -> pl.DataFrame:
    """Keep the calls to each query's best `list_length` target proteins by `score`.

    A target's rank is its best call's score; ties are broken by accession so the list is
    the same on every run.
    """
    best = (
        calls.filter(pl.col(score).is_not_null())
        .group_by("query_acc", "target_acc")
        .agg(pl.col(score).max().alias("_best"))
        .sort(["query_acc", "_best", "target_acc"], descending=[False, True, False])
        .with_columns(_rank=pl.int_range(pl.len()).over("query_acc"))
        .filter(pl.col("_rank") < list_length)
        .select("query_acc", "target_acc")
    )
    return calls.join(best, on=["query_acc", "target_acc"], how="semi").with_columns(
        pl.col(score).alias("rank_score")
    )


def transfer(calls: pl.DataFrame, tmap: pl.DataFrame, ev, min_overlap: float):
    """A call takes the term of every target region it covers by >= min_overlap."""
    return (
        calls.join(tmap, on="target_acc", how="inner")
        .with_columns(
            t_overlap=ev.overlap_expr("tstart", "tend", "t_feat_start", "t_feat_end")
        )
        .filter(
            pl.col("t_overlap")
            >= min_overlap * (pl.col("t_feat_end") - pl.col("t_feat_start"))
        )
    )


def score_regions(labelled: pl.DataFrame, regions: pl.DataFrame, ev) -> pl.DataFrame:
    """Best call and best landing call per (human region, label rule)."""
    key = ["query_acc", "term_id", "true_start", "true_end"]
    both = []
    for rule in ("same_term", "any_term"):
        on = ["query_acc", "term_id"] if rule == "same_term" else ["query_acc"]
        lab = (
            labelled.rename({"t_term": "term_id"})
            if rule == "same_term"
            else labelled.drop("t_term")
        )
        m = (
            lab.join(regions, on=on, how="inner")
            .with_columns(
                ov=ev.overlap_expr("qstart", "qend", "true_start", "true_end")
            )
            .filter(pl.col("ov") > 0)
            .with_columns(
                inside=pl.col("ov")
                / (pl.col("qend") - pl.col("qstart")).clip(lower_bound=1),
                cover=pl.col("ov")
                / (pl.col("true_end") - pl.col("true_start")).clip(lower_bound=1),
                iou=pl.col("ov")
                / (
                    pl.max_horizontal("qend", "true_end")
                    - pl.min_horizontal("qstart", "true_start")
                ).clip(lower_bound=1),
            )
            .with_columns(
                landed=(pl.col("inside") >= INSIDE_MIN) & (pl.col("cover") >= COVER_MIN)
            )
        )
        # Total order so the chosen call does not depend on row order.
        order = ["iou", "rank_score", "target_acc", "tstart", "qstart"]
        desc = [True, True, False, False, False]
        call_cols = [
            "qstart",
            "qend",
            "target_acc",
            "tstart",
            "tend",
            "t_feat_start",
            "t_feat_end",
            "rank_score",
            "iou",
            "inside",
            "cover",
        ]
        best_any = (
            m.sort(order, descending=desc)
            .group_by(key)
            .agg(
                n_overlapping_calls=pl.len(),
                n_overlapping_targets=pl.col("target_acc").n_unique(),
                any_inside_half=(pl.col("inside") >= 0.5).any(),
                landed=pl.col("landed").any(),
                n_landed_targets=pl.col("target_acc")
                .filter(pl.col("landed"))
                .n_unique(),
                *[pl.col(c).first().alias(f"best_{c}") for c in call_cols],
            )
        )
        best_land = (
            m.filter("landed")
            .sort(order, descending=desc)
            .group_by(key)
            .agg(*[pl.col(c).first().alias(f"land_{c}") for c in call_cols])
        )
        both.append(
            best_any.join(best_land, on=key, how="left").with_columns(
                label_rule=pl.lit(rule)
            )
        )
    return pl.concat(both, how="diagonal_relaxed")


def raw_hits(calls: pl.DataFrame, regions: pl.DataFrame, ev) -> pl.DataFrame:
    """Per human region: calls overlapping it at all, before the list cut and transfer."""
    return (
        calls.select("query_acc", "target_acc", "qstart", "qend")
        .join(regions, on="query_acc", how="inner")
        .filter(ev.overlap_expr("qstart", "qend", "true_start", "true_end") > 0)
        .group_by("query_acc", "term_id", "true_start", "true_end")
        .agg(n_hits=pl.len(), n_hit_targets=pl.col("target_acc").n_unique())
    )


def donor_ranks(calls: pl.DataFrame, score: str, donors: pl.Series) -> pl.DataFrame:
    """Rank (1 = best) of every labelled target protein in each query's list, by `score`.

    The rank is over all target proteins the tool reported for the query, not only the
    labelled ones, so it is the same number the list cut uses.
    """
    return (
        calls.filter(pl.col(score).is_not_null())
        .group_by("query_acc", "target_acc")
        .agg(pl.col(score).max().alias("best"))
        .sort(["query_acc", "best", "target_acc"], descending=[False, True, False])
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("query_acc"))
        .filter(pl.col("target_acc").is_in(donors.implode()))
        .select("query_acc", "target_acc", "rank", pl.lit(score).alias("rank_by"))
    )


def length_rho(calls: pl.DataFrame, scores: list[str]) -> pl.DataFrame:
    """Spearman rho of each score with call length, over every call with that score."""
    rows = []
    c = calls.with_columns(length=(pl.col("qend") - pl.col("qstart")).cast(pl.Float64))
    for sc in scores:
        d = c.filter(pl.col(sc).is_not_null() & pl.col(sc).is_finite())
        rho = (
            d.select(pl.corr(sc, "length", method="spearman")).item()
            if d.height > 2
            else None
        )
        rows.append(dict(rank_by=sc, n_calls=d.height, spearman_rho=rho))
    return pl.DataFrame(
        rows, schema=dict(rank_by=pl.String, n_calls=pl.Int64, spearman_rho=pl.Float64)
    )


def reduce_one(job, regions, tmap, ev, min_overlap, list_length):
    calls = load_calls(job, ev)
    if calls is None or calls.height == 0:
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    scores = ["mean_idf", "evalue_score"] if job["tool"] == "kmerseek" else ["score"]
    donors = tmap["target_acc"].unique()
    ranks = pl.concat([donor_ranks(calls, sc, donors) for sc in scores])
    rho = length_rho(calls, scores)
    q = regions.filter(pl.col("query_acc").is_in(calls["query_acc"].unique().implode()))
    hits = raw_hits(calls, q, ev)
    if job["tool"] == "kmerseek":
        # Each human region is scored on the list ranked by the score for its length.
        short = q.filter((pl.col("true_end") - pl.col("true_start")) < SHORT_MAX)
        long_ = q.filter((pl.col("true_end") - pl.col("true_start")) >= SHORT_MAX)
        parts = []
        for sub, score in ((short, "mean_idf"), (long_, "evalue_score")):
            if sub.height == 0:
                continue
            lab = transfer(truncate(calls, score, list_length), tmap, ev, min_overlap)
            parts.append(
                score_regions(lab, sub, ev).with_columns(rank_by=pl.lit(score))
            )
        out = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    else:
        lab = transfer(truncate(calls, "score", list_length), tmap, ev, min_overlap)
        out = score_regions(lab, q, ev).with_columns(rank_by=pl.lit("score"))
    ident = dict(arm=job["arm"], tool=job["tool"], target=job["target"])
    lit = {k: pl.lit(v) for k, v in ident.items()}
    return tuple(
        df.with_columns(**lit) if df.height else df for df in (out, hits, ranks, rho)
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=Path, required=True, help="pipeline results/")
    ap.add_argument("--disprot-dir", type=Path, required=True)
    ap.add_argument("--qfo-bin", type=Path, default=QFO_BIN)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    ap.add_argument(
        "--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )
    ap.add_argument("--only", default=None, help="regex on the input file name")
    ap.add_argument("--list", action="store_true", help="print the job count and exit")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--list-length", type=int, default=LIST_LENGTH)
    args = ap.parse_args()

    sys.path.insert(0, str(args.qfo_bin))
    import evaluate_domain_calls as ev  # noqa: E402

    jobs = list_jobs(args.results)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["path"].name)]
    if args.list:
        n_km = sum(j["tool"] == "kmerseek" for j in jobs)
        print(
            f"{len(jobs)} input files: {n_km} kmerseek, {len(jobs) - n_km} comparison"
        )
        return
    mine = jobs[args.task :: args.n_tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    regions = human_regions(args.disprot_dir)
    maps: dict[str, pl.DataFrame] = {}
    for j in mine:
        stem = f"{j['arm']}.{j['target']}"
        out_r = args.out_dir / f"{stem}.regions.parquet"
        out_h = args.out_dir / f"{stem}.hits.parquet"
        out_k = args.out_dir / f"{stem}.ranks.parquet"
        out_s = args.out_dir / f"{stem}.length_rho.parquet"
        if all(p.exists() for p in (out_r, out_h, out_k, out_s)):
            print(f"skip {stem}: done", flush=True)
            continue
        if j["target"] not in maps:
            maps[j["target"]] = target_map(args.disprot_dir, j["target"])
        out, hits, ranks, rho = reduce_one(
            j, regions, maps[j["target"]], ev, args.min_overlap, args.list_length
        )
        # Temporary name then rename, so an interrupted task leaves nothing the skip
        # check above would take as finished.
        for df, path in ((out, out_r), (hits, out_h), (ranks, out_k), (rho, out_s)):
            tmp = path.with_suffix(".tmp")
            df.write_parquet(tmp)
            tmp.rename(path)
        n_land = (
            int(out.filter(pl.col("label_rule") == "same_term")["landed"].sum())
            if out.height
            else 0
        )
        print(
            f"{stem}: {hits.height} human regions hit, "
            f"{n_land} landed with the same term",
            flush=True,
        )


if __name__ == "__main__":
    main()
