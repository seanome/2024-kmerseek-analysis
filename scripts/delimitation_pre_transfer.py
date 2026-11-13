"""Experiment 2: boundary placement of pre-transfer regions against Pfam domains.

For every human Pfam domain instance in the truth and every tool arm, look at the arm's
raw regions on that query (any target, no label transfer) and record whether any region
covers the domain (overlap >= half the domain), whether any region delimits it (IoU >= 0.5)
and the best IoU. The axis of interest is the domain's length as a fraction of its protein.

Two ways to run it.

Pilot mode (--out) reads the midi-plus extract's region tables
(`mhc_kmerseek_regions.parquet`, `mhc_baseline_regions.parquet`, 255 MHC-window queries),
pools every species, and writes one row per (arm, domain) plus a `.hits.parquet`:

    python scripts/delimitation_pre_transfer.py --out /Users/olga/data/qfo-pfam-region-midi-plus/238_delimitation_pre_transfer_mhc.parquet

Full-run mode (--results) reads the midi-plus results directory on Sherlock directly. Each
kmerseek region parquet and each comparison-tool TSV is one job, parsed the way
`extract_mhc.py` steps D and E parse them, with the same cutoffs as the pilot. Task i of n
takes every n-th job; `--list` prints the jobs, `--only` keeps the jobs whose name matches a
regex. Each job writes `<arm>.<species>.domains.parquet` and `<arm>.<species>.hits.parquet`
into --out-dir, scored against every query in the truth (a query with no regions scores 0).
A job whose domains file already exists is skipped, so a resubmitted array redoes only what
is missing. `make reduce-delimitation` in nextflow-runs/qfo-pfam-region-benchmark runs it
as a SLURM array, and `scripts/concat_delimitation_pre_transfer.py` joins the outputs.

    python scripts/delimitation_pre_transfer.py --results data/midi-plus/results --list
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

from extract_mhc import BASELINE_COLS, BASELINE_NUMERIC, strip_acc

E = Path("/Users/olga/data/qfo-pfam-region-midi-plus/extract")
# The ten targets of the midi-plus run (MIDI_PLUS_TARGETS in the pipeline Makefile): the nine
# scored species and Botryllus. The results directory also holds the 77-target all-QfO run's
# files, which this leaves out.
MIDI_PLUS_SPECIES = (
    "mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli,botryllus"
)
KMERSEEK_FILE = re.compile(
    r"human_vs_(\w+?)\.(\w+)\.k(\d+)\.lc(true|false)\.regions\.parquet$"
)
TOOL_FILE = re.compile(r"human_vs_(\w+?)\.")


def score(
    regions: pl.DataFrame, truth: pl.DataFrame, arm: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """regions: query_acc, qstart, qend, already restricted to the arm's significant hits.

    Returns (per-domain table, per-region-hit table). Per domain: best IoU and best cover
    over the query's regions, plus covered (>= half the domain) and delimited (IoU >= 0.5).
    Per region hit: the IoU of every region with the domain it overlaps, which is the
    distribution that says whether an arm's calls are region-shaped or chain-shaped.
    """
    j = truth.join(regions, left_on="accession", right_on="query_acc", how="left")
    # min_horizontal/max_horizontal skip nulls, so an unmatched left join would score as a
    # full overlap; guard on the join key first (this repo's memory: polars_horizontal_skips_nulls).
    j = j.with_columns(
        pl.when(pl.col("qstart").is_null())
        .then(0)
        .otherwise(
            (
                pl.min_horizontal("qend", "domain_end")
                - pl.max_horizontal("qstart", "domain_start")
            ).clip(lower_bound=0)
        )
        .alias("ov")
    )
    j = j.with_columns(
        (
            pl.col("ov")
            / (
                pl.max_horizontal("qend", "domain_end")
                - pl.min_horizontal("qstart", "domain_start")
            )
        )
        .fill_null(0.0)
        .alias("iou"),
        (pl.col("ov") / (pl.col("domain_end") - pl.col("domain_start")))
        .fill_null(0.0)
        .alias("cover"),
    )
    per_domain = (
        j.group_by(
            "accession", "pfam_id", "domain_start", "domain_end", "protein_length"
        )
        .agg(
            pl.col("iou").max().alias("best_iou"),
            pl.col("cover").max().alias("best_cover"),
            (pl.col("qstart").is_not_null()).any().alias("query_has_regions"),
        )
        .with_columns(
            (pl.col("best_cover") >= 0.5).alias("covered"),
            (pl.col("best_iou") >= 0.5).alias("delimited"),
            pl.lit(arm).alias("arm"),
        )
    )
    per_hit = (
        j.filter(pl.col("ov") > 0)
        .select(
            "accession",
            "pfam_id",
            "domain_start",
            "domain_end",
            "protein_length",
            "qstart",
            "qend",
            "iou",
            "cover",
        )
        .with_columns(pl.lit(arm).alias("arm"))
    )
    return per_domain, per_hit


def list_jobs(results: Path, species: set[str]) -> list[dict]:
    """One job per kmerseek region parquet and per comparison-tool TSV of the given species,
    sorted by output name. Files that match neither pattern (hmmscan's human.hmmscan.tsv.gz,
    ProstT5's uncompressed _skipped.tsv) are not jobs. Nor is a zero-byte file: that is a
    search that wrote nothing (all of hp_thomas_dill2_ext2 in the midi-plus run), not a
    search that found nothing, so it is left out as a missing arm and named by --list.
    """
    jobs = []
    for f in sorted((results / "kmerseek").glob("human_vs_*.regions.parquet")):
        m = KMERSEEK_FILE.match(f.name)
        if not m or m.group(1) not in species:
            continue
        sp, alpha, k, lc = m.group(1), m.group(2), int(m.group(3)), m.group(4) == "true"
        jobs.append(
            dict(
                name=f"kmerseek.{alpha}_k{k}_lc{lc}.{sp}",
                arm=f"kmerseek.{alpha}_k{k}_lc{lc}",
                species=sp,
                kind="kmerseek",
                path=f,
            )
        )
    for f in sorted((results / "regions").glob("*/human_vs_*.tsv.gz")):
        tool = f.parent.name
        m = TOOL_FILE.match(f.name)
        if not m or m.group(1) not in species:
            continue
        jobs.append(
            dict(
                name=f"{tool}.{m.group(1)}",
                arm=tool,
                species=m.group(1),
                kind="tool",
                path=f,
            )
        )
    return sorted(jobs, key=lambda j: j["name"])


def read_kmerseek(path: Path, min_poisson_score: float) -> pl.DataFrame:
    """query_acc, qstart, qend of the significant regions, as extract_mhc.py step D reads them."""
    return (
        pl.scan_parquet(path)
        .with_columns(strip_acc(pl.col("query_name")).alias("query_acc"))
        .filter(pl.col("region_poisson_score") >= min_poisson_score)
        .select(
            "query_acc",
            pl.col("region_start").cast(pl.Int64).alias("qstart"),
            pl.col("region_end").cast(pl.Int64).alias("qend"),
        )
        .collect()
    )


def read_tool(path: Path, max_evalue: float) -> pl.DataFrame:
    """query_acc, qstart, qend of the significant hits, as extract_mhc.py step E reads them:
    every column as text first, then cast, with folddisco's extra column before the evalue.
    """
    tool = path.parent.name
    names = (
        BASELINE_COLS[:7] + ["extra", "evalue"]
        if tool == "folddisco"
        else BASELINE_COLS
    )
    df = (
        pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=names,
            infer_schema_length=0,
        )
        .with_columns(
            [
                pl.col(c).cast(t, strict=False)
                for c, t in BASELINE_NUMERIC.items()
                if c in names
            ]
        )
        .with_columns(strip_acc(pl.col("query_name")).alias("query_acc"))
    )
    return df.filter(pl.col("evalue") <= max_evalue).select(
        "query_acc", pl.col("qstart").cast(pl.Int64), pl.col("qend").cast(pl.Int64)
    )


def run_job(job: dict, truth: pl.DataFrame, args) -> None:
    out = args.out_dir / f"{job['name']}.domains.parquet"
    if out.exists():
        print(f"{job['name']}: already scored, skipped", flush=True)
        return
    r = (
        read_kmerseek(job["path"], args.min_poisson_score)
        if job["kind"] == "kmerseek"
        else read_tool(job["path"], args.max_evalue)
    )
    # score() joins every region to every domain on its query. Scoring in batches of queries
    # keeps that join small on the largest region files; each domain's row depends only on
    # its own query's regions, so the result is the same.
    accs = truth["accession"].unique().sort().to_list()
    doms, hits = [], []
    for i in range(0, len(accs), args.batch_queries):
        batch = accs[i : i + args.batch_queries]
        d, h = score(
            r.filter(pl.col("query_acc").is_in(batch)),
            truth.filter(pl.col("accession").is_in(batch)),
            job["arm"],
        )
        doms.append(d)
        hits.append(h)
    # Hits first, domains last: the domains file is what marks a job as done.
    pl.concat(hits).write_parquet(args.out_dir / f"{job['name']}.hits.parquet")
    d = pl.concat(doms)
    d.write_parquet(out)
    print(
        f"{job['name']}: {r.height:_} significant regions, {d.height} domains, "
        f"covered {d['covered'].mean():.3f}, delimited {d['delimited'].mean():.3f}",
        flush=True,
    )


def main_full(args) -> None:
    jobs = list_jobs(args.results, set((args.species or MIDI_PLUS_SPECIES).split(",")))
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["name"])]
    empty = [j for j in jobs if j["path"].stat().st_size == 0]
    jobs = [j for j in jobs if j["path"].stat().st_size > 0]
    if args.list:
        for j in jobs:
            print(j["name"], j["path"].stat().st_size, sep="\t")
        for j in empty:
            print(f"not a job, zero bytes: {j['path']}", file=sys.stderr)
        print(
            f"{len(jobs)} jobs ({len(empty)} zero-byte files left out)", file=sys.stderr
        )
        return
    mine = jobs[args.task :: args.n_tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth = (
        pl.read_parquet(args.truth)
        .select("accession", "pfam_id", "domain_start", "domain_end", "protein_length")
        .unique()
    )
    print(
        f"task {args.task} of {args.n_tasks}: {len(mine)} of {len(jobs)} jobs; "
        f"{truth.height} domain instances on {truth['accession'].n_unique()} queries",
        flush=True,
    )
    failed = []
    for j in mine:
        try:
            run_job(j, truth, args)
        except Exception as e:
            print(
                f"FAILED {j['name']} ({j['path']}): {type(e).__name__} {e}",
                file=sys.stderr,
                flush=True,
            )
            failed.append(j["name"])
    if failed:
        raise SystemExit(
            f"{len(failed)} of {len(mine)} jobs failed: {', '.join(failed)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmerseek", default=str(E / "mhc_kmerseek_regions.parquet"))
    ap.add_argument("--baseline", default=str(E / "mhc_baseline_regions.parquet"))
    ap.add_argument("--truth", default=str(E / "human_domain_truth.parquet"))
    ap.add_argument(
        "--species",
        default=None,
        help="pilot mode: restrict regions to one target species (default: pool all). "
        "Full-run mode: comma-separated species to score (default: the ten midi-plus targets)",
    )
    ap.add_argument(
        "--max-evalue",
        type=float,
        default=1e-3,
        help="baseline regions kept at evalue <= this",
    )
    ap.add_argument(
        "--min-poisson-score",
        type=float,
        default=3.0,
        help="kmerseek regions kept at region_poisson_score >= this (3 = p <= 1e-3)",
    )
    ap.add_argument("--out", help="pilot mode: the per-domain parquet to write")
    ap.add_argument(
        "--results", type=Path, help="full-run mode: the midi-plus results directory"
    )
    ap.add_argument(
        "--out-dir", type=Path, help="full-run mode: where the per-job parquets go"
    )
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument(
        "--only",
        default=None,
        help="full-run mode: keep jobs whose name matches this regex",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="full-run mode: print the jobs and their count, then stop",
    )
    ap.add_argument("--batch-queries", type=int, default=100)
    args = ap.parse_args()
    if args.results is not None:
        if args.out_dir is None and not args.list:
            ap.error("--results needs --out-dir")
        main_full(args)
        return
    if args.out is None:
        ap.error("pass --out (pilot mode) or --results (full-run mode)")
    truth = (
        pl.read_parquet(args.truth)
        .select("accession", "pfam_id", "domain_start", "domain_end", "protein_length")
        .unique()
    )

    km = pl.scan_parquet(args.kmerseek)
    # The truth covers every query of the run; the region tables may cover a subset (the
    # MHC extract has 255 of 998). Score only the queries the region tables were built for.
    searched = set(km.select("query_acc").unique().collect()["query_acc"]) | set(
        pl.scan_parquet(args.baseline)
        .select("query_acc")
        .unique()
        .collect()["query_acc"]
    )
    truth = truth.filter(pl.col("accession").is_in(list(searched)))
    print(
        f"{truth.height} domain instances on {truth['accession'].n_unique()} searched queries",
        flush=True,
    )
    if args.species:
        km = km.filter(pl.col("species") == args.species)
    arms = km.select("alphabet", "ksize", "lc").unique().collect()
    out, hits = [], []
    for a, k, lc in arms.iter_rows():
        r = (
            km.filter(
                (pl.col("alphabet") == a)
                & (pl.col("ksize") == k)
                & (pl.col("lc") == lc)
                & (pl.col("region_poisson_score") >= args.min_poisson_score)
            )
            .select(
                "query_acc",
                pl.col("region_start").cast(pl.Int64).alias("qstart"),
                pl.col("region_end").cast(pl.Int64).alias("qend"),
            )
            .collect()
        )
        d, h = score(r, truth, f"kmerseek.{a}_k{k}_lc{lc}")
        out.append(d)
        hits.append(h)
        print("kmerseek", a, k, lc, r.height, flush=True)
    bl = pl.scan_parquet(args.baseline)
    if args.species:
        bl = bl.filter(pl.col("species") == args.species)
    for (tool,) in bl.select("tool").unique().collect().iter_rows():
        r = (
            bl.filter(
                (pl.col("tool") == tool)
                & (pl.col("evalue").cast(pl.Float64) <= args.max_evalue)
            )
            .select(
                "query_acc",
                pl.col("qstart").cast(pl.Int64),
                pl.col("qend").cast(pl.Int64),
            )
            .collect()
        )
        d, h = score(r, truth, tool)
        out.append(d)
        hits.append(h)
        print(tool, r.height, flush=True)
    df = pl.concat(out)
    df.write_parquet(args.out)
    pl.concat(hits).write_parquet(str(args.out).replace(".parquet", ".hits.parquet"))
    print(
        df.group_by("arm")
        .agg(
            pl.col("covered").mean().round(3),
            pl.col("delimited").mean().round(3),
            pl.len(),
        )
        .sort("delimited", descending=True)
    )


if __name__ == "__main__":
    main()
