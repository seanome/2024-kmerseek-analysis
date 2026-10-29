"""Read the Sherlock job's kmerseek outputs into two tables.

kmerseek_pairs.parquet  one row per dataset, setting, scaled, search kind (candidates, e2e,
                        e2e_noextend), query and target: the largest n_intersecting_hashes,
                        the number of regions, the smallest region E-value, the largest
                        region Poisson score, and the best region's coordinates (by Poisson
                        score; 0-based, end-exclusive, as kmerseek writes them)
kmerseek_cost.tsv       wall seconds and peak memory of every index and search step,
                        from /usr/bin/time -v, with the exit status and the exact command
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import polars as pl

KINDS = ("candidates", "e2e", "e2e_noextend")


def _read_time(p: Path) -> dict:
    out = {"wall_seconds": None, "peak_rss_mb": None}
    if not p.exists():
        return out
    t = p.read_text()
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    if m:
        out["peak_rss_mb"] = int(m.group(1)) / 1024
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
    if m:
        parts = [float(x) for x in m.group(1).split(":")]
        sec = 0.0
        for x in parts:
            sec = sec * 60 + x
        out["wall_seconds"] = sec
    return out


def _pairs(csv: Path) -> pl.DataFrame | None:
    if not csv.exists() or csv.stat().st_size == 0:
        return None
    df = pl.read_csv(csv, infer_schema_length=0)
    if df.height == 0:
        return df
    num = ["n_intersecting_hashes", "region_evalue", "region_poisson_score"]
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in num if c in df.columns])
    for c in ("region_start", "region_end", "target_start", "target_end"):
        df = df.with_columns(pl.col(c).cast(pl.Int64))
    best = df.sort(["region_poisson_score", "region_end"], descending=[True, False]).unique(
        ["query_name", "target_name"], keep="first", maintain_order=True
    )
    agg = df.group_by("query_name", "target_name").agg(
        pl.col("n_intersecting_hashes").max().alias("n_shared"),
        pl.len().alias("n_regions"),
        pl.col("region_evalue").min().alias("min_region_evalue"),
        pl.col("region_poisson_score").max().alias("max_region_poisson"),
    )
    best = best.select(
        "query_name", "target_name",
        pl.col("region_start").alias("km_q_start"), pl.col("region_end").alias("km_q_end"),
        pl.col("target_start").alias("km_t_start"), pl.col("target_end").alias("km_t_end"),
    )  # fmt: skip
    return agg.join(best, on=["query_name", "target_name"]).rename(
        {"query_name": "query", "target_name": "reference"}
    )


def run(job_dir: Path, results: Path, cfg: dict) -> None:
    results.mkdir(parents=True, exist_ok=True)
    params = pl.read_csv(
        job_dir / "params.tsv", separator="\t", infer_schema_length=0, quote_char=None
    )
    sha = (job_dir / "kmerseek_git_sha.txt").read_text().strip()
    frames, cost = [], []
    for row in params.iter_rows(named=True):
        out = job_dir / "out" / row["dataset"] / row["setting"] / f"s{row['scaled']}"
        t = _read_time(out / f"{row['step']}.time")
        err = (
            (out / f"{row['step']}.stderr").read_text()
            if (out / f"{row['step']}.stderr").exists()
            else ""
        )
        cost.append(
            dict(
                **row, **t, kmerseek_git_sha=sha,
                ka_fit_refused=any(
                    "no Karlin-Altschul fit" in line
                    or ("refus" in line.lower() and "karlin" in line.lower())
                    for line in err.splitlines()
                ),
                stderr_tail=" | ".join(err.strip().splitlines()[-3:])[:500],
            )
        )  # fmt: skip
        if row["step"] in KINDS:
            df = _pairs(out / f"{row['step']}.csv.gz")
            if df is None:
                continue
            frames.append(
                df.with_columns(
                    dataset=pl.lit(row["dataset"]), setting=pl.lit(row["setting"]),
                    scaled=pl.lit(int(row["scaled"])), kind=pl.lit(row["step"]),
                )
            )  # fmt: skip
    pl.DataFrame(cost).write_csv(results / "kmerseek_cost.tsv", separator="\t")
    if not frames:
        raise SystemExit(f"no kmerseek search output found under {job_dir / 'out'}; check the job logs")
    pl.concat(frames, how="diagonal_relaxed").write_parquet(results / "kmerseek_pairs.parquet")
    shutil.copyfile(job_dir / "params.tsv", results / "kmerseek_params.tsv")
    print(f"collected {len(frames)} searches, kmerseek {sha}")
