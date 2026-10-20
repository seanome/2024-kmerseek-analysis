#!/usr/bin/env python3
"""Rescore a finished run's Pfam-N metrics, every split and stratum, from its call files.

Runs scored before restrictTruthToQueries (main.nf) divided Pfam-N recall and Fmax by
every human Pfam-N region (44,315 on 18,724 proteins), not by the regions on the proteins
the run searched. Which calls are correct does not depend on that: a call is checked only
against the truth on its own protein, and every call is on a searched protein. So each
published calls/pfamn.*.calls.parquet is rescored with `evaluate_domain_calls.py
--scored-calls`, the scorer's own metric code, with the same per-species inputs the run's
scoreDomainCalls task staged (domain map, covariates, identity, disorder).

Every call file with a published metrics row is scored twice:

  before  against the full truth file, as the run did, and compared with the run's
          all_domain_metrics.parquet row by row. The script stops if the whole set does
          not match exactly. A breakdown whose definition changed in the scorer after the
          run (pLDDT and disorder were redefined in September 2026) can differ; each row
          says whether it matched, and pfamn_rescore_check.csv gives the largest
          difference per breakdown and metric.
  after   against the truth cut to the run's query proteins. Written, in the shape of
          all_domain_metrics.parquet, to pfamn_rescored_metrics.parquet.

Call files without a published row (the midi-plus calls/ folder also holds the 77-target
run's files) are skipped.

Resuming. Each (truth, species) job saves its rows to pfamn_rescore_parts/ as soon as it
finishes, and a rerun skips every job whose part file exists. The part name carries a hash
of --source-commit, the thresholds and the size and time of every input file, so a part
written under other code or inputs is ignored (it stays in the folder, unused).

Workers are started with "spawn", not Linux's default "fork". The parent has already run
polars, whose thread pool does not survive a fork: on 2026-10-19 job 48521254 forked 18
workers this way, used 35 CPU-minutes of the 54 CPU-hours it held in 3 hours, and wrote
nothing before its time limit. macOS defaults to spawn, which is why the local tests passed.
"""

import argparse
import hashlib
import multiprocessing
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parent
sys.path.insert(0, str(BIN))
from restrict_truth_to_queries import query_accessions, restrict  # noqa: E402

ROW_KEY = [
    "tool",
    "variant",
    "species",
    "dedup_transfers",
    "split",
    "stratum_axis",
    "stratum",
]
ARM_KEY = ["tool", "variant", "species", "dedup_transfers"]
TOL = 1e-9


# Columns that depend on how calls tied on score are ordered. A score cut-off reported next
# to Fmax is one point on a plateau of equal F values (any cut-off when Fmax is 0), and
# sensitivity-to-first-false-positive walks calls in score order. Both moved between runs
# on identical input until ties were broken explicitly (see sensitivity_to_first_fp), so
# published runs can differ here while every metric value matches. Reported in
# pfamn_rescore_check.csv; left out of the per-row match.
def tie_order_dependent(col: str) -> bool:
    return col.endswith("_threshold") or col.startswith("sens_first_fp")


def parse_name(path: Path) -> dict:
    """pfamn.<tool>.<variant>.<species>[.dedup].calls.parquet, as evaluate_domain_calls.py
    names them."""
    stem = path.name[: -len(".calls.parquet")]
    dedup = stem.endswith(".dedup")
    parts = (stem[: -len(".dedup")] if dedup else stem).split(".")
    return {
        "tool": parts[1],
        "variant": ".".join(parts[2:-1]),
        "species": parts[-1],
        "dedup_transfers": dedup,
        "path": str(path),
    }


def optional(flag: str, path: Path) -> list[str]:
    return [flag, str(path)] if path.exists() and path.stat().st_size > 0 else []


def score_species(job: dict) -> tuple[str, str, int, float]:
    """Score one species' call files against one truth file and save the rows to
    job["part"]. Returns (truth label, species, call files, seconds)."""
    sp, run_dir = job["species"], Path(job["run_dir"])
    with tempfile.TemporaryDirectory(prefix=f"rescore_{sp}_") as work:
        work = Path(work)
        manifest = work / "manifest.tsv"
        manifest.write_text(
            "".join(
                f"{a['tool']}\t{a['variant']}\t{a['path']}\t"
                f"{'on' if a['dedup_transfers'] else 'off'}\n"
                for a in job["arms"]
            )
        )
        disorder = Path(job["disorder_dir"]) / f"{sp}.disorder_metapredict.parquet"
        cmd = [
            sys.executable,
            str(BIN / "evaluate_domain_calls.py"),
            "--manifest", str(manifest),
            "--scored-calls",
            "--species", sp,
            "--species-mya", str(job["mya"]),
            "--truth", job["truth"],
            "--domain-map", job["domain_map"],
            "--truth-set", "pfamn",
            "--min-overlap", str(job["min_overlap"]),
            "--strict-iou", str(job["strict_iou"]),
            *optional("--covariates", Path(job["covariates"])),
            *optional("--identity", run_dir / f"identity/{sp}.domain_identity.parquet"),
            *optional("--region-plddt", run_dir / "truth/human_domain_plddt.parquet"),
            *optional(
                "--region-disorder", run_dir / "truth/human_domain_disorder.parquet"
            ),
            *optional("--target-disorder", disorder),
        ]  # fmt: skip
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{sp} ({job['label']}) failed:\n{proc.stderr[-3000:]}")
        files = sorted(work.glob("*.metrics.parquet"))
        rows = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        # Written under a temporary name and renamed, so a job killed mid-write never
        # leaves a part that a rerun would take as finished.
        part = Path(job["part"])
        tmp = part.with_suffix(".partial")
        rows.write_parquet(tmp, compression="zstd")
        tmp.rename(part)
        return job["label"], sp, len(job["arms"]), time.monotonic() - t0


def compare(
    before: pl.DataFrame, published: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-row match flag, and the largest difference per breakdown and metric."""
    numeric = [
        c
        for c in published.columns
        if c in before.columns
        and c not in ROW_KEY
        and published.schema[c].is_numeric()
        and before.schema[c].is_numeric()
    ]
    j = before.select(ROW_KEY + numeric).join(
        published.select(ROW_KEY + numeric), on=ROW_KEY, how="inner", suffix="_pub"
    )
    a = {c: pl.col(c).cast(pl.Float64) for c in numeric}
    b = {c: pl.col(f"{c}_pub").cast(pl.Float64) for c in numeric}
    diffs = [
        pl.when(a[c].is_null() & b[c].is_null())
        .then(0.0)
        .when(a[c].is_null() | b[c].is_null())
        .then(float("inf"))
        .when(a[c].is_nan() & b[c].is_nan())
        .then(0.0)
        .when(a[c].is_nan() | b[c].is_nan())
        .then(float("inf"))
        .otherwise((a[c] - b[c]).abs())
        .alias(c)
        for c in numeric
    ]
    d = j.select(*ROW_KEY, *diffs)
    strict = [c for c in numeric if not tie_order_dependent(c)]
    flags = d.select(
        *ROW_KEY,
        (pl.max_horizontal(strict) <= TOL).alias("before_matches_published"),
    )
    check = (
        d.unpivot(index=ROW_KEY, on=numeric, variable_name="metric", value_name="diff")
        .group_by("stratum_axis", "metric")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("diff").max().alias("max_abs_diff"),
            (pl.col("diff") > TOL).sum().alias("n_rows_differ"),
        )
        .with_columns(
            (
                pl.col("metric").str.ends_with("_threshold")
                | pl.col("metric").str.starts_with("sens_first_fp")
            ).alias("tie_order_dependent")
        )
        .sort("stratum_axis", "metric")
    )
    return flags, check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="the run's --outdir (data/<run>/results)",
    )
    p.add_argument(
        "--annotations", required=True, type=Path, help="the run's --annotations"
    )
    p.add_argument(
        "--pfamn-dir",
        required=True,
        type=Path,
        help="the --pfamn_dir the run used (human truth and domain maps)",
    )
    p.add_argument(
        "--disorder-dir",
        type=Path,
        default=Path("NO_DISORDER_DIR"),
        help="<db_cache>/disorder, where proteomeDisorder stored each species",
    )
    p.add_argument(
        "--covariates",
        type=Path,
        help="query covariates to stratify on; default the run's own "
        "truth/human_query_covariates.parquet. Pass PR 164's "
        "human_query_covariates.strata.parquet to add its query-side breakdowns.",
    )
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--strict-iou", type=float, default=0.8)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument(
        "--source-commit",
        default=None,
        help="git commit of the code running this, written on every output row",
    )
    p.add_argument("--outdir", type=Path, help="default: --run-dir")
    args = p.parse_args()
    outdir = args.outdir or args.run_dir

    full_path = args.pfamn_dir / "human_pfamn_truth.parquet"
    full = pl.read_parquet(full_path)
    cut = restrict(full, query_accessions(args.annotations))
    print(
        f"Pfam-N {full.height:,} regions on {full['accession'].n_unique():,} proteins -> "
        f"{cut.height:,} on {cut['accession'].n_unique():,}",
        file=sys.stderr,
    )

    published = pl.read_parquet(args.run_dir / "all_domain_metrics.parquet").filter(
        pl.col("truth_set") == "pfamn"
    )
    mya = dict(published.select("species", "species_mya").unique().iter_rows())
    calls = sorted((args.run_dir / "calls").glob("pfamn.*.calls.parquet"))
    arms = pl.DataFrame([parse_name(f) for f in calls]).join(
        published.select(ARM_KEY).unique(), on=ARM_KEY, how="semi"
    )
    n_pub = published.select(ARM_KEY).unique().height
    if arms.height != n_pub:
        raise SystemExit(f"{n_pub} published arms but {arms.height} have a call file")

    parts = outdir / "pfamn_rescore_parts"
    parts.mkdir(parents=True, exist_ok=True)
    covariates = (
        args.covariates or args.run_dir / "truth/human_query_covariates.parquet"
    )
    # Parts are reused only when this key matches: same commit, thresholds and input files.
    key_files = [
        full_path,
        covariates,
        args.run_dir / "all_domain_metrics.parquet",
        *calls,
    ]
    key_files += sorted(args.pfamn_dir.glob("*_domain_map.parquet"))
    key_files += (
        sorted(args.disorder_dir.glob("*")) if args.disorder_dir.is_dir() else []
    )
    h = hashlib.sha256(
        repr(
            (
                args.source_commit,
                args.min_overlap,
                args.strict_iou,
                sorted(cut["accession"].unique().to_list()),
                [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in key_files],
            )
        ).encode()
    )
    key = h.hexdigest()[:12]
    with tempfile.TemporaryDirectory(prefix="rescore_truth_") as tdir:
        cut_path = Path(tdir) / "human_pfamn_truth.parquet"
        cut.write_parquet(cut_path)
        jobs = []
        for label, truth in (("before", full_path), ("after", cut_path)):
            for sp in sorted(arms["species"].unique()):
                jobs.append(
                    {
                        "label": label,
                        "species": sp,
                        "part": str(parts / f"{label}.{sp}.{key}.parquet"),
                        "truth": str(truth),
                        "domain_map": str(args.pfamn_dir / f"{sp}_domain_map.parquet"),
                        "run_dir": str(args.run_dir),
                        "covariates": str(covariates),
                        "disorder_dir": str(args.disorder_dir),
                        "mya": mya[sp],
                        "min_overlap": args.min_overlap,
                        "strict_iou": args.strict_iou,
                        "arms": arms.filter(pl.col("species") == sp).to_dicts(),
                    }
                )
        todo = [j for j in jobs if not Path(j["part"]).exists()]
        print(
            f"{arms.height} call files, {len(jobs)} scoring jobs, {len(jobs) - len(todo)} "
            f"already done in {parts}; {len(todo)} to run on {args.jobs} workers",
            file=sys.stderr,
            flush=True,
        )
        t0 = time.monotonic()
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            futures = [ex.submit(score_species, j) for j in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                label, sp, n, secs = fut.result()
                print(
                    f"  [{i}/{len(todo)}] {label} {sp}: {n} call files in {secs:.0f} s "
                    f"({secs / max(n, 1):.2f} s each); {time.monotonic() - t0:.0f} s "
                    "since start",
                    file=sys.stderr,
                    flush=True,
                )
    before = pl.concat(
        [pl.read_parquet(j["part"]) for j in jobs if j["label"] == "before"],
        how="diagonal_relaxed",
    )
    after = pl.concat(
        [pl.read_parquet(j["part"]) for j in jobs if j["label"] == "after"],
        how="diagonal_relaxed",
    )

    flags, check = compare(before, published)
    outdir.mkdir(parents=True, exist_ok=True)
    check.write_csv(outdir / "pfamn_rescore_check.csv")
    by_axis = (
        flags.group_by("stratum_axis")
        .agg(
            pl.len().alias("rows"),
            pl.col("before_matches_published").sum().alias("matched"),
        )
        .sort("stratum_axis")
    )
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(by_axis, file=sys.stderr)
    print(
        f"{flags.height} of {published.height} published rows recomputed",
        file=sys.stderr,
    )

    whole = flags.filter(pl.col("stratum_axis") == "all")
    if whole.height == 0 or not whole["before_matches_published"].all():
        raise SystemExit(
            "the whole-set 'before' numbers do not reproduce the run's metrics; "
            "see pfamn_rescore_check.csv"
        )

    # A breakdown is trusted only where its before numbers reproduce the published ones.
    after = after.join(flags, on=ROW_KEY, how="left").with_columns(
        pl.col("before_matches_published").fill_null(False),
        pl.lit(args.source_commit, dtype=pl.String).alias("rescore_commit"),
    )
    after.write_parquet(outdir / "pfamn_rescored_metrics.parquet", compression="zstd")
    print(
        f"wrote {after.height} rows to {outdir / 'pfamn_rescored_metrics.parquet'}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
