#!/usr/bin/env python3
"""Leaderboard rows of the time-split dataset (notebook 280), experiment
`time_split_pfam33_38`, in the long shape of PR 148's results.parquet and compute.parquet.

    leaderboard_rows.py --scores results-full/scores --inputs inputs/full \
        --trace launch-full/trace.full.txt --index-cache kmerseek_index \
        --settings <leaderboard_settings.tsv> --commit <sha> --half test --out <dir>

Rules, from notebook 280's first cell and PR 147's doc (section 3), fixed before the test
half is read:
- found: a call of the instance's family, at least half inside the t1 domain (the target
  domain placed on the query through the hit), at or above the setting's matched cutoff.
  The cutoff is the one score_transfers.py set on the tune half (setting_summary.csv,
  cutoff_tune): at most 1 family transfer per 100 tune-half decoys.
- recall = found / instances, per tool and setting, on all instances of the half and per
  Pfam type at t1 and per length bin of the t1 domain (<60 aa, >=60 aa; instances.tsv).
- Every recall is reported twice: on all truth (metric `recall`), and on the truth that no
  aligner found at its matched cutoff (metric `recall_aligner_missed`). The aligners are
  MMseqs2 (one round) and phmmer: the pairwise sequence searches PR 147's doc means by "an
  MMseqs2 or BLAST search of the pre-cutoff database". hmmscan is not one of them: it
  searches Pfam 33.1's own profile models. For MMseqs2 and phmmer the second recall is 0
  by construction; the rows are kept so every tool has the same rows.
- decoy_transfers_per_decoy: family transfers to the half's decoys at the cutoff, per decoy.
  One row per setting, stratum "all" (decoys have no Pfam type).
- Every row carries n_instances. A stratum under 30 instances is kept; the site greys it.

--half test reads the test half and is meant to run once, on the full run's scores: it
refuses a scores dir made without --read-test-half, and it writes TEST_HALF_READ in --out
and refuses to run again where that file exists. --half tune runs on the pilot's scores (no
test half there), to test this script; its rows are not results.

Compute rows: one per (tool, setting, step) from the run's Nextflow trace, every row:
completed and failed attempts (they were paid for) and cached ones (a cached row carries
the metrics of the launch that ran it; the failed attempts of an earlier launch are lost
when a resume overwrites the trace, so the cost can be low, never high); the index step from each index's index.log
start and end stamps (the full run serves its indexes from the cache, so its trace has no
index rows). cores come from the trace's cpus column; a trace without it gives null.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from score_transfers import INSTANCE_KEY, found_at_cutoff  # noqa: E402

EXPERIMENT = "time_split_pfam33_38"
TRUTH_SOURCE = "Pfam 38.2 domains on proteins with no Pfam 33.1 domain"
TRUTH_VERSION = "Pfam 38.2 (t1) vs Pfam 33.1 (t0)"
ALIGNERS = ("MMseqs2", "phmmer")
KMERSEEK_VERSION = "0.4.0-rc5"

SCHEMA = {
    "experiment": pl.String,
    "tool": pl.String,
    "tool_version": pl.String,
    "setting": pl.String,
    "query_set": pl.String,
    "target_db": pl.String,
    "target_db_version": pl.String,
    "target_mya_to_human": pl.Float64,
    "stratum_axis": pl.String,
    "stratum_bin": pl.String,
    "metric": pl.String,
    "value": pl.Float64,
    "n_instances": pl.Int64,
    "truth_source": pl.String,
    "truth_version": pl.String,
    "dedup_or_raw": pl.String,
    "source_path": pl.String,
    "source_commit": pl.String,
}
COMPUTE_SCHEMA = {
    "experiment": pl.String,
    "tool": pl.String,
    "setting": pl.String,
    "step": pl.String,
    "n_tasks": pl.Int64,
    "n_target_dbs": pl.Int64,
    "wall_s": pl.Float64,
    "cores": pl.Int64,
    "cpu_h": pl.Float64,
    "peak_ram_gb": pl.Float64,
    "gpu_type": pl.String,
    "gpu_h": pl.Float64,
    "n_queries": pl.Int64,
    "n_targets": pl.Int64,
    "source": pl.String,
}


def tool_of(setting: str) -> dict:
    """score_transfers.py setting name -> tool, version, setting, target database."""
    sp = {"target_db": "Swiss-Prot", "target_db_version": "UniProt 2019_08"}
    if setting.startswith("kmerseek "):
        _, alphabet, k, arm = setting.split()
        return {
            "tool": "kmerseek",
            "tool_version": KMERSEEK_VERSION,
            "setting": f"{alphabet}.{k}.{arm}",
            **sp,
        }
    if setting == "MMseqs2":
        return {
            "tool": "mmseqs2",
            "tool_version": None,
            "setting": "s7_one_round",
            **sp,
        }
    if setting == "phmmer":
        return {
            "tool": "hmmer3_phmmer",
            "tool_version": None,
            "setting": "domE_0.001",
            **sp,
        }
    if setting == "hmmscan Pfam 33.1":
        return {
            "tool": "hmmscan",
            "tool_version": None,
            "setting": "E_0.001_domE_0.001",
            "target_db": "Pfam-A HMMs",
            "target_db_version": "Pfam 33.1",
        }
    raise ValueError(f"unknown setting {setting!r}")


def recall_rows(
    inst: pl.DataFrame, found: pl.DataFrame, settings: list[str]
) -> list[dict]:
    """recall and recall_aligner_missed per setting x stratum. `inst` holds one half's
    instances with pfam_type, length_bin and aligner_found; `found` the (setting, instance)
    pairs found at each setting's cutoff."""
    rows = []
    strata = [("all", pl.lit("all"))] + [
        ("feature_type", pl.col("pfam_type")),
        ("feature_length_bin", pl.col("length_bin")),
    ]
    for s in settings:
        hit = inst.join(
            found.filter(pl.col("setting") == s).select(INSTANCE_KEY),
            on=INSTANCE_KEY,
            how="semi",
        ).select(INSTANCE_KEY)
        flagged = inst.join(
            hit.with_columns(found=pl.lit(True)), on=INSTANCE_KEY, how="left"
        ).with_columns(pl.col("found").fill_null(False))
        for axis, expr in strata:
            g = flagged.with_columns(bin=expr)
            for metric, sub in (
                ("recall", g),
                ("recall_aligner_missed", g.filter(~pl.col("aligner_found"))),
            ):
                agg = sub.group_by("bin").agg(n=pl.len(), k=pl.col("found").sum())
                for b, n, k in agg.sort("bin").iter_rows():
                    rows.append(
                        {
                            "setting_raw": s,
                            "stratum_axis": axis,
                            "stratum_bin": b,
                            "metric": metric,
                            "value": k / n,
                            "n_instances": n,
                        }
                    )
    return rows


def compute_rows(trace: Path, index_cache: Path | None, settings: pl.DataFrame) -> list:
    t = pl.read_csv(trace, separator="\t", infer_schema=False)
    t = t.with_columns(
        process=pl.col("name").str.extract(r"^(\S+)", 1),
        tag=pl.col("name").str.extract(r"\((.*)\)$", 1),
    )
    secs = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

    def seconds(x: str | None) -> float | None:
        if x is None or x.strip() in ("", "-"):
            return None
        return sum(
            float(n) * secs[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)", x)
        )

    unit = {"KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}

    def gb(x: str | None) -> float | None:
        if x is None or x.strip() in ("", "-"):
            return None
        n, u = x.split()
        return float(n) * unit[u]

    def key(process: str, tag: str | None) -> tuple[str, str, str]:
        if process == "kmerseekSearch":
            a, k, arm, _ = tag.split(".")
            return "kmerseek", f"{a}.{k}.{arm}", "search"
        return {
            "mmseqsSearch": ("mmseqs2", "s7_one_round", "search"),
            "phmmerSearch": ("hmmer3_phmmer", "domE_0.001", "search"),
            "hmmscanSearch": ("hmmscan", "E_0.001_domE_0.001", "search"),
            "hmmpressPfam": ("hmmscan", "E_0.001_domE_0.001", "index"),
            "scoreTransfers": ("all", "all", "score"),
        }[process]

    groups: dict[tuple, dict] = {}
    for r in t.iter_rows(named=True):
        k = key(r["process"], r["tag"])
        g = groups.setdefault(
            k, {"n": 0, "wall": 0.0, "cpu_s": 0.0, "peak": None, "cores": set()}
        )
        w = seconds(r["realtime"]) or 0.0
        c = int(r["cpus"]) if r.get("cpus") not in (None, "", "-") else None
        g["n"] += 1
        g["wall"] += w
        g["cores"].add(c)
        g["cpu_s"] = None if (c is None or g["cpu_s"] is None) else g["cpu_s"] + c * w
        p = gb(r["peak_rss"])
        if p is not None:
            g["peak"] = p if g["peak"] is None else max(g["peak"], p)
    rows = []
    for (tool, setting, step), g in sorted(groups.items()):
        cores = g["cores"].pop() if len(g["cores"]) == 1 else None
        rows.append(
            {
                "experiment": EXPERIMENT,
                "tool": tool,
                "setting": setting,
                "step": step,
                "n_tasks": g["n"],
                "n_target_dbs": 1,
                "wall_s": g["wall"],
                "cores": cores,
                "cpu_h": None if g["cpu_s"] is None else g["cpu_s"] / 3600,
                "peak_ram_gb": g["peak"],
                "gpu_type": "none",
                "gpu_h": 0.0,
                "n_queries": None,
                "n_targets": None,
                "source": f"{trace} (every row: a CACHED row carries the metrics of the launch that ran it)",
            }
        )
    if index_cache is not None:
        for a, k in settings.select("alphabet", "ksize").iter_rows():
            log = next(index_cache.glob(f"*.{a}.k{k}.kmerseek.rocksdb/index.log"), None)
            if log is None:
                continue
            stamps = [
                dt.datetime.fromisoformat(m.group(1))
                for line in log.read_text().splitlines()
                if (m := re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) ", line))
            ]
            if len(stamps) < 2:
                continue
            for arm in ("exact", "extend"):
                rows.append(
                    {
                        "experiment": EXPERIMENT,
                        "tool": "kmerseek",
                        "setting": f"{a}.k{k}.{arm}",
                        "step": "index",
                        "n_tasks": 1,
                        "n_target_dbs": 1,
                        "wall_s": (stamps[-1] - stamps[0]).total_seconds(),
                        "cores": 8,
                        "cpu_h": 8 * (stamps[-1] - stamps[0]).total_seconds() / 3600,
                        "peak_ram_gb": None,
                        "gpu_type": "none",
                        "gpu_h": 0.0,
                        "n_queries": None,
                        "n_targets": None,
                        "source": f"{log} first and last stamp; cores = main.nf "
                        "kmerseekIndex cpus 8; one index serves both arms, so both "
                        "rows carry its whole cost",
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--index-cache", type=Path)
    ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--half", choices=("test", "tune"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    summ = pl.read_csv(a.scores / "setting_summary.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    marker = a.out / "TEST_HALF_READ"
    if a.half == "test":
        if "found_test" not in summ.columns:
            raise SystemExit("scores were made without --read-test-half")
        if marker.exists():
            raise SystemExit(
                f"{marker} exists: the test half has been read once already"
            )
        marker.write_text(
            f"read {dt.datetime.now().isoformat(timespec='seconds')} "
            f"from {a.scores} at commit {a.commit}\n"
        )

    q = pl.read_csv(a.inputs / "queries.tsv", separator="\t")
    inst = pl.read_csv(a.inputs / "instances.tsv", separator="\t")
    half_q = q.filter((pl.col("role") == "sample") & (pl.col("half") == a.half))
    inst_h = inst.filter(pl.col("role") == "sample").join(
        half_q.filter(~pl.col("is_decoy")).select("query"), on="query", how="semi"
    )
    cutoffs = dict(zip(summ["setting"], summ["cutoff_tune"]))
    ic = pl.read_parquet(a.scores / "instance_calls.parquet")
    found = found_at_cutoff(ic, cutoffs).join(
        inst_h.select(INSTANCE_KEY), on=INSTANCE_KEY, how="semi"
    )
    by_aligner = found.filter(pl.col("setting").is_in(ALIGNERS)).select(INSTANCE_KEY)
    inst_h = inst_h.join(
        by_aligner.unique().with_columns(aligner_found=pl.lit(True)),
        on=INSTANCE_KEY,
        how="left",
    ).with_columns(pl.col("aligner_found").fill_null(False))
    missing = [s for s in ALIGNERS if s not in cutoffs]
    if missing:
        raise SystemExit(f"no cutoff for aligner(s) {missing}")

    settings = sorted(cutoffs)
    rows = recall_rows(inst_h, found, settings)
    # The scorer's own count of found test instances must match this one.
    if a.half == "test":
        mine = {
            r["setting_raw"]: round(r["value"] * r["n_instances"])
            for r in rows
            if r["stratum_axis"] == "all" and r["metric"] == "recall"
        }
        theirs = dict(zip(summ["setting"], summ["found_test"]))
        bad = {s: (mine.get(s), theirs[s]) for s in theirs if mine.get(s) != theirs[s]}
        if bad:
            raise SystemExit(f"found counts differ from setting_summary.csv: {bad}")
        for r in summ.iter_rows(named=True):
            rows.append(
                {
                    "setting_raw": r["setting"],
                    "stratum_axis": "all",
                    "stratum_bin": "all",
                    "metric": "decoy_transfers_per_decoy",
                    "value": r["decoy_rate_test"],
                    "n_instances": r["decoys_test"],
                }
            )
    query_set = (
        f"time-split {a.half} half: {half_q.filter(~pl.col('is_decoy')).height} queries "
        f"+ {half_q.filter(pl.col('is_decoy')).height} decoys"
    )
    out = pl.DataFrame(
        [
            {
                "experiment": EXPERIMENT,
                **tool_of(r["setting_raw"]),
                "query_set": query_set,
                "target_mya_to_human": None,
                "stratum_axis": r["stratum_axis"],
                "stratum_bin": r["stratum_bin"],
                "metric": r["metric"],
                "value": r["value"],
                "n_instances": r["n_instances"],
                "truth_source": TRUTH_SOURCE,
                "truth_version": TRUTH_VERSION,
                "dedup_or_raw": None,
                "source_path": str(a.scores / "instance_calls.parquet"),
                "source_commit": a.commit,
            }
            for r in rows
        ],
        schema_overrides=SCHEMA,
    ).select([pl.col(c).cast(t) for c, t in SCHEMA.items()])
    stem = f"{EXPERIMENT}.{a.half}"
    out.write_parquet(a.out / f"{stem}.results.parquet")
    out.write_csv(a.out / f"{stem}.results.csv")
    print(f"{out.height} result rows -> {a.out / (stem + '.results.parquet')}")

    if a.trace is not None:
        st = pl.read_csv(a.settings, separator="\t")
        comp = pl.DataFrame(
            compute_rows(a.trace, a.index_cache, st), schema=COMPUTE_SCHEMA
        )
        comp.write_parquet(a.out / f"{stem}.compute.parquet")
        comp.write_csv(a.out / f"{stem}.compute.csv")
        print(f"{comp.height} compute rows")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(
            out.filter(pl.col("stratum_axis") == "all")
            .select("tool", "setting", "metric", "value", "n_instances")
            .sort("metric", "value", descending=[False, True])
        )


if __name__ == "__main__":
    main()
