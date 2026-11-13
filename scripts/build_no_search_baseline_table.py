#!/usr/bin/env python3
"""The no-search baselines as leaderboard tool rows, and the two checks on them.

Reads the output of the region benchmark's `noSearchBaselines` entry
(no_search_baseline_metrics.parquet, or a folder of its *.metrics.parquet files) and its
Nextflow trace, and writes:

    results/leaderboard/no_search_baselines.parquet
        One row per (baseline, setting, truth set, stratum, metric), in the long-table
        schema of scripts/build_leaderboard_long.py (PR 148) plus two columns:
          tool_kind   "no search"
          applicable  False where the baseline cannot address the cell. Such a row has a
                      null value: it is shown as "not applicable", never as zero.
        Not-applicable rows are written for every truth set and stratum the search tools
        of the same experiment have, so the site has a cell to mark.
    results/leaderboard/no_search_compute.parquet
        Seconds per tool from the trace (realtime, COMPLETED tasks, newest attempt),
        and seconds per 1,000 queries.

and prints the two checks the prompt sheet asks for:

  1. Kyte-Doolittle on Swiss-Prot TRANSMEM and INTRAMEM against every kmerseek row of
     the same run (setting x target x dedup mode), with n beside each score.
  2. The Pfam family frequency baseline's Fmax on Pfam (and Pfam-N) beside each search
     tool's Fmax: best, median and worst over its settings and targets.

The search-tool rows come from results/leaderboard/results.parquet of PR 148
(olgabot/leaderboard-all-experiments), passed as --results.

Usage:
    python scripts/build_no_search_baseline_table.py \\
        --metrics <outdir>/no_search/no_search_baseline_metrics.parquet \\
        --trace <run dir>/qfo_pfam_region.<date>.trace.txt \\
        --results results/leaderboard/results.parquet \\
        --source-path /scratch/users/olgabot/.../no_search_baseline_metrics.parquet \\
        --source-commit <commit the run was launched from> \\
        --outdir results/leaderboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "nextflow-runs" / "qfo-pfam-region-benchmark"
sys.path.insert(0, str(PIPE / "bin"))
import no_search_baselines as nsb  # noqa: E402

APPLIES = PIPE / "assets" / "no_search_baselines.tsv"
EXPERIMENT = "qfo_region_midi_plus"
QUERY_SET = "midi-plus"
N_QUERIES = 998
MIN_N = 30   # strata with fewer truth instances are shown grey on the site

# Same names and order as build_leaderboard_long.py (PR 148), so the rows concatenate.
SCHEMA = {
    "experiment": pl.String, "tool": pl.String, "tool_version": pl.String,
    "setting": pl.String, "query_set": pl.String, "target_db": pl.String,
    "target_db_version": pl.String, "target_mya_to_human": pl.Float64,
    "stratum_axis": pl.String, "stratum_bin": pl.String, "metric": pl.String,
    "value": pl.Float64, "n_instances": pl.Int64, "truth_source": pl.String,
    "truth_version": pl.String, "dedup_or_raw": pl.String, "source_path": pl.String,
    "source_commit": pl.String,
    # Added here.
    "tool_kind": pl.String, "applicable": pl.Boolean,
}
TRUTH = {
    "pfam": ("Pfam-A domain instances", "Pfam 38.2"),
    "pfamn": ("Pfam-N domain instances", "Pfam 35.0"),
    "swissprot": ("Swiss-Prot features", None),
    "swissprot_disordered": ('Swiss-Prot REGION features noted "Disordered"', None),
    "swissprot_compbias": ("Swiss-Prot COMPBIAS features", None),
}
METRICS = ["fmax", "fmax_precision", "fmax_recall", "fmax_threshold", "auprc",
           "recall_reachable", "median_iou_tp", "family_fmax"]
STRATUM_AXES = ["all", "identity", "plddt", "feature_type", "feature_length_bin"]
TOOL_VERSION = {
    "kyte_doolittle": "Kyte and Doolittle 1982 scale, notebook 231 scan",
    "iupred": "IUPred 1.0 long (iupred_string.c from InterProScan 5), gcc 14.2.0",
    "metapredict": "metapredict 3.0.1 (docker.io/olgabot/metapredict:2026-08-27)",
    "seg": "segmasker, BLAST+ 2.16.0",
    "pfam_family_frequency": "Pfam 38.2 human annotation, queries excluded",
}
# Trace process -> baseline tool. baselineCalls is split by its tag (the tool name).
PROCESS_TOOL = {
    "kdScan": "kyte_doolittle",
    "iupredLong": "iupred",
    "metapredictScores": "metapredict",
    "segmasker": "seg",
    "familyFrequency": "pfam_family_frequency",
}


def read_metrics(path: Path) -> pl.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.metrics.parquet"))
        if not files:
            raise SystemExit(f"no *.metrics.parquet under {path}")
        m = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    else:
        m = pl.read_parquet(path)
    if "applicable" not in m.columns:
        m = nsb.mark_applicable(m, nsb.read_applies(APPLIES))
    return m


def long_rows(m: pl.DataFrame, source_path: str, commit: str | None) -> pl.DataFrame:
    m = m.filter((pl.col("split") == "all") & pl.col("stratum_axis").is_in(STRATUM_AXES))
    metrics = [c for c in METRICS if c in m.columns]
    long = m.select("truth_set", "tool", "variant", "stratum_axis", "stratum",
                    "n_truth_instances", "applicable", *metrics).unpivot(
        index=["truth_set", "tool", "variant", "stratum_axis", "stratum",
               "n_truth_instances", "applicable"],
        on=metrics, variable_name="metric", value_name="value")
    return long.select(
        pl.lit(EXPERIMENT).alias("experiment"),
        pl.col("tool"),
        pl.col("tool").replace_strict(TOOL_VERSION, default=None).alias("tool_version"),
        pl.col("variant").alias("setting"),
        pl.lit(QUERY_SET).alias("query_set"),
        pl.lit("none (query sequence only)").alias("target_db"),
        pl.lit(None, dtype=pl.String).alias("target_db_version"),
        pl.lit(None, dtype=pl.Float64).alias("target_mya_to_human"),
        pl.col("stratum_axis"),
        pl.col("stratum").cast(pl.String).alias("stratum_bin"),
        pl.col("metric"),
        # Never a number where the baseline cannot address the cell.
        pl.when(pl.col("applicable")).then(pl.col("value").cast(pl.Float64))
        .otherwise(None).alias("value"),
        pl.col("n_truth_instances").cast(pl.Int64).alias("n_instances"),
        pl.col("truth_set").replace_strict({k: v[0] for k, v in TRUTH.items()},
                                           default=pl.col("truth_set"))
        .alias("truth_source"),
        pl.col("truth_set").replace_strict({k: v[1] for k, v in TRUTH.items()},
                                           default=None, return_dtype=pl.String)
        .alias("truth_version"),
        pl.lit("raw").alias("dedup_or_raw"),
        pl.lit(source_path).alias("source_path"),
        pl.lit(commit, dtype=pl.String).alias("source_commit"),
        pl.lit("no search").alias("tool_kind"),
        pl.col("applicable"),
    ).select([pl.col(c).cast(t) for c, t in SCHEMA.items()])


def fill_not_applicable(rows: pl.DataFrame, results: pl.DataFrame | None,
                        skip_truth_sets: list[str] = ()) -> pl.DataFrame:
    """A not-applicable row for every cell the search tools have and a baseline lacks.

    A truth set left out with --drop-truth-set gets no rows at all: the baseline was not
    scored there correctly, which is "not run", not "not applicable".
    """
    if results is None:
        return rows
    skip = [TRUTH[t][0] for t in skip_truth_sets]
    cells = (results.filter((pl.col("experiment") == EXPERIMENT)
                            & ~pl.col("truth_source").is_in(skip))
             .select("truth_source", "truth_version", "stratum_axis", "stratum_bin",
                     "metric", "n_instances")
             .unique(["truth_source", "stratum_axis", "stratum_bin", "metric"]))
    settings = rows.select("tool", "tool_version", "setting").unique()
    have = rows.select("tool", "setting", "truth_source", "stratum_axis", "stratum_bin",
                       "metric")
    missing = (settings.join(cells, how="cross")
               .join(have, on=["tool", "setting", "truth_source", "stratum_axis",
                               "stratum_bin", "metric"], how="anti"))
    if missing.height == 0:
        return rows
    first = rows.row(0, named=True)
    fill = missing.with_columns(
        pl.lit(EXPERIMENT).alias("experiment"), pl.lit(QUERY_SET).alias("query_set"),
        pl.lit("none (query sequence only)").alias("target_db"),
        pl.lit(None, dtype=pl.String).alias("target_db_version"),
        pl.lit(None, dtype=pl.Float64).alias("target_mya_to_human"),
        pl.lit(None, dtype=pl.Float64).alias("value"),
        pl.lit("raw").alias("dedup_or_raw"),
        pl.lit(first["source_path"]).alias("source_path"),
        pl.lit(first["source_commit"], dtype=pl.String).alias("source_commit"),
        pl.lit("no search").alias("tool_kind"), pl.lit(False).alias("applicable"),
    ).select([pl.col(c).cast(t) for c, t in SCHEMA.items()])
    return pl.concat([rows, fill]).sort("tool", "setting", "truth_source", "stratum_axis",
                                        "stratum_bin", "metric")


def compute_rows(trace: Path, n_queries: int) -> pl.DataFrame:
    """Seconds per baseline from the trace; scoring tasks are not counted, as for every
    search tool in build_leaderboard_long.py."""
    t = pl.read_csv(trace, separator="\t", infer_schema=False)
    t = t.with_columns(pl.col("process").str.split(":").list.last().alias("step"))
    t = t.filter(pl.col("status") == "COMPLETED")
    rows = []
    for r in t.iter_rows(named=True):
        if r["step"] == "baselineCalls":
            tool = r["tag"]
        else:
            tool = PROCESS_TOOL.get(r["step"])
        if tool is None:
            continue
        rows.append({"tool": tool, "step": r["step"], "tag": r["tag"],
                     "complete": r["complete"],
                     "wall_s": _seconds(r["realtime"]), "cpus": int(r["cpus"]),
                     "peak_ram_gb": _gb(r["peak_rss"]), "native_id": r["native_id"]})
    if not rows:
        raise SystemExit(f"{trace}: no COMPLETED no-search task")
    df = (pl.DataFrame(rows).sort("complete")
          .unique(["tool", "step", "tag"], keep="last", maintain_order=True))
    return (df.group_by("tool").agg(
        pl.col("wall_s").sum(),
        (pl.col("wall_s") * pl.col("cpus") / 3600).sum().alias("cpu_h"),
        pl.col("peak_ram_gb").max(),
        pl.len().alias("n_tasks"),
        pl.col("step").unique().sort().str.join(", ").alias("steps"),
    ).with_columns(
        (pl.col("wall_s") / n_queries * 1000).alias("seconds_per_1000_queries"),
        pl.lit(n_queries).alias("n_queries"),
        pl.lit("no search").alias("tool_kind"),
        pl.lit(EXPERIMENT).alias("experiment"),
        pl.lit(str(trace)).alias("source"),
    ).sort("tool"))


def _seconds(text: str | None) -> float | None:
    import re
    if text in (None, "", "-"):
        return None
    units = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    return sum(float(n) * units[u]
               for n, u in re.findall(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)", text))


def _gb(text: str | None) -> float | None:
    if text in (None, "", "-"):
        return None
    num, unit = text.split()
    return float(num) * {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit] / 1024**3


def check_kd(m: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Check 1: Kyte-Doolittle against every kmerseek row on TRANSMEM and INTRAMEM."""
    out = []
    for feature in ("TRANSMEM", "INTRAMEM"):
        kd = m.filter((pl.col("tool") == "kyte_doolittle") & pl.col("applicable")
                      & (pl.col("split") == "all") & (pl.col("stratum") == feature))
        km = results.filter(
            (pl.col("experiment") == EXPERIMENT) & (pl.col("tool") == "kmerseek")
            & (pl.col("metric") == "fmax") & (pl.col("stratum_axis") == "feature_type")
            & (pl.col("stratum_bin") == feature))
        if kd.height != 1 or km.height == 0:
            raise SystemExit(f"check 1, {feature}: {kd.height} baseline rows and "
                             f"{km.height} kmerseek rows")
        v, n = kd["fmax"][0], kd["n_truth_instances"][0]
        out.append({
            "feature_type": feature, "n_truth_instances": n,
            "grey_below_30": n < MIN_N,
            "kyte_doolittle_fmax": v,
            "kmerseek_rows": km.height,
            "kmerseek_settings": km["setting"].n_unique(),
            "kmerseek_targets": km["target_db"].n_unique(),
            "kmerseek_best_fmax": km["value"].max(),
            "kmerseek_rows_at_or_above": int((km["value"] >= v).sum()),
        })
    return pl.DataFrame(out)


def check_feature_types(m: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Each baseline's Fmax on a Swiss-Prot feature type (TRANSMEM, REGION, REPEAT, ...)
    beside every search tool's rows on the same stratum of the same run."""
    out = []
    b = m.filter(pl.col("applicable") & (pl.col("split") == "all")
                 & (pl.col("truth_set") == "swissprot")
                 & (pl.col("stratum_axis") == "feature_type"))
    for r in b.sort("stratum", "tool", "variant").iter_rows(named=True):
        s = results.filter(
            (pl.col("experiment") == EXPERIMENT) & (pl.col("metric") == "fmax")
            & (pl.col("truth_source") == TRUTH["swissprot"][0])
            & (pl.col("stratum_axis") == "feature_type") & (pl.col("stratum_bin") == r["stratum"]))
        for (tool,), g in s.group_by("tool", maintain_order=True):
            out.append({"feature_type": r["stratum"], "n_truth_instances": r["n_truth_instances"],
                        "grey_below_30": r["n_truth_instances"] < MIN_N,
                        "baseline": r["tool"], "baseline_fmax": r["fmax"], "tool": tool,
                        "n_rows": g.height, "best_fmax": g["value"].max(),
                        "rows_at_or_above_baseline": int((g["value"] >= r["fmax"]).sum())})
    return pl.DataFrame(out).sort("feature_type", "baseline", "best_fmax",
                                  descending=[False, False, True])


def check_frequency(m: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Check 2: the frequency floor beside each search tool's Fmax, per Pfam truth set."""
    out = []
    for ts in ("pfam", "pfamn"):
        f = m.filter((pl.col("tool") == "pfam_family_frequency") & (pl.col("truth_set") == ts)
                     & (pl.col("split") == "all") & (pl.col("stratum_axis") == "all"))
        if f.height == 0:
            continue
        floor, n = f["fmax"][0], f["n_truth_instances"][0]
        s = results.filter(
            (pl.col("experiment") == EXPERIMENT) & (pl.col("metric") == "fmax")
            & (pl.col("stratum_axis") == "all") & (pl.col("truth_source") == TRUTH[ts][0]))
        for (tool,), g in s.group_by("tool", maintain_order=True):
            out.append({
                "truth_set": ts, "n_truth_instances": n,
                "frequency_fmax": floor, "tool": tool,
                "n_rows": g.height,
                "best_fmax": g["value"].max(), "median_fmax": g["value"].median(),
                "worst_fmax": g["value"].min(),
                "rows_below_floor": int((g["value"] < floor).sum()),
            })
    return pl.DataFrame(out).sort("truth_set", "best_fmax", descending=[False, True])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--results", type=Path, help="PR 148 results/leaderboard/results.parquet")
    p.add_argument("--trace", type=Path)
    p.add_argument("--trace-source-path", default=None,
                   help="where the trace lives on Sherlock, written as the cost rows' source")
    p.add_argument("--source-path", default="")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--n-queries", type=int, default=N_QUERIES)
    p.add_argument("--outdir", type=Path)
    p.add_argument("--drop-truth-set", action="append", default=[],
                   help="leave a truth set's rows out, e.g. pfamn when the run scored it "
                        "against the whole-proteome table (before restrictBaselineTruth)")
    a = p.parse_args()

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    m = read_metrics(a.metrics)
    if a.drop_truth_set:
        print(f"leaving out truth sets {a.drop_truth_set}: "
              f"{m.filter(pl.col('truth_set').is_in(a.drop_truth_set)).height} metric rows")
        m = m.filter(~pl.col("truth_set").is_in(a.drop_truth_set))
    results = pl.read_parquet(a.results) if a.results else None

    print("Applicable baseline cells (split all), Fmax with n:")
    print(m.filter(pl.col("applicable") & (pl.col("split") == "all"))
          .select("truth_set", "tool", "variant", "stratum_axis", "stratum",
                  "n_truth_instances", pl.col("fmax").round(4))
          .sort("truth_set", "tool", "variant", "stratum_axis", "stratum"))

    if results is not None:
        print("\nCheck 1. Kyte-Doolittle against every kmerseek row of the same run:")
        print(check_kd(m, results))
        print("\nCheck 2. Pfam family frequency floor beside each tool's Fmax "
              "(stratum all; rows = setting x target x dedup mode):")
        print(check_frequency(m, results).with_columns(
            pl.col(pl.Float64).round(4)))
        print("\nEvery baseline on a Swiss-Prot feature type the search tools were also "
              "scored on, against each search tool's rows:")
        print(check_feature_types(m, results).with_columns(pl.col(pl.Float64).round(4)))

    if a.outdir:
        a.outdir.mkdir(parents=True, exist_ok=True)
        rows = fill_not_applicable(long_rows(m, a.source_path, a.source_commit), results,
                                   a.drop_truth_set)
        rows.write_parquet(a.outdir / "no_search_baselines.parquet")
        print(f"\nwrote {a.outdir / 'no_search_baselines.parquet'}: {rows.height} rows, "
              f"{int(rows['applicable'].sum())} applicable")
        if a.trace:
            c = compute_rows(a.trace, a.n_queries)
            if a.trace_source_path:
                c = c.with_columns(pl.lit(a.trace_source_path).alias("source"))
            c.write_parquet(a.outdir / "no_search_compute.parquet")
            print(c)


if __name__ == "__main__":
    main()
