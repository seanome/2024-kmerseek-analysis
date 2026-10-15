#!/usr/bin/env python3
"""Build the three tables the protein annotation leaderboard site reads.

    results/leaderboard/results.parquet  one row per (truth set, dedup, tool, setting,
                                         species, stratum): Fmax, AUPRC, median IoU, counts
    results/leaderboard/compute.parquet  one row per (tool, setting, species): CPU-hours and
                                         peak memory of the search task, from Nextflow traces
    results/leaderboard/coverage.md      tool x stratum grid: measured or not run
    results/leaderboard/provenance.json  every input file with its sha256

Inputs are the QfO Pfam region benchmark's midi-plus run (998 human queries from chr6 and
three immune gene sets, searched against 15 QfO proteomes), finished 2026-09-13 on Sherlock:

    --metrics  all_domain_metrics.parquet   written by aggregate_domain_metrics.py
    --traces   the run's Nextflow trace files (run-midi-plus and run-midi-plus-all-qfo)
    --summary  midi_testset_summary.json    holds n_queries

Nothing is filled in. A (tool, stratum) pair with no metrics row has no row here either, and
the site shows it as "not run".
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import polars as pl

# The stratum axes the site has a chip row for, and the bins each axis is defined with in
# nextflow-runs/qfo-pfam-region-benchmark/bin/evaluate_domain_calls.py (IDENTITY_BINS,
# STRATA["plddt"], FEATURE_LENGTH_BINS, the Swiss-Prot FT vocabulary). A bin listed here
# with no metrics row is what coverage.md marks "not run": the grid has to know every bin
# that COULD exist, not only the ones the run wrote.
AXES = {
    "identity": ["0-20%", "20-30%", "30-40%", "40-60%", "60-100%", "no_homolog"],
    "plddt": ["0-50", "50-70", "70-90", "90-100"],
    "feature_length_bin": [
        "1",
        "2-15",
        "16-30",
        "31-60",
        "61-120",
        "121-250",
        "251+",
        "whole_protein",
    ],
    "feature_type": [
        "ACT_SITE",
        "BINDING",
        "COILED",
        "DNA_BIND",
        "DOMAIN",
        "INTRAMEM",
        "MOTIF",
        "REGION",
        "REPEAT",
        "SITE",
        "TRANSMEM",
        "ZN_FING",
    ],
}
TRUTH_SETS = ["swissprot", "pfam", "pfamn"]

METRIC_COLS = [
    "fmax",
    "fmax_precision",
    "fmax_recall",
    "fmax_threshold",
    "auprc",
    "median_iou_tp",
    "n_truth_instances",
    "n_reachable_instances",
    "n_instances_found",
    "n_calls",
    "n_tp_calls",
    "n_fp_calls",
    "n_proteins_scored",
    "median_feature_length",
]

# Trace process -> (tool, setting) as the metrics table names them. Only search tasks:
# building a target database (kmerseekIndex, foldseekDb, prostt5Db, hhblitsBuildDB,
# mmseqsDb) is a one-off per proteome, and storeDir cache hits leave no trace row for it,
# so its cost cannot be counted completely from these traces.
SEARCH_PROCESSES = {
    "phmmerSearch": ("hmmer3_phmmer", "default"),
    "jackhmmerSearch": ("hmmer3_jackhmmer", "n3"),
    "hhblitsSearch": ("hhblits", "single_seq"),
    "foldseekSearch": ("foldseek", "3di_aa"),
    "prostt5Search": ("prostt5", "3di_from_seq"),
    "hmmscanAnnotate": ("hmmscan", "direct"),
}
HARDWARE = "Stanford Sherlock, SLURM partition hns; searches on CPU only"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)")
_SECONDS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def duration_seconds(text: str | None) -> float | None:
    """Nextflow's '1h 2m 3.4s' / '34ms' / '-' as seconds."""
    if text is None or text.strip() in ("", "-"):
        return None
    parts = _DURATION.findall(text)
    if not parts:
        raise ValueError(f"unreadable duration {text!r}")
    return sum(float(n) * _SECONDS[u] for n, u in parts)


_SIZE = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def size_gb(text: str | None) -> float | None:
    """Nextflow's '3.3 MB' / '1.1 GB' as GiB."""
    if text is None or text.strip() in ("", "-"):
        return None
    num, unit = text.split()
    return float(num) * _SIZE[unit] / 1024**3


def percent(text: str | None) -> float | None:
    if text is None or text.strip() in ("", "-"):
        return None
    return float(text.rstrip("%"))


def tool_setting_species(process: str, tag: str) -> tuple[str, str, str] | None:
    """Map one trace row to the (tool, setting, species) the metrics table uses."""
    if process == "kmerseekSearch":
        # '<species>_<alphabet>_k<k>_lc<false|true>'; species codes carry no underscore,
        # alphabets do (hp_thomas_dill2).
        m = re.fullmatch(r"([a-z0-9]+)_(.+_k\d+)_lc(false|true)", tag)
        if not m:
            return None
        return "kmerseek", f"{m[2]}_lc{m[3].capitalize()}", m[1]
    if process == "mmseqs2Search":
        m = re.fullmatch(r"human_vs_(\w+) \[(mmseqs2_seqseq|mmseqs2_iterative)\]", tag)
        return (m[2], "s7", m[1]) if m else None
    if process == "hmmscanAnnotate":
        return "hmmscan", "direct", "all"
    if process in SEARCH_PROCESSES:
        m = re.fullmatch(r"human_vs_(\w+)", tag)
        return (*SEARCH_PROCESSES[process], m[1]) if m else None
    return None


def read_traces(paths: list[Path]) -> pl.DataFrame:
    frames = [
        pl.read_csv(p, separator="\t", infer_schema=False).with_columns(
            pl.lit(str(p)).alias("trace_file")
        )
        for p in paths
    ]
    return pl.concat(frames, how="diagonal")


def build_compute(traces: pl.DataFrame, n_queries: int) -> pl.DataFrame:
    """One row per (tool, setting, species): the search task's cost, from the traces.

    Some searches completed more than once (a re-run after a code change or a crash, with
    a new task hash). The trace does not say which run wrote the calls that were scored,
    so the newest completion gives the headline numbers and every completed run is kept in
    the spread: n_completed_runs, cpu_h_per_1000_queries_min/_max, peak_rss_gb_max.
    CACHED rows are not counted: they reuse an earlier row, they are not a second run.

    GPU: no search used one. ProstT5 also needs its 3Di letters predicted for the query
    proteins, which ran on one GPU (prostt5Db on the human query set, --gres=gpu:1). That
    step's run time is counted as gpu_h on every ProstT5 row, since every search needs the
    queries converted. The same step for each relative's proteome is a one-off per
    relative and is not counted, like every other tool's database build.
    """
    rows = []
    gpu_steps = []
    for r in traces.filter(pl.col("status") == "COMPLETED").iter_rows(named=True):
        process = r["process"].split(":")[-1]
        if process == "prostt5Db" and (r["tag"] or "").startswith("human"):
            gpu_steps.append(
                {
                    "complete": r["complete"],
                    "gpu_h": duration_seconds(r["realtime"]) / 3600,
                    "task_hash": r["hash"],
                }
            )
            continue
        key = tool_setting_species(process, r["tag"] or "")
        if key is None:
            continue
        realtime = duration_seconds(r["realtime"])
        cpu_pct = percent(r["%cpu"])
        rows.append(
            {
                "tool": key[0],
                "setting": key[1],
                "species": key[2],
                "process": process,
                "task_hash": r["hash"],
                "complete": r["complete"],
                "trace_file": r["trace_file"],
                "cpus_requested": int(r["cpus"]),
                "realtime_s": realtime,
                "cpu_percent": cpu_pct,
                "peak_rss_gb": size_gb(r["peak_rss"]),
            }
        )
    keys = ["tool", "setting", "species"]
    df = pl.DataFrame(rows).with_columns(
        # CPU time actually used: wall time x average CPU utilisation.
        (pl.col("realtime_s") * pl.col("cpu_percent") / 100 / 3600).alias("cpu_h")
    )
    spread = df.group_by(keys).agg(
        pl.len().alias("n_completed_runs"),
        (pl.col("cpu_h").min() * 1000 / n_queries).alias("cpu_h_per_1000_queries_min"),
        (pl.col("cpu_h").max() * 1000 / n_queries).alias("cpu_h_per_1000_queries_max"),
        pl.col("peak_rss_gb").max().alias("peak_rss_gb_max"),
    )
    latest = df.sort("complete", descending=True).unique(
        keys, keep="first", maintain_order=True
    )
    gpu = max(gpu_steps, key=lambda g: g["complete"]) if gpu_steps else None
    is_prostt5 = pl.col("tool") == "prostt5"
    return (
        latest.join(spread, on=keys, how="left")
        .with_columns(
            (pl.col("realtime_s") * pl.col("cpus_requested") / 3600).alias(
                "cpu_h_reserved"
            ),
            pl.when(is_prostt5 & pl.lit(gpu is not None))
            .then(pl.lit(gpu["gpu_h"] if gpu else 0.0))
            .otherwise(0.0)
            .alias("gpu_h"),
            pl.when(is_prostt5 & pl.lit(gpu is not None))
            .then(pl.lit("1 GPU, query 3Di step"))
            .otherwise(pl.lit("none"))
            .alias("gpu"),
            pl.lit(n_queries).alias("n_queries"),
            pl.lit("measured").alias("provenance"),
            pl.lit(HARDWARE).alias("hardware"),
            # A submitter's own number goes here, with their hardware in `hardware`.
            pl.lit(None, dtype=pl.Float64).alias("cpu_h_reported"),
        )
        .with_columns(
            (pl.col("cpu_h") * 1000 / pl.col("n_queries")).alias(
                "cpu_h_per_1000_queries"
            ),
            (pl.col("cpu_h_reserved") * 1000 / pl.col("n_queries")).alias(
                "cpu_h_reserved_per_1000_queries"
            ),
            (pl.col("gpu_h") * 1000 / pl.col("n_queries")).alias(
                "gpu_h_per_1000_queries"
            ),
        )
        .sort(keys)
    )


def build_results(metrics: pl.DataFrame) -> pl.DataFrame:
    axes = ["all", *AXES]
    return (
        metrics.filter(
            (pl.col("split") == "all")
            & pl.col("stratum_axis").is_in(axes)
            & pl.col("truth_set").is_in(TRUTH_SETS)
        )
        .rename({"variant": "setting", "dedup_transfers": "dedup"})
        .select(
            "truth_set",
            "dedup",
            "tool",
            "setting",
            "species",
            "species_mya",
            "stratum_axis",
            "stratum",
            *METRIC_COLS,
        )
        .sort(
            "truth_set",
            "dedup",
            "tool",
            "setting",
            "species",
            "stratum_axis",
            "stratum",
        )
    )


def coverage_markdown(results: pl.DataFrame, species_order: list[str]) -> str:
    """Tool x stratum grid. ■ = at least one metrics row, □ = none (not run)."""
    tools = results["tool"].unique().sort().to_list()
    columns: list[tuple[str, str, pl.Expr]] = []
    for t in TRUTH_SETS:
        columns.append(("truth set", t, pl.col("truth_set") == t))
    for d in (False, True):
        columns.append(("dedup", "dedup" if d else "raw", pl.col("dedup") == d))
    for s in species_order:
        columns.append(("species", s, pl.col("species") == s))
    for axis, bins in AXES.items():
        for b in bins:
            columns.append(
                (axis, b, (pl.col("stratum_axis") == axis) & (pl.col("stratum") == b))
            )

    lines = [
        "# Coverage: what was measured and what was not run",
        "",
        "■ at least one row in results.parquet. □ no row: not run.",
        "",
        "| axis | value | " + " | ".join(tools) + " |",
        "|---|---|" + "---|" * len(tools),
    ]
    for axis, value, expr in columns:
        present = set(results.filter(expr)["tool"].unique().to_list())
        cells = ["■" if t in present else "□" for t in tools]
        lines.append(f"| {axis} | {value} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--traces", type=Path, nargs="+", required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("results/leaderboard"))
    args = ap.parse_args()

    n_queries = json.loads(args.summary.read_text())["n_queries"]
    results = build_results(pl.read_parquet(args.metrics))
    compute = build_compute(read_traces(sorted(args.traces)), n_queries)

    # Keep compute to the (tool, setting, species) the results table scores; the all-QfO
    # traces also hold searches of species this run never scored.
    scored = results.select("tool", "setting", "species").unique()
    compute = compute.join(scored, on=["tool", "setting", "species"], how="semi")

    missing = scored.join(compute, on=["tool", "setting", "species"], how="anti")
    species_order = (
        results.filter(pl.col("species") != "all")
        .select("species", "species_mya")
        .unique()
        .sort("species_mya", "species")["species"]
        .to_list()
    ) + ["all"]

    args.outdir.mkdir(parents=True, exist_ok=True)
    results.write_parquet(args.outdir / "results.parquet")
    compute.write_parquet(args.outdir / "compute.parquet")
    (args.outdir / "coverage.md").write_text(coverage_markdown(results, species_order))
    inputs = [args.metrics, args.summary, *sorted(args.traces)]
    root = os.path.commonpath([p.resolve() for p in inputs])
    inputs = [p.resolve() for p in inputs]
    provenance = {
        "n_queries": n_queries,
        "inputs": {os.path.relpath(p, root): sha256(p) for p in inputs},
        "outputs": {
            n: sha256(args.outdir / n)
            for n in ("results.parquet", "compute.parquet", "coverage.md")
        },
        "scored_without_compute_row": missing.sort(
            "tool", "setting", "species"
        ).to_dicts(),
    }
    (args.outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=1) + "\n"
    )
    print(f"results.parquet  {results.height:>6} rows")
    print(f"compute.parquet  {compute.height:>6} rows")
    print(f"scored (tool, setting, species) with no trace row: {missing.height}")


if __name__ == "__main__":
    main()
