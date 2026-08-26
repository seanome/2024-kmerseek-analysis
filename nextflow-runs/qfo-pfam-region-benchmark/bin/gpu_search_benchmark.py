#!/usr/bin/env python3
"""Turn a --gpu_benchmark trace into CPU-vs-GPU throughput per baseline tool.

The question this answers is the one a reviewer asks about the benchmark: were the
structure and profile baselines given a GPU, and did it help. `--gpu_benchmark true`
runs both modes of foldseekSearch and mmseqs2Search in ONE Nextflow session against one
set of databases, so a single trace holds both arms and no cross-run correction is needed.

Two numbers are reported for every arm, and they can disagree:

  per-task queries/s    n_queries / realtime, per search task. This is the number the
                        paper reports, because it is the tool's speed and does not depend
                        on how many jobs the cluster chose to run at once.

  wall-clock queries/s  (n_queries * n_tasks) / (last complete - first submit) over the
                        arm. This is what the pipeline operator feels, and it charges the
                        arm for queue time. On a partition with 4 GPUs and 18 mmseqs tasks
                        the GPU arm serialises 4 at a time while the CPU arm spreads over
                        maxForks, so GPU can win per task and lose on wall clock.

Reporting only the first hides the queue cost; reporting only the second blames the tool
for the scheduler. Both are printed, and the ratio of each is what the report states.

Reads the trace with bin/mqc_trace.py so unit parsing, the process-to-tool map and the
tag conventions stay in one place.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mqc_trace import load_trace, species_from_tag  # noqa: E402

# The two arms that have a GPU path in the pinned binaries. reseek and folddisco are
# absent on purpose: neither links a CUDA library, neither exposes a device flag, so
# there is no GPU mode to compare against.
GPU_CAPABLE_PROCESSES = ("foldseekSearch", "mmseqs2Search")

# Nextflow writes submit/start/complete as local wall-clock timestamps at millisecond
# resolution. They are the only source of queue wait: `duration` covers submit->complete
# and `realtime` covers the task body, so wait is duration - realtime only when nothing
# else intervened. Subtracting the timestamps directly is exact.
_TS = "%Y-%m-%d %H:%M:%S.%f"


def parse_ts(text: str | None) -> float | None:
    if not text or text == "-":
        return None
    try:
        return datetime.strptime(text, _TS).timestamp()
    except ValueError:
        return None


def count_fasta_records(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                n += 1
    return n


def load(trace: Path, n_queries: int) -> pl.DataFrame:
    """Per-task rows for the GPU-capable search processes, CPU and GPU arms both."""
    df = load_trace(trace)
    if df.height == 0:
        return df

    raw = pl.read_csv(trace, separator="\t", infer_schema_length=0,
                      truncate_ragged_lines=True)
    for col in ("submit", "start", "complete"):
        if col not in raw.columns:
            raw = raw.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    stamps = raw.select(
        pl.col(c).map_elements(parse_ts, return_dtype=pl.Float64).alias(f"{c}_ts")
        for c in ("submit", "start", "complete")
    )
    df = pl.concat([df, stamps], how="horizontal")

    df = df.filter(
        pl.col("process").is_in(list(GPU_CAPABLE_PROCESSES))
        & (pl.col("status") == "COMPLETED")
        & pl.col("realtime_s").is_not_null()
        & (pl.col("realtime_s") > 0)
    )
    if df.height == 0:
        return df

    # The mode is read off the tag, which mmseqs2Search and foldseekSearch both mark with
    # a bare `gpu` token when --gpu 1 ran. Reading the tag rather than the process name is
    # what lets one process carry both arms in one session.
    df = df.with_columns(
        pl.when(pl.col("tag").str.contains(r"\bgpu\b")).then(pl.lit("gpu"))
          .otherwise(pl.lit("cpu")).alias("mode"),
        pl.col("tag").map_elements(species_from_tag, return_dtype=pl.String).alias("species"),
        (pl.lit(float(n_queries)) / pl.col("realtime_s")).alias("queries_per_s"),
        (pl.col("start_ts") - pl.col("submit_ts")).alias("queue_wait_s"),
    )
    # mmseqs runs two variants; keep them apart so a seq-seq speedup is not averaged with
    # an iterative one. The arm has to be the SAME string in both modes or the CPU and GPU
    # rows will not join, so the mode token is stripped out of the bracket rather than left
    # in it -- that is what turns foldseek's `[gpu]` back into the same arm as its
    # unbracketed CPU tag.
    arm = (
        pl.col("tag").str.extract(r"\[([^\]]*)\]", 1).fill_null("")
        .str.replace_all(r"\bgpu\b", "")
        .str.strip_chars()
    )
    return df.with_columns(
        pl.when(arm.str.len_chars() == 0).then(pl.lit("default")).otherwise(arm).alias("arm")
    )


def summarise(tasks: pl.DataFrame, n_queries: int) -> pl.DataFrame:
    """One row per (process, arm, mode)."""
    return (
        tasks.group_by(["process", "arm", "mode"])
        .agg(
            pl.len().alias("n_tasks"),
            pl.col("realtime_s").median().alias("median_realtime_s"),
            pl.col("queries_per_s").median().alias("median_queries_per_s"),
            pl.col("queue_wait_s").median().alias("median_queue_wait_s"),
            pl.col("cpu_hours").sum().alias("cpu_hours"),
            pl.col("pct_cpu").median().alias("median_pct_cpu"),
            pl.col("submit_ts").min().alias("_first_submit"),
            pl.col("complete_ts").max().alias("_last_complete"),
        )
        .with_columns(
            (pl.col("_last_complete") - pl.col("_first_submit")).alias("wallclock_s")
        )
        .with_columns(
            # Wall clock charges the arm for every queue wait and every serialisation the
            # scheduler imposed, which is exactly the difference from the per-task number.
            pl.when(pl.col("wallclock_s") > 0)
              .then(pl.col("n_tasks") * n_queries / pl.col("wallclock_s"))
              .otherwise(None)
              .alias("wallclock_queries_per_s")
        )
        .drop("_first_submit", "_last_complete")
        .sort(["process", "arm", "mode"])
    )


def speedups(summary: pl.DataFrame) -> pl.DataFrame:
    """GPU against CPU for each (process, arm) that has both modes."""
    cpu = summary.filter(pl.col("mode") == "cpu")
    gpu = summary.filter(pl.col("mode") == "gpu")
    if cpu.height == 0 or gpu.height == 0:
        return pl.DataFrame()
    joined = cpu.join(gpu, on=["process", "arm"], how="inner", suffix="_gpu")
    if joined.height == 0:
        return pl.DataFrame()
    return joined.select(
        "process", "arm",
        pl.col("median_queries_per_s").alias("cpu_qps_per_task"),
        pl.col("median_queries_per_s_gpu").alias("gpu_qps_per_task"),
        (pl.col("median_queries_per_s_gpu") / pl.col("median_queries_per_s"))
            .alias("per_task_speedup"),
        pl.col("wallclock_queries_per_s").alias("cpu_qps_wallclock"),
        pl.col("wallclock_queries_per_s_gpu").alias("gpu_qps_wallclock"),
        (pl.col("wallclock_queries_per_s_gpu") / pl.col("wallclock_queries_per_s"))
            .alias("wallclock_speedup"),
        pl.col("median_queue_wait_s").alias("cpu_queue_wait_s"),
        pl.col("median_queue_wait_s_gpu").alias("gpu_queue_wait_s"),
    ).sort(["process", "arm"])


def mqc_section(out: Path, speed: pl.DataFrame) -> None:
    """A grouped bar chart for the MultiQC report: per-task and wall-clock side by side.

    Written as a plain *_mqc.json so it joins the existing report without
    build_multiqc_inputs.py having to know about it. Both bars are present in every group
    on purpose: a per-task speedup shown alone is the number that flatters GPU most.
    """
    if speed.height == 0:
        return
    data = {}
    for row in speed.iter_rows(named=True):
        label = f"{row['process']}/{row['arm']}"
        data[label] = {
            "per-task speedup": row["per_task_speedup"],
            "wall-clock speedup": row["wallclock_speedup"],
        }
    out.write_text(json.dumps({
        "id": "qfo_gpu_search",
        "section_name": "GPU vs CPU search",
        "description": (
            "GPU speedup of the baseline search arms, measured in one Nextflow session "
            "against one set of databases. <b>per-task</b> is the tool's own speed "
            "(queries/s within a task). <b>wall-clock</b> charges the arm for queue wait "
            "and for GPU tasks serialising on a partition with fewer GPUs than tasks. "
            "Above 1 means GPU was faster. Note that <code>--gpu 1</code> also replaces "
            "the k-mer prefilter with an exhaustive ungapped one in both tools, so the "
            "GPU arm is scored as its own variant rather than treated as the same result "
            "arriving sooner."),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_gpu_search_plot", "title": "GPU speedup of baseline search",
                    "ylab": "speedup (x, >1 is faster)", "cpswitch": False,
                    "stacking": "group", "height": 400},
        "data": data,
    }, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True,
                    help="Nextflow trace from a --gpu_benchmark run")
    ap.add_argument("--queries-fasta", type=Path,
                    help="query FASTA; its record count is the per-task query count")
    ap.add_argument("--n-queries", type=int,
                    help="query count, if the FASTA is not to hand")
    ap.add_argument("--tasks-out", type=Path, help="per-task parquet")
    ap.add_argument("--summary-out", type=Path, help="per (process, arm, mode) parquet")
    ap.add_argument("--mqc-out", type=Path, help="*_mqc.json bar chart for the report")
    args = ap.parse_args()

    if args.n_queries:
        n_queries = args.n_queries
    elif args.queries_fasta and args.queries_fasta.exists():
        n_queries = count_fasta_records(args.queries_fasta)
    else:
        ap.error("give --n-queries or an existing --queries-fasta")

    tasks = load(args.trace, n_queries)
    if tasks.height == 0:
        # Not an error. A trace with no completed GPU-capable search task is what a run
        # that died before the search stage looks like, and saying so beats an empty table.
        print(f"no completed foldseekSearch/mmseqs2Search tasks in {args.trace}")
        return 1

    summary = summarise(tasks, n_queries)
    speed = speedups(summary)

    print(f"queries per task: {n_queries}")
    print(f"tasks read: {tasks.height}\n")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print("--- per (process, arm, mode) ---")
        print(summary)
        if speed.height:
            print("\n--- GPU against CPU ---")
            print(speed)
        else:
            print("\nOnly one mode present in this trace. Run with --gpu_benchmark true "
                  "to get both arms in one session.")

    if args.tasks_out:
        tasks.write_parquet(args.tasks_out)
    if args.summary_out:
        summary.write_parquet(args.summary_out)
    if args.mqc_out:
        mqc_section(args.mqc_out, speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
