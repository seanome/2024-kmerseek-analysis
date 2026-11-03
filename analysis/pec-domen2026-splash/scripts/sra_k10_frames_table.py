#!/usr/bin/env python3
"""Per run: reads kept after choosing each read's frame with protein 10-mers, and why the others
were dropped; windows per kept read at HP k 24, 27, 30; CPU time and peak memory per step.

Reads $S/sra_search/k10_frames/<run>/ (from slurm/26_sra_k10_frame_sketch.sbatch):
<run>.summary.tsv (pec-tools best-frame), windows.tsv, best_frame.time and manysketch.time
(GNU time -v).

Drop reasons, in the order best-frame applies them (each read counts once):
- low_complexity_read: fastp complexity below 0.3, so no frame is tried;
- no_allowed_frame: every frame has a stop codon or is a low-complexity peptide;
- no_10mer_match: the chosen frame has no protein 10-mer in the reference (includes frames
  shorter than 10 residues, `no_window`).

Usage: sra_k10_frames_table.py <runs.tsv> <k10_frames_dir> <out.tsv>
"""

import re
import sys
from pathlib import Path

import polars as pl

runs_p, frames_dir, out_p = sys.argv[1:]
runs = pl.read_csv(runs_p, separator="\t", infer_schema_length=0)


def gnu_time(f: Path) -> tuple[float | None, float | None, float | None]:
    """CPU seconds, wall seconds and peak memory (GB) from a GNU time -v file."""
    if not f.exists():
        return None, None, None
    t = f.read_text()
    u = re.search(r"User time \(seconds\): ([\d.]+)", t)
    s = re.search(r"System time \(seconds\): ([\d.]+)", t)
    w = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    wall = None
    if w:
        parts = [float(x) for x in w.group(1).split(":")]
        wall = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    return (
        float(u.group(1)) + float(s.group(1)) if u and s else None,
        wall,
        int(m.group(1)) / 1024**2 if m else None,
    )


rows = []
for r in runs.iter_rows(named=True):
    d = Path(frames_dir) / r["run"]
    sm = d / f"{r['run']}.summary.tsv"
    if not (d / "sketch.done").exists() or not sm.exists():
        continue
    s = dict(pl.read_csv(sm, separator="\t").iter_rows())
    w = pl.read_csv(d / "windows.tsv", separator="\t").row(0, named=True)
    bf_cpu, bf_wall, bf_mem = gnu_time(d / "best_frame.time")
    ms_cpu, ms_wall, ms_mem = gnu_time(d / "manysketch.time")
    kept = s["fraction_ge_0"]
    low = s["low_complexity_read"]
    row = {
        "run": r["run"],
        "sample": r["sample"],
        "dataset": r["dataset"],
        "condition": r["condition"],
        "contrast": r["contrast"],
        "reference": (d / "reference.txt").read_text().strip().split("/")[-1],
        "reads": s["all_reads"],
        "gbp": s["bases"] / 1e9,
        "low_complexity_read": low,
        "no_allowed_frame": s["no_allowed_frame"] - low,
        "no_10mer_match": s["no_window"] + s["zero_match"],
        "kept": kept,
        "kept_share": kept / s["all_reads"] if s["all_reads"] else None,
        "kept_frame_tied": s["any_match_frame_tied"],
        "mean_residues_per_kept_read": w["residues"] / w["peptides"]
        if w["peptides"]
        else None,
    }
    for k in (24, 27, 30):
        row[f"windows_per_kept_read_k{k}"] = (
            w[f"windows_k{k}"] / w["peptides"] if w["peptides"] else None
        )
        row[f"share_kept_reads_with_window_k{k}"] = (
            w[f"peptides_with_window_k{k}"] / w["peptides"] if w["peptides"] else None
        )
    row |= {
        "best_frame_cpu_s": bf_cpu,
        "best_frame_wall_s": bf_wall,
        "best_frame_peak_gb": bf_mem,
        "manysketch_cpu_s": ms_cpu,
        "manysketch_wall_s": ms_wall,
        "manysketch_peak_gb": ms_mem,
    }
    assert (
        row["low_complexity_read"]
        + row["no_allowed_frame"]
        + row["no_10mer_match"]
        + kept
        == s["all_reads"]
    ), r["run"]
    rows.append(row)
t = pl.DataFrame(rows, infer_schema_length=None)
t = t.with_columns(pl.col(pl.Float64).round(4))
t.write_csv(out_p, separator="\t")
species = pl.when(pl.col("dataset") == "PRJNA1219309").then(pl.lit("Botryllus"))
summ = (
    t.with_columns(species=species.otherwise(pl.lit("human")))
    .group_by("species")
    .agg(
        runs=pl.len(),
        gbp=pl.col("gbp").sum(),
        reads=pl.col("reads").sum(),
        low_complexity_read=pl.col("low_complexity_read").sum(),
        no_allowed_frame=pl.col("no_allowed_frame").sum(),
        no_10mer_match=pl.col("no_10mer_match").sum(),
        kept=pl.col("kept").sum(),
        kept_share_min=pl.col("kept_share").min(),
        kept_share_median=pl.col("kept_share").median(),
        kept_share_max=pl.col("kept_share").max(),
        windows_k24=pl.col("windows_per_kept_read_k24").median(),
        windows_k27=pl.col("windows_per_kept_read_k27").median(),
        windows_k30=pl.col("windows_per_kept_read_k30").median(),
        best_frame_cpu_h=pl.col("best_frame_cpu_s").sum() / 3600,
        best_frame_cpu_s_per_gbp=pl.col("best_frame_cpu_s").sum() / pl.col("gbp").sum(),
        best_frame_peak_gb=pl.col("best_frame_peak_gb").max(),
        manysketch_cpu_h=pl.col("manysketch_cpu_s").sum() / 3600,
        manysketch_peak_gb=pl.col("manysketch_peak_gb").max(),
    )
    .sort("species")
)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
print(summ)
summ.write_csv(out_p.replace(".tsv", "_by_species.tsv"), separator="\t")
