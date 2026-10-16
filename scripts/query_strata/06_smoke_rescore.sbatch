#!/bin/bash
#SBATCH --job-name=qs-smoke-rescore
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/olgabot/leaderboard-query-strata/logs/%x-%j.log
#
# Before the whole midi-plus run is rescored: rerun ONE finished scoring task of that run
# (yeast / phmmer, all three truth sets, work dir 9d/5a865b of run-midi-plus-all-qfo) with
# the code of branch olgabot/query-strata-rescore and the strata covariates, then check
#   - every row the original task wrote comes back with the same values (nothing else moved);
#   - the eight new axes are there, and on each, n_truth_instances over the bins adds up to
#     the whole set.
# Ask: 2 CPUs, 16 GB, 1 h. Measured: that task took 2 m 25 s at peak 10 GB in the midi-plus
# traces (scoreDomainCalls rows for yeast/hhblits and yeast/phmmer, 2026-09-10 to 09-13).
set -euo pipefail
source /scratch/users/olgabot/leaderboard-query-strata/repo/scripts/query_strata/paths.sh
SRC=$QFO_PIPE/run-midi-plus-all-qfo/work/9d/5a865b36abf88eab4c40c39525ca3a
CODE=$WD/repo-rescore/nextflow-runs/qfo-pfam-region-benchmark/bin
RUN=$WD/smoke-rescore
command rm -rf "$RUN"
mkdir -p "$RUN"
cd "$RUN"
# The task's staged inputs, as the links Nextflow made; the outputs stay behind.
for f in "$SRC"/* "$SRC"/.command.sh; do
  case "$(basename "$f")" in *.calls.parquet|*.metrics.parquet|*.curve.parquet) continue;; esac
  command cp -a "$f" .
done
ln -sfn "$WD/out/human_query_covariates.strata.parquet" human_query_covariates.parquet
apptainer exec --bind "$WD" --bind "$QFO_PIPE" --env PATH="$CODE:/usr/local/bin:/usr/bin:/bin" \
  "$PY_IMG" bash -ue .command.sh
apptainer exec --bind "$WD" --bind "$QFO_PIPE" "$PY_IMG" python3 - "$SRC" "$RUN" <<'PY'
import sys, glob, os
import polars as pl
src, run = sys.argv[1:]
new_axes = ["nearest_identity", "membrane", "low_complexity", "protein_length",
            "region_length", "family_superkingdoms", "family_phyla", "busco"]
bad = 0
for f in sorted(glob.glob(f"{src}/*.metrics.parquet")):
    old = pl.read_parquet(f)
    new = pl.read_parquet(f"{run}/{os.path.basename(f)}")
    key = ["split", "stratum_axis", "stratum"]
    kept = new.filter(~pl.col("stratum_axis").is_in(new_axes))
    same = old.sort(key).equals(kept.select(old.columns).sort(key))
    added = new.filter(pl.col("stratum_axis").is_in(new_axes))
    whole = new.filter((pl.col("stratum_axis") == "all")).select("split", pl.col("n_truth_instances").alias("w"))
    sums = (added.group_by("split", "stratum_axis").agg(pl.col("n_truth_instances").sum().alias("s"))
            .join(whole, on="split"))
    off = sums.filter(pl.col("s") != pl.col("w"))
    print(os.path.basename(f), "old rows identical:", same, "| old", old.height, "new", new.height,
          "| new axes", sorted(added["stratum_axis"].unique().to_list()),
          "| axis sums off:", off.height)
    if off.height:
        print(off)
    bad += (not same) + off.height
print("SMOKE", "PASS" if bad == 0 else f"FAIL {bad}")
PY
