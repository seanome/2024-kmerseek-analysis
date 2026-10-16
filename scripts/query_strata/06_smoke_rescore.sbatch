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
    kept = new.filter(~pl.col("stratum_axis").is_in(new_axes)).select(old.columns)
    # Value by value, NaN equal to NaN: DataFrame.equals says False on a NaN in both
    # (job 48149070 reported every file as changed for that reason alone).
    j = old.join(kept, on=key, how="full", suffix="_new", coalesce=True)
    same = old.height == kept.height == j.height
    for c in (c for c in old.columns if c not in key):
        a, b = j[c], j[c + "_new"]
        eq = (a == b) | (a.is_null() & b.is_null())
        if a.dtype.is_float():
            eq = eq | (a.is_nan() & b.is_nan())
        same &= bool(eq.fill_null(False).all())
    added = new.filter(pl.col("stratum_axis").is_in(new_axes))
    whole = new.filter((pl.col("stratum_axis") == "all")).select("split", pl.col("n_truth_instances").alias("w"))
    sums = (added.group_by("split", "stratum_axis").agg(pl.col("n_truth_instances").sum().alias("s"))
            .join(whole, on="split"))
    off = sums.filter(pl.col("s") != pl.col("w"))
    # Pfam-N's answer key is every human Pfam-N region (44,315 on 18,724 proteins), not
    # only the queries'; a protein-level bin holds only queries, so there the bins cannot
    # add up to the whole set. Reported, not counted as a failure of the new axes.
    if os.path.basename(f).startswith("pfamn."):
        print("  Pfam-N protein-level sums vs whole:", off.select("split", "stratum_axis", "s", "w").rows())
        off = off.filter(pl.col("stratum_axis").is_in(["region_length", "family_superkingdoms", "family_phyla"]))
    print(os.path.basename(f), "old rows identical:", same, "| old", old.height, "new", new.height,
          "| new axes", sorted(added["stratum_axis"].unique().to_list()),
          "| axis sums off:", off.height)
    if off.height:
        print(off)
    bad += (not same) + off.height
print("SMOKE", "PASS" if bad == 0 else f"FAIL {bad}")
PY
