#!/bin/bash
# Runs computeFamilyAuc's full alphabet x ksize combo sweep directly (bypassing Nextflow), same
# reasoning as run_rbh_f1_direct.sh: searchHumanVsMouse's storeDir cache in params.outdir is
# empty (raw csv.zst deleted), so `nextflow run -resume` on this .nf file would redo the full
# multi-day kmerseek search sweep just to rebuild the `combo_tuples` channel that now feeds
# computeFamilyAuc -- even though compute_family_auc_combo.py itself only ever reads existing
# .parquet/.csv.zst/.csv.gz files, never the search step's output directly.
#
# Combo list comes from list_metric_leaderboard_combos.py, the same script the pipeline's
# listMetricLeaderboardCombos process calls -- it rescans the filesystem, not Nextflow state, so
# it finds the same ~100 combos here as it would inside the pipeline (78 HP + 11 dayhoff +
# 11 protein). Notebook 206 section 4's hand-picked-family plots only need the fixed 9/10-combo
# subset; this full sweep is for section 9's all-HGNC-family generalization.
#
# Produces the same 206_family_auc/206_family_auc.<enc>.k<k>.csv per-combo files (storeDir-
# equivalent, resumable -- rerun this script and it skips anything already done) and the same
# aggregated 206_family_auc_all_combos.csv notebook 206 section 9 reads.
set -euo pipefail

OUTDIR=/Users/olga/data/gencode/results-human-mouse-orthologs
FAMDIR="$OUTDIR/206_family_auc"
BINDIR=/Users/olga/code/2024-kmerseek-analysis/nextflow-runs/human-mouse-gencode-orthologs/bin
mkdir -p "$FAMDIR"

echo "=== listing combos from disk (same script the pipeline uses) ==="
COMBOS_CSV="$FAMDIR/.combos.csv"
"$BINDIR/list_metric_leaderboard_combos.py" --data-dir "$OUTDIR" --output "$COMBOS_CSV"

/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 <<PYEOF
import csv
import subprocess
from pathlib import Path

famdir = Path("$FAMDIR")
script = "$BINDIR/compute_family_auc_combo.py"
outdir = "$OUTDIR"

with open("$COMBOS_CSV") as f:
    combos = list(csv.DictReader(f))

print(f"{len(combos)} combos to process", flush=True)
for i, row in enumerate(combos, 1):
    dash_enc, disp_enc, k = row["dash_encoding"], row["display_encoding"], row["ksize"]
    out = famdir / f"206_family_auc.{dash_enc}.k{k}.csv"
    if out.exists() and out.stat().st_size > 0:
        print(f"[{i}/{len(combos)}] {dash_enc} k={k} -- already done, skipping", flush=True)
        continue
    print(f"[{i}/{len(combos)}] {dash_enc} k={k}", flush=True)
    subprocess.run([
        script,
        "--dash-encoding", dash_enc, "--display-encoding", disp_enc, "--ksize", k,
        "--data-dir", outdir, "--output", str(out),
    ], check=True)
PYEOF

echo "=== aggregating (same logic as aggregateFamilyAuc) ==="
/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 <<'PYEOF'
import polars as pl
from pathlib import Path

famdir = Path("/Users/olga/data/gencode/results-human-mouse-orthologs/206_family_auc")
frames = [pl.read_csv(f) for f in sorted(famdir.glob("206_family_auc.*.csv"))]
non_empty = [f for f in frames if f.height > 0]
combined = pl.concat(non_empty, how="diagonal_relaxed") if non_empty else pl.DataFrame()
out = Path("/Users/olga/data/gencode/results-human-mouse-orthologs/206_family_auc_all_combos.csv")
combined.write_csv(out)
print(f"{len(non_empty)}/{len(frames)} combo files had rows; {combined.height} total rows -> {out}")
PYEOF

echo "ALL DONE"
