#!/bin/bash
# Runs computeRbhF1's full combo sweep directly (bypassing Nextflow), same reasoning as
# run_family_auc_direct.sh: searchHumanVsMouse's storeDir cache in params.outdir is empty (raw
# csv.zst deleted), so `nextflow run -resume` on this .nf file would redo the full multi-day
# kmerseek search sweep just to rebuild the `combo_tuples` channel that feeds computeRbhF1 --
# even though compute_rbh_f1_combo.py itself only ever reads existing .parquet/.csv.zst/.csv.gz
# files via ou.load_pair_table / ou.scan_genome_wide_results (parquet-preferring), never the
# search step's output directly.
#
# Combo list comes from list_metric_leaderboard_combos.py, the same script the pipeline's
# listMetricLeaderboardCombos process calls -- it rescans the filesystem, not Nextflow state, so
# it finds the same combos here as it would inside the pipeline.
#
# Produces the same 200_rbh_f1/200_rbh_f1.<enc>.k<k>.csv per-combo files (storeDir-equivalent,
# resumable -- rerun this script and it skips anything already done) and the same aggregated
# 200_rbh_f1_all_combos.csv notebooks 200/205/206 read.
set -euo pipefail

OUTDIR=/Users/olga/data/gencode/results-human-mouse-orthologs
RBHDIR="$OUTDIR/200_rbh_f1"
BINDIR=/Users/olga/code/2024-kmerseek-analysis/nextflow-runs/human-mouse-gencode-orthologs/bin
OF_TSV="/Users/olga/data/gencode/data-for-orthofinder/OrthoFinder/Results_Mar03/Orthologues/Orthologues_gencode.v49.pc_translations/gencode.v49.pc_translations__v__gencode.vM38.pc_translations.tsv"
mkdir -p "$RBHDIR"

echo "=== listing combos from disk (same script the pipeline uses) ==="
COMBOS_CSV="$RBHDIR/.combos.csv"
"$BINDIR/list_metric_leaderboard_combos.py" --data-dir "$OUTDIR" --output "$COMBOS_CSV"

/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 <<PYEOF
import csv
import subprocess
from pathlib import Path

rbhdir = Path("$RBHDIR")
script = "$BINDIR/compute_rbh_f1_combo.py"
of_tsv = "$OF_TSV"
outdir = "$OUTDIR"

with open("$COMBOS_CSV") as f:
    combos = list(csv.DictReader(f))

print(f"{len(combos)} combos to process", flush=True)
for i, row in enumerate(combos, 1):
    dash_enc, disp_enc, k = row["dash_encoding"], row["display_encoding"], row["ksize"]
    out = rbhdir / f"200_rbh_f1.{dash_enc}.k{k}.csv"
    if out.exists() and out.stat().st_size > 0:
        print(f"[{i}/{len(combos)}] {dash_enc} k={k} -- already done, skipping", flush=True)
        continue
    print(f"[{i}/{len(combos)}] {dash_enc} k={k}", flush=True)
    subprocess.run([
        script,
        "--dash-encoding", dash_enc, "--display-encoding", disp_enc, "--ksize", k,
        "--data-dir", outdir, "--of-tsv", of_tsv, "--output", str(out),
    ], check=True)
PYEOF

echo "=== aggregating (same logic as aggregateRbhF1) ==="
/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 <<'PYEOF'
import polars as pl
from pathlib import Path

rbhdir = Path("/Users/olga/data/gencode/results-human-mouse-orthologs/200_rbh_f1")
frames = [pl.read_csv(f) for f in sorted(rbhdir.glob("200_rbh_f1.*.csv"))]
non_empty = [f for f in frames if f.height > 0]
combined = pl.concat(non_empty, how="diagonal_relaxed") if non_empty else pl.DataFrame()
out = Path("/Users/olga/data/gencode/results-human-mouse-orthologs/200_rbh_f1_all_combos.csv")
combined.write_csv(out)
print(f"{len(non_empty)}/{len(frames)} combo files had rows; {combined.height} total rows -> {out}")
PYEOF

echo "ALL DONE"
