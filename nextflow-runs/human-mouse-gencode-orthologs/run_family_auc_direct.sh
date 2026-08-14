#!/bin/bash
# Runs computeFamilyAuc's 10 combos directly (bypassing Nextflow) so this doesn't drag in
# searchHumanVsMouse/convertResultsToParquet -- their storeDir cache in params.outdir was
# emptied out (raw csv.zst deleted), so any `nextflow run -resume` on this .nf file right now
# would redo the full multi-day kmerseek search sweep for ~all 100 combos. All 10 combos below
# already have a .parquet (or .csv.gz for bare "hp" k30) on disk, so this reads existing data
# only -- no search, no Docker, no new raw files.
set -euo pipefail

OUTDIR=/Users/olga/data/gencode/results-human-mouse-orthologs
FAMDIR="$OUTDIR/206_family_auc"
mkdir -p "$FAMDIR"
SCRIPT=/Users/olga/code/2024-kmerseek-analysis/nextflow-runs/human-mouse-gencode-orthologs/bin/compute_family_auc_combo.py

run_combo() {
  dash_enc="$1"; display_enc="$2"; ksize="$3"
  out="$FAMDIR/206_family_auc.${dash_enc}.k${ksize}.csv"
  if [ -s "$out" ]; then
    echo "=== $dash_enc k=$ksize -- already done, skipping ==="
    return
  fi
  echo "=== $dash_enc k=$ksize ==="
  "$SCRIPT" --dash-encoding "$dash_enc" --display-encoding "$display_enc" --ksize "$ksize" \
    --data-dir "$OUTDIR" --output "$out"
}

run_combo protein protein 15
run_combo dayhoff dayhoff 20
run_combo hp hp 30
run_combo hp-lehninger hp_lehninger 30
run_combo hp-thomas-dill hp_thomas_dill 30
run_combo hp-kyte-doolittle hp_kyte_doolittle 30
run_combo hp-thomas-dill-no-c hp_thomas_dill_no_c 30
run_combo hp-lehninger-plus-c hp_lehninger_plus_c 30
run_combo hp-pbotc-1st-ed hp_pbotc_1st_ed 30
run_combo hp-pbotc-1st-ed hp_pbotc_1st_ed 19

echo "=== aggregating ==="
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
