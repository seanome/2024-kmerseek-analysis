#!/bin/bash
#SBATCH --job-name=352-gate
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --constraint=NO_GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=352-gate-%j.out
#
# Notebook 352: runs after the 10 billion residue step. Predicts the third size's job time,
# memory and local disk from that step's measurements (a linear prediction x 1.2; these
# are predictions, written to results/gate.json as such) and submits the largest of 100,
# 50, 30 and 20 billion residues that fits the 48 h budget, 950 GB and 2.8 TB. If none
# fits, nothing more runs. Rule: notebooks/352_data/preregistered.yaml, budget.gate.
# Sizing: reads two small JSON files; seconds, well under 1 G.
set -euo pipefail
START=$(date +%s)

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${NB352_WORK:-$SCRATCH/352-kmerseek-scale}
IMG=${NB352_IMAGE:-$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img}
cd "$REPO"
export CUDA_VISIBLE_DEVICES=""
PY=(apptainer exec --bind /scratch "$IMG" python3 "$REPO/scripts/kmerseek_scale_352.py")
trap '"${PY[@]}" budget-add "$WORK/budget.tsv" gate "${SLURM_JOB_ID:-none}" "$START" "$(date +%s)" || true' EXIT

"${PY[@]}" gate "$WORK/results/size_10000000000/summary.json" "$WORK/budget.tsv" \
    "$WORK/results/subsets.json" "$WORK/results/gate.json" | tee "$WORK/results/gate.log"
THIRD=$(sed -n 's/^THIRD_SIZE=//p' "$WORK/results/gate.log")
if [ "$THIRD" = 0 ]; then
    echo "No third size fits the budget, memory and disk. Nothing more submitted."
    echo "gate: no third size" >> "$WORK/job_ids.txt"
    exit 0
fi
# Time limit: 1.5 x the predicted job hours, rounded up, capped at the hours left.
HOURS=$(apptainer exec --bind /scratch "$IMG" python3 -c '
import json, math
g = json.load(open("'"$WORK"'/results/gate.json"))
p = [r for r in g["predictions"] if r["size_residues"] == g["chosen_size_residues"]][0]
left = 48 - g["used_h"]
print(max(1, min(math.ceil(1.5 * p["predicted_job_h"]), math.floor(left))))')
JOB=$(sbatch --parsable --chdir "$REPO" --export=ALL,SIZE="$THIRD" --time="$HOURS:00:00" notebooks/352_size.sbatch)
echo "third size $THIRD residues submitted as job $JOB with --time=$HOURS:00:00"
echo "size $THIRD	$JOB" >> "$WORK/job_ids.txt"
