#!/bin/bash
#SBATCH --job-name=352-size
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --exclusive
#SBATCH --constraint=CPU_SKU:7742&NO_GPU
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --output=352-size-%j.out
#
# Notebook 352, one database size: copy the subset to the node's own SSD, build the kmerseek
# index there, run the 1,000 queries. The size comes in SIZE (residues) and the time limit
# with sbatch --time, both set by notebooks/352_submit_all.sh or by the gate.
#
# --exclusive: no other job shares the node while it is timed. CPU_SKU:7742 fixes the node
# type for every size (AMD EPYC 7742, 128 cores, 1,024 GB, 3 TB local SSD; nine hns nodes
# on 2026-10-20). NO_GPU and CUDA_VISIBLE_DEVICES="" keep it CPU only, and apptainer runs
# without --nv.
set -euo pipefail
START=$(date +%s)

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
SIZE=${SIZE:?set SIZE, the database size in residues}
WORK=${NB352_WORK:-$SCRATCH/352-kmerseek-scale}
IMG=${NB352_IMAGE:-$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img}
LOCAL=${L_SCRATCH_JOB:?no node-local scratch}/nb352
OUT=$WORK/results/size_$SIZE
mkdir -p "$LOCAL" "$OUT"
cd "$REPO"

export CUDA_VISIBLE_DEVICES=""
PY=(apptainer exec --bind "/scratch,$L_SCRATCH_JOB" "$IMG" python3 "$REPO/scripts/kmerseek_scale_352.py")
record_budget() {
    "${PY[@]}" budget-add "$WORK/budget.tsv" "size-$SIZE" "${SLURM_JOB_ID:-none}" "$START" "$(date +%s)" || true
}
trap record_budget EXIT

LEFT=$("${PY[@]}" budget-left "$WORK/budget.tsv")
echo "size $SIZE residues; budget left at start: $LEFT h of 48"
if awk -v l="$LEFT" 'BEGIN { exit !(l <= 0) }'; then
    echo "no budget left; not run"
    exit 1
fi

# The node, recorded next to the results.
{
    echo "host	$(hostname)"
    echo "cpu	$(lscpu | grep 'Model name' | sed 's/.*: *//')"
    echo "cores	$(nproc)"
    echo "memory_total_kb	$(awk '/MemTotal/ { print $2 }' /proc/meminfo)"
    echo "local_disk	$(df -B1 --output=size,avail "$L_SCRATCH_JOB" | tail -1)"
    echo "other_jobs_on_node	$(squeue -h -w "$(hostname -s)" -o %i | grep -vc "^${SLURM_JOB_ID}$" || true)"
    echo "gpu_devices	$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)"
    echo "image	$IMG"
    echo "image_sha256	$(sha256sum "$IMG" | cut -d' ' -f1)"
    echo "kmerseek_version	$(apptainer exec "$IMG" kmerseek --version)"
    echo "repo_commit	$(git rev-parse HEAD)"
} | tee "$OUT/node.tsv"
"${PY[@]}" check-cpu

# The job's Slurm end time: run-size stops starting queries 30 min before it, so its
# tables are written before Slurm stops the job.
JOB_END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
echo "job ends at $(date -d "@$JOB_END")"

"${PY[@]}" run-size \
    --size-residues "$SIZE" \
    --subsets-json "$WORK/results/subsets.json" \
    --slice-root "$WORK/slices" \
    --queries-fasta "$WORK/results/queries.fasta" \
    --local-dir "$LOCAL" \
    --out-dir "$OUT" \
    --budget-left-h "$LEFT" \
    --job-end-epoch "$JOB_END"
echo "size $SIZE done"
