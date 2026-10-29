# Sourced by every notebook 353 job on Sherlock: paths, the kmerseek build, the CPU-only
# check, and a ledger of wall time per job against the agreed budget.
set -euo pipefail

: "${SCRATCH:?}"
REPO=${NB353_REPO:?set by submit.sh to the repository root}
A=$REPO/analysis/353_measurement_retrieval
D=${NB353_DIR:-$SCRATCH/353_measurement_retrieval}
KMERSEEK_TAG=v0.4.0
KMERSEEK_COMMIT=0aed5ca
KMERSEEK=$D/kmerseek-$KMERSEEK_TAG/target/release/kmerseek
BUDGET_CPU_HOURS=${NB353_BUDGET_CPU_HOURS:-104}
mkdir -p "$D/ledger" "$D/logs"

if [ -n "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${SLURM_GPUS:-}" ]; then
    echo "353: a GPU was allocated to this job; this analysis is CPU only" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES=""

NB353_T0=$(date +%s)
NB353_STEP=${NB353_STEP:-$(basename "$0")}
ledger_write() {
    local status=$? t1 elapsed total
    t1=$(date +%s)
    elapsed=$((t1 - NB353_T0))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:--}" \
        "$NB353_STEP" "$elapsed" "${SLURM_CPUS_PER_TASK:-1}" "$status" \
        > "$D/ledger/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.tsv"
    total=$(cat "$D"/ledger/*.tsv | awk -F'\t' '{s += $4 * $5} END {printf "%.2f", s / 3600}')
    echo "353: this job ${elapsed} s; CPU-hours of all 353 jobs so far: ${total} (proposed ceiling $BUDGET_CPU_HOURS; this line only reports, it does not stop jobs)"
}
trap ledger_write EXIT
