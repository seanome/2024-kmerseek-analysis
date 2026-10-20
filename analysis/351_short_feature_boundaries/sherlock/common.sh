# Sourced by every notebook 351 job on Sherlock. Paths, images, the CPU-only check, and
# the wall-time ledger that keeps a running total across all jobs.
set -euo pipefail

: "${SCRATCH:?}"
REPO=${NB351_REPO:?set by submit.sh to the repository root}
A=$REPO/analysis/351_short_feature_boundaries
D=${NB351_DIR:-$SCRATCH/351_short_feature_boundaries}
IMG_DIR=$SCRATCH/apptainer-cache
BLAST_IMG=$IMG_DIR/quay.io-biocontainers-blast-2.17.0--hb02a186_1.sif
HMMER_IMG=$IMG_DIR/quay.io-biocontainers-hmmer-3.4--h7d74f8d_5.sif
MMSEQS_IMG=$IMG_DIR/quay.io-biocontainers-mmseqs2-18.8cc5c--hd6d6fdc_0.sif
KMERSEEK_TAG=v0.4.0
KMERSEEK_COMMIT=0aed5ca
KMERSEEK=$D/kmerseek-$KMERSEEK_TAG/target/release/kmerseek
EVALUE=0.01
mkdir -p "$D/ledger" "$D/logs"

# CPU only: no job here asks for a GPU (no --gres), and each one refuses to run if
# Slurm handed it one anyway. CUDA_VISIBLE_DEVICES is emptied so no library can find one.
if [ -n "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${SLURM_GPUS:-}" ]; then
    echo "351: a GPU was allocated to this job; this analysis is CPU only" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES=""

# Wall-time ledger: one line per job or array task, written when it exits.
NB351_T0=$(date +%s)
NB351_STEP=${NB351_STEP:-$(basename "$0")}
ledger_write() {
    local status=$?
    local t1 elapsed total
    t1=$(date +%s)
    elapsed=$((t1 - NB351_T0))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:--}" \
        "$NB351_STEP" "$elapsed" "${SLURM_CPUS_PER_TASK:-1}" "$status" \
        > "$D/ledger/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.tsv"
    total=$(cat "$D"/ledger/*.tsv | awk -F'\t' '{s += $4} END {printf "%.2f", s / 3600}')
    echo "351: this job ${elapsed} s; running total of all 351 jobs so far: ${total} h (budget 48 h)"
}
trap ledger_write EXIT

ap() { local img=$1; shift; apptainer exec --bind /scratch "$img" "$@"; }
