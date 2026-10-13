#!/bin/bash
#SBATCH --job-name=286-step1-pilot
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=286-step1-pilot-%j.out
#
# Notebook 286, first job: step 1 (Ced9, P66 and 20 shuffled copies of each against the
# human proteome at notebook 241's 152 settings), the check against notebook 241, a timed
# pilot of step 2, and the plan. With SUBMIT=1 (the default) it then submits the step 2
# arrays sized from the pilot and the score job (286_step2.sbatch, 286_score.sbatch).
#
# Needs notebook 283's rebuilt indexes ($SCRATCH/283-bhf-low-complexity/indexes/idx) and
# the kmerseek 982a055 build; the script stops if any of the 152 indexes is missing.
# Submit from a checkout of olgabot/intermediate-sequence-286, after 283's index job:
#
#   git clone --branch olgabot/intermediate-sequence-286 https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/286-repo
#   cd $SCRATCH/286-repo && sbatch --dependency=afterok:<283 job id> notebooks/286_intermediate_sequence.sbatch
#
# Sizing, from measured history (no earlier run of this search exists):
#   step1  42 queries (21 of 280 aa, 21 of 597 aa), 18_417 residues. Notebook 256 searched
#          301 queries of 252 aa (75_852 residues) at the same 152 settings in 20 min on a
#          16-core laptop with 3 processes; this is 24% of that, about 5 min there.
#   pilot  150 random human proteins (seed 286) and the 3 longest (TTN 35_991, MUC16
#          14_507, OBSCN 8_925 aa): 143_889 residues, 1.3% of the proteome's 11_347_201.
#          By 256's rate per residue the whole proteome is about 50 h on that laptop, so
#          the pilot is about 40 min there; 1-3 h here if Sherlock's cores are up to 4x slower.
#   memory 256's 3 processes fit in 128 GB (their peak was never measured). The pilot has
#          4 processes and the longest human proteins (TTN, 35_991 aa), so 256 GB. The pilot
#          records every process's peak, and the step 2 asks come from it.
#   After the run: sacct -j <jobid> -o JobID,MaxRSS,Elapsed
#
# Copy back only the small tables (after the score job):
#   ssh sherlock-dtn mkdir -p /scratch/users/olgabot/286-intermediate-sequence/results
#   pixi exec rsync -av sherlock-dtn:/scratch/users/olgabot/286-intermediate-sequence/{results,pilot.csv,step2_plan.csv,step1_timing.csv} \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/286/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
export NB286_DIR=${NB286_DIR:-$SCRATCH/286-intermediate-sequence}
export NB241_KMERSEEK=${NB241_KMERSEEK:-$SCRATCH/241-alphabet-ranking-kmerseek/target/release/kmerseek}
export NB241_IDX=${NB241_IDX:-$SCRATCH/283-bhf-low-complexity/indexes/idx}
export NB241_HUMAN=${NB241_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
export NB256_ARMS=${NB256_ARMS:-$SCRATCH/241-alphabet-ranking/null_bhf_dipeptide/arms.json}
export NB241_QUERIES=${NB241_QUERIES:-$SCRATCH/241-alphabet-ranking/queries.fa}
export NB241_REGIONS=${NB241_REGIONS:-$SCRATCH/241-alphabet-ranking/regions.parquet}
WORKERS=${NB286_WORKERS:-4}
export NB286_WORKERS=$WORKERS
export RAYON_NUM_THREADS=$((SLURM_CPUS_PER_TASK / WORKERS))
# notebook 283's venv (polars 1.34.0, numpy, matplotlib) on the python/3.12.1 module
PY=$HOME/venvs/nb283/bin/python
module load python/3.12.1
# kmerseek 982a055 needs a newer libstdc++ than the system's (283's build used gcc 14)
module load gcc/14.2.0

for f in "$NB241_KMERSEEK" "$NB241_HUMAN" "$NB256_ARMS" "$NB241_QUERIES" "$NB241_REGIONS" "$PY"; do
    [ -s "$f" ] || { echo "286: missing input $f" >&2; exit 2; }
done
"$NB241_KMERSEEK" --version

cd "$SLURM_SUBMIT_DIR/notebooks"
$PY 286_intermediate_sequence.py step1 --workers "$WORKERS"
$PY 286_intermediate_sequence.py check
$PY 286_intermediate_sequence.py pilot --workers "$WORKERS"
if [ "${SUBMIT:-1}" = 1 ]; then
    $PY 286_intermediate_sequence.py plan --workers "$WORKERS" --cpus "$SLURM_CPUS_PER_TASK" --submit
else
    $PY 286_intermediate_sequence.py plan --workers "$WORKERS" --cpus "$SLURM_CPUS_PER_TASK"
fi
echo "286: step 1, check, pilot and plan done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
