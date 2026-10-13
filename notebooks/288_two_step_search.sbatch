#!/bin/bash
#SBATCH --job-name=288-two-step
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --output=288-two-step-%j.out
#
# Notebook 288 on Sherlock. STEP picks what runs; resources go on the sbatch line, sized below.
#
#   STEP=pairwise  small indexes (BCL2, CD47, BCL-2 family) x 70 settings, 202 queries.
#                  sbatch --export=ALL,STEP=pairwise -c 8 --mem=16G --time=01:00:00 ...
#   STEP=heavy     the 70 human indexes, the proteome search of 202 queries, then step 2.
#                  sbatch --export=ALL,STEP=heavy -c 16 --mem=128G --time=10:00:00 ...
#
# Submit both with notebooks/288_submit.sh, which holds the heavy job until the pairwise
# one succeeds. From a clone of olgabot/two-step-search-288:
#   git clone --branch olgabot/two-step-search-288 https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/288-two-step-repo
#   cd $SCRATCH/288-two-step-repo && bash notebooks/288_submit.sh
#
# Sizing, from notebook 241's runs.jsonl (laptop, 3 queries, the same 70 settings):
#   indexes  2_091 s of index time with the 200-query fit; here no fit, 2 workers.
#   search   199 s for 3 queries; 202 queries is about 67x, 3.7 h of one process,
#            3 workers. 283 asked 128 GB for 2-3 such workers on this proteome; no peak
#            was ever measured, so the same ask. Read it after with
#            sacct -j <jobid> -o JobID,MaxRSS,Elapsed and shrink any rerun.
#   Disk: about 25 GB of human indexes in $NB288_DIR/idx (241's 152 were 55 GB).
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}" "${STEP:?set STEP=pairwise or STEP=heavy}"
REPO=$SLURM_SUBMIT_DIR
export NB288_DIR=${NB288_DIR:-$SCRATCH/288-two-step-search}
export NB241_DIR=${NB241_DIR:-$SCRATCH/241-alphabet-ranking}
export NB241_KMERSEEK=${NB241_KMERSEEK:-$SCRATCH/241-alphabet-ranking-kmerseek/target/release/kmerseek}
export NB241_HUMAN=${NB241_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
# notebook 283's venv: polars 1.34.0 on the python/3.12.1 module (needs the module loaded)
module load python/3.12.1
PY=$HOME/venvs/nb283/bin/python

for f in "$NB241_KMERSEEK" "$NB241_HUMAN" "$NB241_DIR/queries.fa" "$NB241_DIR/null_bhf_dipeptide/arms.json" "$PY"; do
    [ -s "$f" ] || { echo "288: missing input $f" >&2; exit 2; }
done
echo "e68d27427d1b119d9ca1999803fa20e9  $NB241_HUMAN" | md5sum -c -
"$NB241_KMERSEEK" --version
(cd "$REPO" && git log -1 --format='288 code: %h %s')  # Sherlock's git 1.8 has no -C

S=$REPO/notebooks/288_two_step_search.py
case "$STEP" in
    pairwise) "$PY" "$S" pairwise --workers "$SLURM_CPUS_PER_TASK" ;;
    heavy)
        [ -s "$NB288_DIR/queries.fa" ] || { echo "288: run STEP=pairwise first (it writes queries.fa)" >&2; exit 2; }
        "$PY" "$S" indexes --workers 2
        "$PY" "$S" search --workers 3
        "$PY" "$S" twostep --workers 8 ;;
    *) echo "288: unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "288: $STEP done"
