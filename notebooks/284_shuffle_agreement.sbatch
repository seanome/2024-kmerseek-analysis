#!/bin/bash
#SBATCH --job-name=284-shuf-agree
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=284-shuf-agree-%j.out
#
# Notebook 284's shuffled-query control: Ced9, P66 and BHF plus 20 dipeptide-shuffled
# copies of each (63 queries), searched against the human proteome at notebook 241's 152
# settings with every region kept, then summarised as alphabet agreement per query.
#
# Needs notebook 283's rebuilt indexes ($SCRATCH/283-bhf-low-complexity/indexes/idx) and
# its kmerseek 982a055 build; the script stops if any of the 152 indexes is missing.
# Submit after 283 (PR 124), from a checkout of olgabot/alphabet-agreement-explainer:
#
#   git clone --branch olgabot/alphabet-agreement-explainer https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/284-agreement-repo
#   cd $SCRATCH/284-agreement-repo && sbatch --dependency=afterany:<283 job id> notebooks/284_shuffle_agreement.sbatch
#
# Sizing, from measured history:
#   search     notebook 256 searched 301 queries in 548 chunks in 20 min wall with 3 workers
#              on the 128 GB laptop; this run has 63 queries in about 220 chunks but keeps every
#              region, so the same order: under 1 h here even with cores 3x slower. Memory: the
#              same 3-worker search that ran in 128 GB; peak never measured, so the same ask.
#   summarise  one query at two cuts took about 30 s on the laptop (Ced9, 2.3 M regions);
#              63 queries x 4 cuts, about 1 h, 2-3 h here.
#   Disk: about 3 regions tables of 100-150 MB per query, roughly 8 GB in $SCRATCH.
#   After the run: sacct -j <jobid> -o JobID,MaxRSS,Elapsed, and shrink --mem for a rerun.
#
# Copy back only the two small tables:
#   pixi exec rsync -av sherlock-dtn:/scratch/users/olgabot/284-shuffle-agreement/284_shuffle_agreement_*.csv \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/284/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
export NB284_DIR=${NB284_DIR:-$SCRATCH/284-shuffle-agreement}
export NB241_KMERSEEK=${NB241_KMERSEEK:-$SCRATCH/241-alphabet-ranking-kmerseek/target/release/kmerseek}
export NB241_IDX=${NB241_IDX:-$SCRATCH/283-bhf-low-complexity/indexes/idx}
export NB256_ARMS=${NB256_ARMS:-$SCRATCH/241-alphabet-ranking/null_bhf_dipeptide/arms.json}
export NB241_QUERIES=${NB241_QUERIES:-$SCRATCH/241-alphabet-ranking/queries.fa}
# notebook 283's venv (polars 1.34.0, numpy, matplotlib) on the python/3.12.1 module
PY=$HOME/venvs/nb283/bin/python
module load python/3.12.1

for f in "$NB241_KMERSEEK" "$NB256_ARMS" "$NB241_QUERIES" "$PY"; do
    [ -s "$f" ] || { echo "284: missing input $f" >&2; exit 2; }
done
"$NB241_KMERSEEK" --version

cd "$SLURM_SUBMIT_DIR/notebooks"
$PY 284_shuffle_agreement.py search --workers 3
$PY 284_shuffle_agreement.py summarise
echo "284: done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
