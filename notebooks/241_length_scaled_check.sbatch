#!/bin/bash
#SBATCH --job-name=241-length-scaled
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=241-length-scaled-%j.out
#
# Does the size-scaled whole-protein p-value of seanome/kmerseek PR 136 rank BCL2 (for
# Ced9) and CD47 (for P66) better than the other measures, at every alphabet and k-size
# pair of notebook 241? Builds kmerseek main and PR 136 at pinned commits, indexes the human
# proteome once per pair, searches it with both builds, then writes length_scaled_ranks.csv.
#
# Submit from the same clone as 241_alphabet_ranking.sbatch:
#
#   cd $SCRATCH/241-alphabet-ranking-repo && git pull && sbatch notebooks/241_length_scaled_check.sbatch
#
# Sizing: the forward run (152 indexes, one search and two pair runs each) took 40 minutes
# on a 16-core, 128 GB laptop with 2 workers. This is the same indexes and two searches each,
# plus two kmerseek builds of a few minutes each, so 6 h leaves room for Sherlock cores up to
# 5x slower. Peak memory has never been measured: read it after with
# `sacct -j <jobid> -o JobID,MaxRSS,Elapsed`.
#
# Copy the table back with:
#
#   rsync -av sherlock-dtn:/scratch/users/olgabot/241-length-scaled/length_scaled_ranks.csv \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
export NB241_CHECK_DIR=${NB241_CHECK_DIR:-$SCRATCH/241-length-scaled}
export NB241_HUMAN=${NB241_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
IMG=$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-2026-09-13-extend-chain.img

for f in "$NB241_HUMAN" "$IMG"; do
    [ -s "$f" ] || { echo "241 length check: missing input $f" >&2; exit 2; }
done

module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0 python/3.12.1
export CARGO_HOME=$SCRATCH/.cargo

# kmerseek main at 2c39796 and PR 136 at 6fb49ed, each in its own checkout and target dir.
build() {
    local dir=$1 sha=$2
    if [ ! -x "$dir/target/release/kmerseek" ]; then
        [ -d "$dir/.git" ] || git clone --quiet https://github.com/seanome/kmerseek "$dir"
        (cd "$dir" && git fetch --quiet origin && git checkout --quiet "$sha" \
            && cargo build --release --bin kmerseek)
    fi
}
build "$SCRATCH/241-kmerseek-main" 2c39796
build "$SCRATCH/241-kmerseek-pr136" 6fb49ed
export NB241_MAIN=$SCRATCH/241-kmerseek-main/target/release/kmerseek
export NB241_KMERSEEK=$SCRATCH/241-kmerseek-pr136/target/release/kmerseek
"$NB241_MAIN" --version
"$NB241_KMERSEEK" --version

cd "$REPO/notebooks"
# Each pair's k, penalty and give-up margin come from notebooks/241_plan.json, the first
# run's plan, so no kappa table is read here.
python3 241_length_scaled_check.py --all --workers 2
apptainer exec --bind /scratch "$IMG" python3 241_length_scaled_check.py --all --collect

echo "241 length check: done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
