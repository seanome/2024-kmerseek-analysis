#!/bin/bash
#SBATCH --job-name=301-build
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=301-build-%j.out
#
# Notebook 301: build kmerseek at tag v0.4.0 (commit 0aed5ca), make the step 3 decoys, and
# pick the pilot queries. The build steps are the ones 241_alphabet_ranking.sbatch uses on
# Sherlock (modules, and CPATH without LLVM's libc++ headers).
set -euo pipefail
: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
BASE=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}
KMERSEEK_REF=v0.4.0
KMERSEEK_SRC=$SCRATCH/301-kmerseek-$KMERSEEK_REF

module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0 python/3.12.1
LLVM_ROOT=/share/software/user/open/llvm/17.0.6
CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v "^$LLVM_ROOT" || true; } | paste -sd: -)
export CPATH LIBCLANG_PATH=$LLVM_ROOT/lib CC=gcc CXX=g++

if [ ! -x "$KMERSEEK_SRC/target/release/kmerseek" ]; then
    [ -d "$KMERSEEK_SRC/.git" ] || git clone --quiet https://github.com/seanome/kmerseek "$KMERSEEK_SRC"
    (cd "$KMERSEEK_SRC" && git checkout --quiet "$KMERSEEK_REF" && git rev-parse HEAD)
    export CARGO_HOME=$SCRATCH/.cargo
    (cd "$KMERSEEK_SRC" && cargo build --release --bin kmerseek)
fi
"$KMERSEEK_SRC/target/release/kmerseek" --version

python3 -I "$REPO/scripts/build_301_step3_decoys.py" \
    --queries "$BASE/step2/queries.fasta" --truth "$BASE/step2/truth.tsv" --out "$BASE/step3"
mkdir -p "$BASE/pilot"
python3 -I "$REPO/scripts/pilot_301_kmerseek.py" select \
    --step2 "$BASE/step2" --step3 "$BASE/step3" --out "$BASE/pilot/pilot_queries.fasta"
echo "301 build done"
