#!/bin/bash
#SBATCH --job-name=353-build
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Notebook 353, step 1: build kmerseek v0.4.0 (commit 0aed5ca) and check the inputs.
# Sizing: the same build in notebook 351 took 4 min 52 s at 9.7 GB peak (job 47646815).
NB353_STEP=build
source "$NB353_REPO/analysis/353_measurement_retrieval/sherlock/common.sh"
for f in evidence_proteins.fasta queries_and_controls.fasta kmerseek_settings.tsv; do
    [ -s "$D/in/$f" ] || { echo "353: $D/in/$f missing; copy it from the laptop first" >&2; exit 2; }
done
(cd "$D/in" && sha256sum evidence_proteins.fasta queries_and_controls.fasta kmerseek_settings.tsv)
if [ ! -x "$KMERSEEK" ]; then
    module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0
    # The llvm module puts libc++ headers on CPATH and RocksDB then fails to compile.
    LLVM_ROOT=/share/software/user/open/llvm/17.0.6
    CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v "^$LLVM_ROOT" || true; } | paste -sd: -)
    export CPATH LIBCLANG_PATH=$LLVM_ROOT/lib CC=gcc CXX=g++ CARGO_HOME=$SCRATCH/.cargo
    src=$D/kmerseek-$KMERSEEK_TAG
    [ -d "$src/.git" ] || git clone --quiet --depth 1 --branch "$KMERSEEK_TAG" https://github.com/seanome/kmerseek "$src"
    [ "$(cd "$src" && git rev-parse --short=7 HEAD)" = "$KMERSEEK_COMMIT" ] || { echo "353: $src is not $KMERSEEK_COMMIT" >&2; exit 2; }
    (cd "$src" && cargo build --release --bin kmerseek)
fi
"$KMERSEEK" --version
