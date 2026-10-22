#!/bin/bash
#SBATCH --job-name=nb300-build
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
: "${OUT:?}" "${KMERSEEK_SRC:?}" "${KMERSEEK_SHA:?}"
module purge
module load devel rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0
CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v '/llvm/17.0.6' || true; } | paste -sd: -)
export CPATH LIBCLANG_PATH=/share/software/user/open/llvm/17.0.6/lib CC=gcc CXX=g++
export CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK
export CARGO_HOME=$SCRATCH/.cargo
export CARGO_TARGET_DIR=$OUT/build-target
mkdir -p "$OUT/step4"
cd "$KMERSEEK_SRC"
test "$(git rev-parse HEAD)" = "$KMERSEEK_SHA"
/usr/bin/time -v -o "$OUT/step4/compile.time.txt" cargo build --locked --release --bin kmerseek
INSTALL_DIR=$GROUP_HOME/opt/nb300-kmerseek/$KMERSEEK_SHA
mkdir -p "$INSTALL_DIR"
command cp "$CARGO_TARGET_DIR/release/kmerseek" "$INSTALL_DIR/kmerseek"
printf '%s\n' "$KMERSEEK_SHA" > "$INSTALL_DIR/source_sha.txt"
"$INSTALL_DIR/kmerseek" index --help > "$OUT/step4/index_help.txt"
"$INSTALL_DIR/kmerseek" search --help > "$OUT/step4/search_help.txt"
grep -q massspec18 "$OUT/step4/index_help.txt"
sha256sum "$INSTALL_DIR/kmerseek" > "$OUT/step4/binary_sha256.txt"
