#!/bin/bash
#SBATCH --job-name=nb354-build
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.log
# Build kmerseek 4d7e27b (seanome/kmerseek PR 156, branch olgabot/massspec18-alphabet: adds the
# massspec18 alphabet, I = L and Q = K) into $KMERSEEK_SRC/target/release/kmerseek.
# Sizing: the same build recipe took 4 min 52 s, peak 9.7 GB, for 982a055 (job 47646815).
set -euo pipefail
: "${KMERSEEK_SRC:?set KMERSEEK_SRC}"
KMERSEEK_SHA=4d7e27b

# python: PyO3 needs >= 3.7 (job 48679463). ncurses: llvm libclang needs libtinfo.so.6, missing on sh04 nodes (job 48678977 failed there).
module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0 python/3.12.1 system ncurses/6.4
# The llvm module puts libc++ headers on CPATH, which breaks RocksDB's C++ build
# (memory note sherlock_kmerseek_build_cpath_fix). Keep gcc's headers; give bindgen libclang.
LLVM_ROOT=/share/software/user/open/llvm/17.0.6
CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v "^$LLVM_ROOT" || true; } | paste -sd: -)
export CPATH LIBCLANG_PATH=$LLVM_ROOT/lib CC=gcc CXX=g++
export CARGO_HOME=$SCRATCH/.cargo

if [ ! -d "$KMERSEEK_SRC/.git" ]; then
  git clone --quiet https://github.com/seanome/kmerseek "$KMERSEEK_SRC"
fi
cd "$KMERSEEK_SRC"
git fetch --quiet origin olgabot/massspec18-alphabet
git checkout --quiet "$KMERSEEK_SHA"
[ "$(git rev-parse --short=7 HEAD)" = "$KMERSEEK_SHA" ]
cargo build --release --bin kmerseek
./target/release/kmerseek --version
./target/release/kmerseek index --help | grep -c massspec18
