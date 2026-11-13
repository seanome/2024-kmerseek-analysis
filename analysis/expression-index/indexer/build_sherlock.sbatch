#!/bin/bash
#SBATCH -J expr-indexer-build
#SBATCH -p hns
#SBATCH -t 1:00:00
#SBATCH -c 8
#SBATCH --mem 16G
# Build and test the expression indexer on Sherlock.
#   sbatch -o build.%j.log build_sherlock.sbatch   (from this directory)
# The llvm module puts libc++ headers on CPATH, which breaks rocksdb's C++ build; they are
# removed and libclang is pointed at directly (see notebook 283's sbatch).
set -euo pipefail
module load rust llvm/17.0.6 cmake gcc/14.2.0
# sourmash_plugin_branchwater builds pyo3, which needs Python 3.7 or newer; the system one is 3.6
module load python/3.12.1
export PYO3_PYTHON=$(command -v python3)
export CPATH=$(echo "${CPATH:-}" | tr ':' '\n' | { grep -v '/share/software/user/open/llvm/17.0.6' || true; } | paste -sd:)
export LIBCLANG_PATH=/share/software/user/open/llvm/17.0.6/lib CC=gcc CXX=g++
cargo test --release -- --test-threads 4
cargo build --release
ls -la target/release/expression-indexer
