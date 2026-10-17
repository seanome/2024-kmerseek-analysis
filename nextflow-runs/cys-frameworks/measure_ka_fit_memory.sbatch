#!/bin/bash
# Measure the memory of a cys2 k16 kmerseek index, whose Karlin-Altschul fit was killed at
# 700 GB on every full Swiss-Prot reference (run-phylum, 2026-10-16), with the fit that
# counts regions per score bin as they are found (seanome/kmerseek branch
# olgabot/cys-runs-040-ka-bins, the cys-runs-040 build branch plus that change).
#
# Two indexes, same command the pipeline's indexReference runs:
#   1. the pilot's Cnidaria reference (27,570 entries). The old build peaked at 27.1 GB
#      here and printed its fit; the new one must print the same fit.
#   2. the full Arthropoda-removed reference (561,674 entries), killed at 700 GB before.
#
# Submit from a Sherlock login node:
#   sbatch nextflow-runs/cys-frameworks/measure_ka_fit_memory.sbatch
#SBATCH --job-name=cys2-ka-fit-memory
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=cys2-ka-fit-memory.%j.log

set -euo pipefail
KMERSEEK_REF=${KMERSEEK_REF:-olgabot/cys-runs-040-ka-bins}
WORK=$SCRATCH/cys2-ka-fit-memory
RUNS=$SCRATCH/2024-kmerseek-analysis-cys-frameworks/nextflow-runs/cys-frameworks
PILOT_REF=$RUNS/pilot/work/88/1889138ba2588b1e92bd20cce41b7c/reference.fasta.gz
FULL_REF=$RUNS/run-phylum/work/d0/16b6c5f494081bd884650fccc82083/reference.fasta.gz
mkdir -p "$WORK"
cd "$WORK"

# Build, with the modules notebooks/283_bhf_low_complexity_control.sbatch (PR 124) builds
# kmerseek with. python/3.12.1 is there for libtinfo.so.6, which libclang needs: without it
# bindgen fails with "libtinfo.so.6: cannot open shared object file" (job 48247472). The
# llvm module puts libc++ headers on CPATH, which breaks librocksdb-sys under g++; strip
# them (memory note sherlock_kmerseek_build_cpath_fix).
module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0 python/3.12.1
LLVM_ROOT=/share/software/user/open/llvm/17.0.6
CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v "^$LLVM_ROOT" || true; } | paste -sd: -)
export CPATH LIBCLANG_PATH=$LLVM_ROOT/lib CC=gcc CXX=g++ CARGO_HOME=$SCRATCH/.cargo
[ -d kmerseek ] || git clone -q https://github.com/seanome/kmerseek.git
(cd kmerseek && git fetch -q origin && git checkout -q "origin/$KMERSEEK_REF" && git log --oneline -1)
(cd kmerseek && cargo build --release --no-default-features --bins 2>&1 | tail -2)
BIN=$WORK/kmerseek/target/release/kmerseek

index() {
    local name=$1 ref=$2
    command rm -rf "$name.index.db"
    /usr/bin/time -v "$BIN" index --input "$ref" --output "$name.index.db" --ksize 16 \
        --alphabet cys2 --scaled 1 --remove-low-complexity 2> "$name.err"
    grep -E "Fitting|fitted|Stored|refused|too few" "$name.err" || true
    grep -E "Maximum resident|Elapsed \(wall" "$name.err"
}

echo "== pilot Cnidaria reference"
index pilot "$PILOT_REF"
echo "== full Arthropoda-removed reference"
index full_arthropoda "$FULL_REF"
