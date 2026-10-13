#!/bin/bash
#SBATCH --job-name=283-bhf-lc
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=283-bhf-lc-%j.out
#
# Notebook 283 on Sherlock: build kmerseek 982a055 (the build notebooks 241, 242 and 256
# used), rebuild notebook 241's 152 human indexes with 241's own driver, check the rebuilt
# indexes reproduce notebook 256 on 6 settings, then search BHF and 300 copies twice:
# copies that keep BHF's low-complexity stretches (null 2) and BHF and copies without them
# (null 3). Every step resumes: a finished index or search chunk is not redone.
#
# Submit from a checkout of olgabot/bhf-low-complexity-control:
#
#   git clone --branch olgabot/bhf-low-complexity-control https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/283-bhf-lc-repo
#   cd $SCRATCH/283-bhf-lc-repo && sbatch notebooks/283_bhf_low_complexity_control.sbatch
#
# BUILD_ONLY=1 stops after the kmerseek build (sbatch --export=ALL,BUILD_ONLY=1 ...).
#
# Sizing, from measured history:
#   build    the laptop builds 982a055 in 1 min; 10 min allowed here.
#   indexes  241's runs.jsonl: 49 min of index time, 24 min of fit retries and 9 min of
#            search over its 152 settings; 40 min wall on a 16-core, 128 GB laptop with 2
#            workers, which fit in 128 GB (peak never measured, so the same ask here).
#   searches 256's 548 search chunks took 20 min wall with 3 workers; two new nulls of the
#            same size, 40 min, plus 6 settings for the reproduction check.
#   About 1.5 h on the laptop; 8 h covers Sherlock cores up to 4x slower. Disk: about
#   55 GB of indexes in $SCRATCH. After the run, read the real peak with
#   sacct -j <jobid> -o JobID,MaxRSS,Elapsed and shrink --mem for any rerun.
#
# The notebook reads only the top/ tables (about 10 MB per null). Copy them back with:
#   rsync -av sherlock-dtn:/scratch/users/olgabot/283-bhf-low-complexity/{null_kept,null_removed,reproduce_null1} \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/283/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
export NB283_DIR=${NB283_DIR:-$SCRATCH/283-bhf-low-complexity}
export NB241_HUMAN=${NB241_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
# md5 d0774e68fbf8aa38cc4c2bfbfe6280df, the same file as the laptop copy 241 read.
export NB241_KAPPA=${NB241_KAPPA:-$SCRATCH/2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set/assets/kappa_by_alphabet.tsv}
# 256's run folder: queries.fa (BHF and its 300 copies), arms.json and top/.
export NB256_RUN=${NB256_RUN:-$SCRATCH/241-alphabet-ranking/null_bhf_dipeptide}
KMERSEEK_TAG=nb241-kmerseek-982a055
export KMERSEEK_SRC=$SCRATCH/241-alphabet-ranking-kmerseek
export NB241_KMERSEEK=$KMERSEEK_SRC/target/release/kmerseek
# polars 1.34.0 (the laptop's), numpy 2.2.6 and matplotlib 3.10.9 in a small venv on the
# python/3.12.1 module; 256's search code imports all three. numpy 2.3 has no wheel for
# Sherlock's glibc 2.17, so the newest with one. Made once, on a login node (2026-10-13):
#   module load python/3.12.1 && python3 -m venv $HOME/venvs/nb283 && \
#     $HOME/venvs/nb283/bin/pip install --upgrade pip && \
#     $HOME/venvs/nb283/bin/pip install --only-binary=:all: polars==1.34.0 'numpy<2.4' 'matplotlib<3.11'
PY=$HOME/venvs/nb283/bin/python

for f in "$NB241_HUMAN" "$NB241_KAPPA" "$NB256_RUN/queries.fa" "$NB256_RUN/arms.json" "$NB256_RUN/../queries.fa" "$PY"; do
    [ -s "$f" ] || { echo "283: missing input $f" >&2; exit 2; }
done
[ "$(ls "$NB256_RUN/top" | wc -l)" -eq 548 ] || { echo "283: $NB256_RUN/top does not hold 256's 548 tables" >&2; exit 2; }

# rust for the build, llvm for bindgen's libclang, gcc for RocksDB's C++ and libstdc++.
module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0 python/3.12.1
# The llvm module puts LLVM's libc++ headers on CPATH, and g++ then compiles RocksDB
# against libc++'s stdlib.h ("'abort' has not been declared in 'std'"); both 241 jobs of
# 2026-10-06 failed this way. Keep only gcc's headers, and give bindgen libclang directly.
LLVM_ROOT=/share/software/user/open/llvm/17.0.6
CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v "^$LLVM_ROOT" || true; } | paste -sd: -)
export CPATH LIBCLANG_PATH=$LLVM_ROOT/lib CC=gcc CXX=g++

if [ ! -x "$NB241_KMERSEEK" ]; then
    [ -d "$KMERSEEK_SRC/.git" ] || \
        git clone --quiet --depth 1 --branch "$KMERSEEK_TAG" https://github.com/seanome/kmerseek "$KMERSEEK_SRC"
    export CARGO_HOME=$SCRATCH/.cargo
    (cd "$KMERSEEK_SRC" && [ "$(git rev-parse --short=7 HEAD)" = 982a055 ] && cargo build --release --bin kmerseek)
fi
"$NB241_KMERSEEK" --version
[ "${BUILD_ONLY:-0}" = 1 ] && { echo "283: build only, done"; exit 0; }

cd "$REPO/notebooks"
export MPLBACKEND=Agg
$PY -c "import polars, numpy, matplotlib"
$PY 283_bhf_low_complexity_control.py indexes --workers 2
$PY 283_bhf_low_complexity_control.py reproduce --workers 3
$PY 283_bhf_low_complexity_control.py search --null kept --workers 3
$PY 283_bhf_low_complexity_control.py search --null removed --workers 3

echo "283: done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
