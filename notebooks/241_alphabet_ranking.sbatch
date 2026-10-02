#!/bin/bash
#SBATCH --job-name=241-alphabet-ranking
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=241-alphabet-ranking-%j.out
#
# Notebook 241 on Sherlock: build kmerseek at the commit the first run used, run the
# driver (index, search and pair for every alphabet and k), then the collector that
# writes arms.csv, ranks.csv and regions.parquet.
#
# Submit from a checkout of this branch on Sherlock (the job finds the scripts through
# $SLURM_SUBMIT_DIR). It has its own clone so it never switches the branch under another
# run's checkout. First time only:
#
#   git clone --branch olgabot/alphabet-ranking-three-cases https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/241-alphabet-ranking-repo
#
# Then, each run:
#
#   cd $SCRATCH/241-alphabet-ranking-repo && git pull && sbatch notebooks/241_alphabet_ranking.sbatch
#
# Sizing comes from the first run (2026-09-23), on a 16-core, 128 GB laptop with
# --workers 2: 40 minutes from start to end, and it fit in 128 GB. Same cores, memory and
# workers here. 4 h covers Sherlock cores up to 5x slower plus the kmerseek build, which
# has not been timed on Sherlock. Peak memory was never measured: after this run, read it
# with `sacct -j <jobid> -o JobID,MaxRSS,Elapsed` and shrink --mem.
#
# The notebook needs only the small tables. Copy them back to the laptop with:
#
#   rsync -av sherlock-dtn:/scratch/users/olgabot/241-alphabet-ranking/{arms.csv,ranks.csv,pair} \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
export NB241_DIR=${NB241_DIR:-$SCRATCH/241-alphabet-ranking}
export NB241_HUMAN=${NB241_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
# Byte-identical (sha256 ecfb72b4...) to the laptop copy the first run read.
export NB241_KAPPA=${NB241_KAPPA:-$SCRATCH/2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set/assets/kappa_by_alphabet.tsv}
# kmerseek commit 982a055, the build the first run used. It is on no branch, so it is
# kept on GitHub as this tag.
KMERSEEK_TAG=nb241-kmerseek-982a055
KMERSEEK_SRC=$SCRATCH/241-alphabet-ranking-kmerseek
export NB241_KMERSEEK=$KMERSEEK_SRC/target/release/kmerseek
# Only for the collector, which needs polars (1.43 in this image).
IMG=$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-2026-09-13-extend-chain.img

for f in "$REPO/notebooks/241_alphabet_ranking_driver.py" "$NB241_HUMAN" "$NB241_KAPPA" \
         "$NB241_DIR/queries.fa" "$IMG"; do
    [ -s "$f" ] || { echo "241: missing input $f" >&2; exit 2; }
done

# rust for the build, llvm for RocksDB's bindings, gcc for RocksDB's C++ and for
# libstdc++ when kmerseek runs, python for the driver (standard library only).
module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0 python/3.12.1

if [ ! -x "$NB241_KMERSEEK" ]; then
    [ -d "$KMERSEEK_SRC/.git" ] || \
        git clone --quiet --depth 1 --branch "$KMERSEEK_TAG" https://github.com/seanome/kmerseek "$KMERSEEK_SRC"
    export CARGO_HOME=$SCRATCH/.cargo
    (cd "$KMERSEEK_SRC" && cargo build --release --bin kmerseek)
fi
"$NB241_KMERSEEK" --version

cd "$REPO/notebooks"
python3 241_alphabet_ranking_driver.py --workers 2
apptainer exec --bind /scratch "$IMG" python3 241_alphabet_ranking_collect.py

echo "241: done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
