#!/bin/bash
#SBATCH --job-name=241-rerun-settings
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=241-rerun-settings-%j.out
#
# Notebook 241: search a few settings again in the existing run folder, then rebuild
# arms.csv, ranks.csv, top_hits.csv and regions.parquet from all 152 search CSVs.
#
# Written for hp_lehninger_hpc3 k=21 (2026-10-20). Its search CSV came from an extended
# search with a fit the first run of 2026-09-23 had; the run after it marked the setting
# unfitted and kept the CSV. The driver now notices this and searches it with exact
# regions, as every other unfitted setting was.
#
# 241's own indexes were deleted; notebook 283 rebuilt them with the same kmerseek
# (982a055) and flags. Search opens an index read-only, so linking 283's copy is safe
# while 283 runs. Submit from a checkout of this branch:
#
#   NB241_ONLY=hp_lehninger_hpc3.k21 sbatch notebooks/241_rerun_settings.sbatch
#
# The tables this replaces are copied first to $NB241_DIR/before_rerun_<job id>/.
#
# Sizing: the driver part is one index-free search, 2 s on 2026-09-23. The collector
# has never been timed alone; the whole 241 job of 2026-10-16 (index, search, collect)
# peaked at 69 GB and took 3 h, so 96 GB covers the collector with room.
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}" "${NB241_ONLY:?set NB241_ONLY=alphabet.kN[,...]}"
REPO=$SLURM_SUBMIT_DIR
export NB241_DIR=${NB241_DIR:-$SCRATCH/241-alphabet-ranking}
export NB241_HUMAN=${NB241_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
export NB241_KAPPA=${NB241_KAPPA:-$SCRATCH/2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set/assets/kappa_by_alphabet.tsv}
KMERSEEK_SRC=${KMERSEEK_SRC:-$SCRATCH/241-alphabet-ranking-kmerseek}
export NB241_KMERSEEK=$KMERSEEK_SRC/target/release/kmerseek
IDX_FROM=${NB241_IDX_FROM:-$SCRATCH/283-bhf-low-complexity/indexes/idx}
IMG=$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-2026-09-13-extend-chain.img

for f in "$NB241_HUMAN" "$NB241_KAPPA" "$NB241_DIR/queries.fa" "$NB241_DIR/plan.json" \
         "$NB241_KMERSEEK" "$IMG"; do
    [ -s "$f" ] || { echo "241: missing input $f" >&2; exit 2; }
done
# The build must be the one the first run used (Sherlock's git 1.8 has no `git -C`).
built=$(cd "$KMERSEEK_SRC" && git rev-parse --short=7 HEAD)
[ "$built" = 982a055 ] || { echo "241: kmerseek at $built, not 982a055" >&2; exit 2; }

mkdir -p "$NB241_DIR/idx"
for tag in ${NB241_ONLY//,/ }; do
    idx=human.$tag.rocksdb
    if [ ! -e "$NB241_DIR/idx/$idx" ]; then
        [ -s "$IDX_FROM/$idx/CURRENT" ] || { echo "241: no index $IDX_FROM/$idx" >&2; exit 2; }
        ln -s "$IDX_FROM/$idx" "$NB241_DIR/idx/$idx"
    fi
done

keep=$NB241_DIR/before_rerun_$SLURM_JOB_ID
mkdir -p "$keep"
for f in arms.csv ranks.csv top_hits.csv regions.parquet runs.jsonl; do
    if [ -e "$NB241_DIR/$f" ]; then cat "$NB241_DIR/$f" > "$keep/$f"; fi
done

module load python/3.12.1 gcc/14.2.0
cd "$REPO/notebooks"
python3 241_alphabet_ranking_driver.py --only "$NB241_ONLY" --workers 1
apptainer exec --bind /scratch "$IMG" python3 241_alphabet_ranking_collect.py

echo "241: done. Old tables in $keep. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
