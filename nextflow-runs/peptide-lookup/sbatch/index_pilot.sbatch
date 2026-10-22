#!/bin/bash
#SBATCH --job-name=nb300-index-pilot
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
: "${OUT:?}" "${BIN:?}" "${KMERSEEK_SHA:?}" "${SETTINGS:?}"
module purge
module load devel gcc/14.2.0 biology diamond/2.1.23 ncbi-blast+/2.16.0
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
KMERSEEK=$GROUP_HOME/opt/nb300-kmerseek/$KMERSEEK_SHA/kmerseek
PILOT=$OUT/step4/pilot
mkdir -p "$PILOT"
# RocksDB temporary files stay on this job's local disk.
TASK_TMP=$L_SCRATCH_JOB/nb300-pilot
mkdir -p "$TASK_TMP"
export TMPDIR=$TASK_TMP
trap 'command rm -rf "$TASK_TMP"' EXIT
apptainer exec --bind /scratch "$IMG" python3 "$BIN/index_pilot_inputs.py" "$OUT/step1/search_db.fasta" "$PILOT"
while IFS=$'\t' read -r alphabet k; do
  for n in 20000 100000; do
    stem=$PILOT/${alphabet}.k${k}.n${n}
    test ! -e "$stem.complete" || continue
    /usr/bin/time -v -o "$stem.time.txt" "$KMERSEEK" index --input "$PILOT/db_${n}.fasta" \
      --output "$TASK_TMP/index" --alphabet "$alphabet" --ksize "$k" --scaled 1 \
      --extend-mismatch-penalty 2 --extend-xdrop 8 --ka-queries 0 \
      --kmer-stats-out "$stem.spectrum.csv" > "$stem.log" 2>&1
    du -sb "$TASK_TMP/index" > "$stem.disk.txt"
    command rm -rf "$TASK_TMP/index"
    touch "$stem.complete"
  done
done < "$SETTINGS"
