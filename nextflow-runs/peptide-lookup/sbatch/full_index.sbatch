#!/bin/bash
#SBATCH --job-name=nb300-full-index
#SBATCH --partition=hns,bigmem
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.log
# Memory and time are mandatory arguments from the measured resource table.
set -euo pipefail
: "${OUT:?}" "${KMERSEEK_SHA:?}" "${ALPHABET:?}" "${KSIZE:?}"
module purge
module load system ncurses/6.4 devel gcc/14.2.0 python/3.12.1
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
KMERSEEK=$GROUP_HOME/opt/nb300-kmerseek/$KMERSEEK_SHA/kmerseek
BASE=$OUT/step4/full/${ALPHABET}.k${KSIZE}.s1.C2.X8.masknone
mkdir -p "$OUT/step4/full"
if [ -f "$BASE.complete" ]; then
  echo "Already completed: $BASE"
  exit 0
fi
TASK_TMP=$L_SCRATCH_JOB/nb300-full
mkdir -p "$TASK_TMP"
export TMPDIR=$TASK_TMP
# Results remain on shared scratch. Only transient construction files use local SSD.
test ! -e "$BASE.rocksdb" || { echo "Incomplete index exists: $BASE.rocksdb" >&2; exit 1; }
/usr/bin/time -v -o "$BASE.time.txt" "$KMERSEEK" index \
  --input "$OUT/step1/search_db.fasta" --output "$BASE.rocksdb" \
  --alphabet "$ALPHABET" --ksize "$KSIZE" --scaled 1 \
  --extend-mismatch-penalty 2 --extend-xdrop 8 \
  --ka-queries 200 --ka-seed 300 --ka-null database \
  --ka-reference shuffled-dipeptide --ka-reference-shuffles 1 \
  --ka-survival-out "$BASE.ka_survival.csv" \
  --kmer-stats-out "$BASE.spectrum.csv" > "$BASE.log" 2>&1
if [ -s "$BASE.ka_survival.csv" ]; then
  printf 'fitted\n' > "$BASE.fit_status.txt"
else
  printf 'unfitted\n' > "$BASE.fit_status.txt"
fi
du -sb "$BASE.rocksdb" > "$BASE.disk.txt"
printf '%s\n' "$KMERSEEK_SHA" > "$BASE.source_sha.txt"
touch "$BASE.complete"
