#!/bin/bash
#SBATCH --job-name=nb300-comparator-indexes
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
: "${OUT:?}"
module purge
module load biology diamond/2.1.23 ncbi-blast+/2.16.0
S4=$OUT/step4
mkdir -p "$S4/comparators"
TASK_TMP=$L_SCRATCH_JOB/nb300-comparators
mkdir -p "$TASK_TMP"
export TMPDIR=$TASK_TMP
trap 'command rm -rf "$TASK_TMP"' EXIT
command cp "$OUT/step1/search_db.fasta" "$TASK_TMP/db.fasta"
diamond version > "$S4/comparators/versions.txt"
blastp -version >> "$S4/comparators/versions.txt"
if [ ! -f "$S4/comparators/diamond.complete" ]; then
  /usr/bin/time -v -o "$S4/comparators/diamond.time.txt" diamond makedb \
    --in "$TASK_TMP/db.fasta" --db "$TASK_TMP/diamond" --threads "$SLURM_CPUS_PER_TASK"
  command cp "$TASK_TMP/diamond.dmnd" "$S4/comparators/diamond.dmnd"
  touch "$S4/comparators/diamond.complete"
fi
if [ ! -f "$S4/comparators/blast.complete" ]; then
  /usr/bin/time -v -o "$S4/comparators/blast.time.txt" makeblastdb \
    -in "$TASK_TMP/db.fasta" -dbtype prot -parse_seqids -out "$TASK_TMP/blast" -title nb300_search_db
  blastdbcmd -db "$TASK_TMP/blast" -info > "$S4/comparators/blast_info.txt"
  command cp "$TASK_TMP"/blast.* "$S4/comparators/"
  touch "$S4/comparators/blast.complete"
fi
