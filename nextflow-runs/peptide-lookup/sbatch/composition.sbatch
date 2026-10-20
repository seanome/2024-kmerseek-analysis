#!/bin/bash
#SBATCH --job-name=nb300-composition
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.log
# Residue counts of one database. Usage: sbatch composition.sbatch <fasta.gz> <name>
# Swiss-Prot takes 17 s of CPU per pass on the laptop; UniRef90 is ~270x larger, ~1.5 h.
set -euo pipefail
: "${OUT:?set OUT}" "${IMG:?set IMG}" "${BIN:?set BIN}"
mkdir -p "$OUT/step0"
apptainer exec --bind /scratch "$IMG" python3 "$BIN/db_composition.py" \
  "$1" "$2" "$OUT/step0/composition_$2.json"
