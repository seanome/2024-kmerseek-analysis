#!/bin/bash
#SBATCH --job-name=oyster-mvp-kallisto-index
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
# kallisto 0.51.1 index of oyster_v9 transcripts (k 31, the default). Built once.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
idx="$ROOT/references/oyster_v9.kallisto.idx"
[ -f "$idx" ] && { echo "index present"; exit 0; }
timed kallisto-index "$SHAZAM_ENV/bin/kallisto" index -t 8 -i "$idx.part" "$ROOT/references/rna.fna.gz"
mv "$idx.part" "$idx"
