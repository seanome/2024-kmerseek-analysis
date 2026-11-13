#!/bin/bash
#SBATCH --job-name=botryllus-four-sketch-pilot
#SBATCH --partition=hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
PYTHON="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
test -s "$ROOT/raw/SRR32233926/validated.json"
"$PYTHON" "$HOME/botryllus-shazam-code/test_sketches.py"
/usr/bin/time -v "$PYTHON" "$HOME/botryllus-shazam-code/index_reads.py" --run SRR32233926 \
    --reads "$ROOT/raw/SRR32233926/SRR32233926_1.fastq.gz" "$ROOT/raw/SRR32233926/SRR32233926_2.fastq.gz" \
    --output "$ROOT/indexing-pilot/SRR32233926"
