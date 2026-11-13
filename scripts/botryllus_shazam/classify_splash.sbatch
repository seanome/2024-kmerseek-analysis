#!/bin/bash
#SBATCH --job-name=botryllus-SPLASH-DE-evidence
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/dekupl-20261024"
GENOME="$ROOT/references/botznik-chr.fa"
if [ ! -e "$GENOME" ]; then
    /bin/cp "$GROUP_HOME/Botryllus_Genome/botznik-chr.fa" "$GENOME.partial"
    mv "$GENOME.partial" "$GENOME"
fi
"$GROUP_HOME/opt/bin/minimap2" --version
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/classify_splash.py" "$ROOT" "$CODE" "$GENOME" "$GROUP_HOME/opt/bin/minimap2"
