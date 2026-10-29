#!/bin/bash
#SBATCH --job-name=botryllus-DEkupl-setup-test
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
mkdir -p "$ROOT/images" "$ROOT/dekupl-setup"
export APPTAINER_CACHEDIR="$ROOT/images/cache"
export APPTAINER_TMPDIR="$L_SCRATCH_JOB"
IMAGE="$ROOT/images/dekupl-run-1.3.5.sif"
if [ ! -e "$IMAGE" ]; then
    apptainer pull "$IMAGE.partial" docker://transipedia/dekupl-run:1.3.5
    mv "$IMAGE.partial" "$IMAGE"
fi
sha256sum "$IMAGE" > "$ROOT/dekupl-setup/image.sha256"
apptainer exec "$IMAGE" Rscript -e 'library(DESeq2); sessionInfo()' > "$ROOT/dekupl-setup/R-session.txt"
apptainer exec "$IMAGE" sh -c 'snakemake --version; jellyfish --version; /dekupl/bin/kallisto version; sha256sum /dekupl/Snakefile; cat /dekupl/bin/DESeq2_ref_transcripts.R' > "$ROOT/dekupl-setup/programs.txt"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/botryllus-shazam-code/dekupl-20261024/dekupl_smoke.py" "$ROOT" "$IMAGE"
