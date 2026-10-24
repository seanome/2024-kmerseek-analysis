#!/bin/bash
#SBATCH --job-name=botryllus-DEkupl-conda-test
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
PREFIX="$GROUP_HOME/opt/botryllus-dekupl-1.3.3-20261024"
export MAMBA_ROOT_PREFIX="$GROUP_HOME/opt/botryllus-mamba"
export MAMBA_DOWNLOAD_THREADS=4
export OMP_NUM_THREADS=4
mkdir -p "$ROOT/dekupl-setup"
if [ -e "$PREFIX" ]; then echo 'Prefix exists; preserve it and inspect before resuming' >&2; exit 1; fi
"$GROUP_HOME/opt/bin/micromamba" create -y -p "$PREFIX" --override-channels --strict-channel-priority -c transipedia -c conda-forge -c bioconda dekupl-run=1.3.3 python=3.10 snakemake-minimal=7.32.4 pulp=2.7 r-base=4.3 bioconductor-deseq2
export PATH="$PREFIX/bin:$PATH"
"$GROUP_HOME/opt/bin/micromamba" list -p "$PREFIX" --explicit > "$ROOT/dekupl-setup/conda-explicit.txt"
"$PREFIX/bin/Rscript" -e 'library(DESeq2); sessionInfo()' > "$ROOT/dekupl-setup/R-session.txt"
sha256sum "$PREFIX/share/dekupl/Snakefile" > "$ROOT/dekupl-setup/Snakefile.sha256"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/botryllus-shazam-code/dekupl-20261024/dekupl_smoke.py" "$ROOT" "$PREFIX"
