#!/bin/bash
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --output=364-%x-%j.out
#
# Notebook 364, one step per job; submitted and chained by
#   python3 notebooks/364_alphabet_reach.py submit
# from the repo root (STEP, --mem, --time and --cpus-per-task are set there).
#
# Every step runs in one pinned conda environment holding sourmash and the branchwater
# plugin (no container has branchwater 0.9.15; it is on conda-forge only). The fetch step
# builds it with micromamba the first time, on a compute node, not the login node.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${NB364_WORK:-$SCRATCH/364-alphabet-reach}
ENVS=$GROUP_HOME/olgabot/envs
ENV=$ENVS/branchwater-0.9.15
MAMBA=$ENVS/bin/micromamba-2.3.3-0
mkdir -p "$WORK"
echo "host $(hostname); step $STEP"

# A conda env cannot be moved (its scripts hold the prefix), so a marker file, written
# last, says the build finished.
if [ "$STEP" = fetch ] && [ ! -e "$ENV/.complete" ]; then
    mkdir -p "$ENVS/bin"
    if [ ! -x "$MAMBA" ]; then
        curl -fsSL -o "$MAMBA.part" \
            https://github.com/mamba-org/micromamba-releases/releases/download/2.3.3-0/micromamba-linux-64
        chmod +x "$MAMBA.part"
        mv "$MAMBA.part" "$MAMBA"
    fi
    export MAMBA_ROOT_PREFIX=$SCRATCH/.micromamba
    rm -rf "$ENV"
    "$MAMBA" create -y -p "$ENV" --override-channels -c conda-forge -c bioconda \
        python=3.12 sourmash=4.9.4 sourmash_plugin_branchwater=0.9.15
    touch "$ENV/.complete"
fi

export PATH=$ENV/bin:$PATH
sourmash --version
sourmash scripts manysketch --help > /dev/null
PY=(python "$REPO/notebooks/364_alphabet_reach.py")
case "$STEP" in
    fetch) "${PY[@]}" fetch --work "$WORK"; "${PY[@]}" entropy --work "$WORK" ;;
    sketch) /usr/bin/time -v "${PY[@]}" sketch --work "$WORK" --cores "$SLURM_CPUS_PER_TASK" ;;
    compare) /usr/bin/time -v "${PY[@]}" compare --work "$WORK" --cores "$SLURM_CPUS_PER_TASK" ;;
    *) echo "unknown STEP $STEP"; exit 1 ;;
esac
