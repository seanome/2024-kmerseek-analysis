#!/bin/bash
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --output=364-%x-%A_%a.out
#
# Notebook 364, one step per job; submitted and chained by
#   python3 notebooks/364_alphabet_reach.py submit
# from the repo root (STEP, --mem and --time are set there).
#
# Sizing (2026-11-08): notebook 299 measured `sourmash sketch translate` on Sherlock at
# about 4.2 min per GB per setting with 2-3 GB resident (singleton sketches,
# $SCRATCH/299-omrgc-pilot/step1/*.time). The largest genome here is 2.7 GB, six settings,
# so about 1-1.5 h per translate task; 6 h covers a 4x miss. One whole-genome sketch at
# scaled 200 holds up to ~80 M hashes per setting (2.7e9 x 6 frames / 200), ~4 GB for six,
# so 32 GB. compare loads seven such sketches at once: 64 GB.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${NB364_WORK:-$SCRATCH/364-alphabet-reach}
IMG=$GROUP_HOME/olgabot/containers/sourmash_4.9.4.sif
mkdir -p "$WORK"
echo "host $(hostname); step $STEP; task ${SLURM_ARRAY_TASK_ID:-}"

if [ "$STEP" = fetch ] && [ ! -s "$IMG" ]; then
    export APPTAINER_CACHEDIR=$SCRATCH/.apptainer-cache
    apptainer pull "$IMG.part" docker://quay.io/biocontainers/sourmash:4.9.4--hdfd78af_0
    mv "$IMG.part" "$IMG"
fi

# /usr/bin/time cannot time a shell function, so the command is an array.
RUN=(apptainer exec --bind /scratch,/home/groups "$IMG" python "$REPO/notebooks/364_alphabet_reach.py")
case "$STEP" in
    fetch) "${RUN[@]}" fetch --work "$WORK"; "${RUN[@]}" entropy --work "$WORK" ;;
    sketch) /usr/bin/time -v "${RUN[@]}" sketch --work "$WORK" --index "$SLURM_ARRAY_TASK_ID" ;;
    compare) /usr/bin/time -v "${RUN[@]}" compare --work "$WORK" ;;
    *) echo "unknown STEP $STEP"; exit 1 ;;
esac
