#!/bin/bash
#SBATCH --job-name=botryllus-64-nextflow
#SBATCH --partition=hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --qos=normal
set -euo pipefail
CODE=${1:?immutable versioned source}
LAUNCH=${2:?prepared launch directory}
# Prevent two controllers from writing the same launch, including queued duplicates.
exec 9>"${LAUNCH}.controller.lock"
flock -n 9 || { echo "Another controller owns $LAUNCH" >&2; exit 1; }
ml devel java/21.0.4
export NXF_OPTS='-Xms1g -Xmx10g'
if test "$#" -eq 5; then
    PREFLIGHT=$3
    REFERENCE_CHECK=$4
    FIXTURE=$5
    test -s "$FIXTURE/fixture-verified.json"
    test -s "$PREFLIGHT/complete.json"
    test -s "$REFERENCE_CHECK/verified.json"
    PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
    "$PY" "$CODE/scripts/botryllus_shazam/prepare_production.py" "$CODE" "$PREFLIGHT" "$REFERENCE_CHECK" "$LAUNCH"
    printf '%s\n' "$PREFLIGHT" > "$LAUNCH/preflight-source.txt"
    PREFLIGHT_JOB=${PREFLIGHT##*-}
    sacct --noheader --parsable2 -j "$PREFLIGHT_JOB" --format=JobID,State,Elapsed,MaxRSS,ExitCode > "$LAUNCH/preflight-sacct.tsv"
    "$PY" "$CODE/scripts/botryllus_shazam/preflight_resource_trace.py" "$PREFLIGHT" "$LAUNCH/preflight-trace.tsv" "$LAUNCH/preflight-sacct.tsv"
else
    test "$#" -eq 2
    PREFLIGHT=$(cat "$LAUNCH/preflight-source.txt")
fi
# Keep every resource prediction and include this launch's completed task history.
export RESOURCE_DIR="$LAUNCH/measured-${SLURM_JOB_ID}"
shopt -s nullglob
TRACES=("$LAUNCH/preflight-trace.tsv" "$LAUNCH"/results/diagnostics/trace-*.tsv)
bash "$CODE/nextflow-runs/botryllus-shazam/prepare-launch.sh" "$LAUNCH" --libraries "$SCRATCH/botryllus-shazam-PEC-20261023/trimmed-inputs.validated.json" --sizing-library "$PREFLIGHT/largest-library.json" "${TRACES[@]}"

cd "$LAUNCH"
test -s params.json
test -s "$RESOURCE_DIR/resources.config"
"$SCRATCH/oyster-shazam-20261023/software/nextflow-25.10.4" run "$CODE/nextflow-runs/botryllus-shazam" -profile sherlock -c "$RESOURCE_DIR/resources.config" -params-file "$LAUNCH/params.json" -work-dir "$LAUNCH/work" -ansi-log false -resume
