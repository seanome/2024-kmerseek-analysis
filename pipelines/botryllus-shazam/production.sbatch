#!/bin/bash
#SBATCH --job-name=botryllus-64-nextflow
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --qos=long
set -euo pipefail
CODE=${1:?immutable versioned source}
LAUNCH=${2:?prepared launch directory}
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
    PREFLIGHT_JOB=${PREFLIGHT##*-}
    sacct --noheader --parsable2 -j "$PREFLIGHT_JOB" --format=JobID,State,Elapsed,MaxRSS,ExitCode > "$LAUNCH/preflight-sacct.tsv"
    "$PY" "$CODE/scripts/botryllus_shazam/preflight_resource_trace.py" "$PREFLIGHT" "$LAUNCH/preflight-trace.tsv" "$LAUNCH/preflight-sacct.tsv"
    bash "$CODE/pipelines/botryllus-shazam/prepare-launch.sh" "$LAUNCH" --libraries "$SCRATCH/botryllus-shazam-PEC-20261023/trimmed-inputs.validated.json" --sizing-library "$PREFLIGHT/largest-library.json" "$LAUNCH/preflight-trace.tsv"
fi
cd "$LAUNCH"
test -s params.json
test -s measured/resources.config
"$SCRATCH/oyster-shazam-20261023/software/nextflow-25.10.4" run "$CODE/pipelines/botryllus-shazam" -profile sherlock -c "$LAUNCH/measured/resources.config" -params-file "$LAUNCH/params.json" -work-dir "$LAUNCH/work" -ansi-log false -resume
