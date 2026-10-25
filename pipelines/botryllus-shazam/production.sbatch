#!/bin/bash
#SBATCH --job-name=botryllus-64-nextflow
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
set -euo pipefail
CODE=${1:?immutable versioned source}
LAUNCH=${2:?prepared launch directory}
ml devel java/21.0.4
export NXF_OPTS='-Xms1g -Xmx10g'
cd "$LAUNCH"
test -s params.json
test -s measured/resources.config
"$SCRATCH/oyster-shazam-20261023/software/nextflow-25.10.4" run "$CODE/pipelines/botryllus-shazam" -profile sherlock -c "$LAUNCH/measured/resources.config" -params-file "$LAUNCH/params.json" -work-dir "$LAUNCH/work" -ansi-log false -resume
