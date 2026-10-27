#!/bin/bash
# Prepare only. Production launch requires a reviewed resource table and passing fixture.
set -euo pipefail
PIPELINE=$(cd "$(dirname "$0")" && pwd)
LAUNCH=${1:?new launch directory}
shift
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
"$PY" "$PIPELINE/../../scripts/botryllus_shazam/measure_nextflow_resources.py" --output "${RESOURCE_DIR:-$LAUNCH/measured}" "$@"
"$SCRATCH/oyster-shazam-20261023/software/nextflow-25.10.4" -c "${RESOURCE_DIR:-$LAUNCH/measured}/resources.config" config "$PIPELINE" -profile sherlock -flat > "${RESOURCE_DIR:-$LAUNCH/measured}/resolved.config"
