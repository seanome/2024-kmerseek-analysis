#!/bin/bash
#SBATCH --job-name=oyster-replacement-provenance
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
CODE=$1
MANIFEST=$2
ARCHIVE=$3
OUTPUT=$4
mkdir "$OUTPUT"
ROOT="$L_SCRATCH_JOB/provenance-audit"
mkdir "$ROOT"
finish() {
    result=$?
    trap - EXIT
    tar -C "$ROOT" -czf "$OUTPUT/evidence.tar.gz.partial" .
    command mv "$OUTPUT/evidence.tar.gz.partial" "$OUTPUT/evidence.tar.gz"
    if test -f "$ROOT/results/verified-provenance.json"; then
        command cp "$ROOT/results/verified-provenance.json" "$OUTPUT/verified-provenance.json"
    fi
    printf '%s\n' "$result" > "$OUTPUT/exit-code.txt"
    exit "$result"
}
trap finish EXIT
command cp -R "$CODE" "$ROOT/code"
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
SM="$GROUP_HOME/opt/oyster-shazam-20261023/bin/sourmash"
"$PY" "$ROOT/code/nextflow/verify_replacements.py" "$MANIFEST" "$ARCHIVE" "$ROOT/results" "$SM"
