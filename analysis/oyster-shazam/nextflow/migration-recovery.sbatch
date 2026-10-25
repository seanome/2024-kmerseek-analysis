#!/bin/bash
#SBATCH --job-name=oyster-migration-audit-recovery
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
CODE=$1
PRIOR=$2
OUTPUT=$3
mkdir "$OUTPUT"
TEST_ROOT="$L_SCRATCH_JOB/oyster-migration-recovery"
mkdir "$TEST_ROOT"
cd "$TEST_ROOT"
finish() {
    result=$?
    trap - EXIT
    mkdir "$OUTPUT/summaries"
    "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" - "$TEST_ROOT" "$OUTPUT/summaries" <<'PYEND'
import sys,shutil
from pathlib import Path
root,out=map(Path,sys.argv[1:])
for pattern in ('historical-word-audit.json','public-pilot/corrected-public-pilot.json','migration/migration-audit.json','migration/*.trace.tsv','migration/*.log','recovery-tests.log'):
    for p in root.glob(pattern):
        shutil.copyfile(p,out/str(p.relative_to(root)).replace('/','--'))
PYEND
    tar -C "$TEST_ROOT" -czf "$OUTPUT/evidence.tar.gz.partial" .
    command mv "$OUTPUT/evidence.tar.gz.partial" "$OUTPUT/evidence.tar.gz"
    printf '%s\n' "$result" > "$OUTPUT/exit-code.txt"
    exit "$result"
}
trap finish EXIT
mkdir imported
tar -xzf "$PRIOR/evidence.tar.gz" -C imported audit corrected-sketches
command cp -R "$CODE" code
CODE="$TEST_ROOT/code"
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
SM="$GROUP_HOME/opt/oyster-shazam-20261023/bin/sourmash"
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
ml biology nextflow/25.04.7
NF=/share/software/user/open/nextflow/25.04.7/bin/nextflow
export NXF_OPTS='-Xms256m -Xmx2g'
"$PY" -m unittest discover -s "$CODE/nextflow" -p test_artifacts.py -v > recovery-tests.log 2>&1
"$PY" "$CODE/nextflow/audit_historical_words.py" "$TEST_ROOT/historical-word-audit.json"
"$PY" "$CODE/nextflow/corrected_public_pilot.py" "$TEST_ROOT/public-pilot" "$TEST_ROOT/imported/corrected-sketches" "$SM"
"$PY" "$CODE/nextflow/migration_tests.py" "$TEST_ROOT/migration" "$TEST_ROOT/imported/audit" "$NF" "$PY" "$SM"
