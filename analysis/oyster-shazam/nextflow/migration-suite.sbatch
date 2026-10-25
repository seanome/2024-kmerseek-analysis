#!/bin/bash
#SBATCH --job-name=oyster-migration-audit
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
CODE=$1
OUTPUT=$2
mkdir "$OUTPUT"
TEST_ROOT="$L_SCRATCH_JOB/oyster-migration"
mkdir "$TEST_ROOT"
# Preserve work/cache, logs and partial files even when a gate fails.
finish() {
    result=$?
    trap - EXIT
    mkdir "$OUTPUT/summaries"
    "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" - "$TEST_ROOT" "$OUTPUT/summaries" <<'PYEND'
import sys,shutil
from pathlib import Path
root,out=map(Path,sys.argv[1:])
for pattern in ('protocol/protocol-audit.json','audit/*audit.json','corrected-sketches/validated.json','migration/migration-audit.json','migration/*.trace.tsv','migration/*.log','*tests.log','sherlock-resolved.config'):
    for p in root.glob(pattern):
        shutil.copyfile(p,out/str(p.relative_to(root)).replace('/','--'))
PYEND
    tar -C "$TEST_ROOT" -czf "$OUTPUT/evidence.tar.gz.partial" .
    command mv "$OUTPUT/evidence.tar.gz.partial" "$OUTPUT/evidence.tar.gz"
    printf '%s\n' "$result" > "$OUTPUT/exit-code.txt"
    exit "$result"
}
trap finish EXIT
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
SM="$GROUP_HOME/opt/oyster-shazam-20261023/bin/sourmash"
ml biology nextflow/25.04.7
NF=/share/software/user/open/nextflow/25.04.7/bin/nextflow
export NXF_OPTS='-Xms256m -Xmx2g'
command cp -R "$CODE" "$TEST_ROOT/code"
CODE="$TEST_ROOT/code"
"$PY" -m unittest discover -s "$CODE" -p test_peptide_protocol.py -v > "$TEST_ROOT/translation-tests.log" 2>&1
"$PY" -m unittest discover -s "$CODE/nextflow" -p test_artifacts.py -v > "$TEST_ROOT/artifact-tests.log" 2>&1
"$NF" config "$CODE/nextflow" -profile sherlock > "$TEST_ROOT/sherlock-resolved.config"
"$PY" "$CODE/nextflow/verify_protocol.py" "$TEST_ROOT/protocol" "$SM"
"$PY" "$CODE/nextflow/audit_completed.py" "$TEST_ROOT/audit"
"$PY" "$CODE/peptide_protocol.py" "$TEST_ROOT/audit/corrected-pilot.fa" "$TEST_ROOT/corrected-translation"
"$PY" "$CODE/sketch_corrected_peptides.py" "$TEST_ROOT/corrected-translation" "$TEST_ROOT/corrected-sketches" --sourmash "$SM" --threads 4
"$PY" "$CODE/nextflow/migration_tests.py" "$TEST_ROOT/migration" "$TEST_ROOT/audit" "$NF" "$PY" "$SM"
