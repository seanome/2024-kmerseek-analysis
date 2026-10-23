#!/bin/bash
#SBATCH --job-name=301-score-recovery
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=301-score-recovery-%j.out
# Submit once with afterany on the 16 recovery jobs. Failed settings remain incomplete.
# Completed filtered output is small: 54 original chunks total about 2 MB compressed.
# 4 GB and 30 minutes cover reading 19 settings and writing per-feature scores.
set -euo pipefail
: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
module load python/3.12.1
BASE=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}
OUT=$BASE/scored-recovery-$SLURM_JOB_ID
mkdir -p "$OUT"
python3 -I "$SLURM_SUBMIT_DIR/scripts/score_301_completed.py" --base "$BASE" --out "$OUT"
python3 -I - "$BASE" "$OUT" <<'PY'
import csv
import sys
from pathlib import Path
base,out=map(Path,sys.argv[1:])
with open(out/'301_settings_coverage_swissprot.tsv','w') as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n')
    w.writerow(['alphabet','state'])
    for d in sorted((base/'step5').iterdir()):
        if not d.is_dir(): continue
        status=d/'status.tsv'
        rows=list(csv.DictReader(status.open(),delimiter='\t')) if status.exists() else []
        state='completed' if any(r['fit']=='fitted' for r in rows) else 'incomplete_or_no_fit'
        w.writerow([d.name,state])
print(out)
PY
