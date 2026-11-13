#!/bin/bash
#SBATCH --job-name=oyster-public-zero-validate
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
ml biology trim_galore/0.6.10
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
"$PY" - <<'PY'
import pathlib,os
p=pathlib.Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023/runs/DRR350/DRR350416'
if p.exists() and not (p/'validated.json').exists():
 saved=p.with_name('DRR350416.interrupted-49222146')
 if saved.exists():raise ValueError('Interrupted backup exists; inspect before retry')
 p.rename(saved)
 print('Preserved interrupted attempt',saved,flush=True)
PY
"$PY" "$HOME/oyster-shazam-code/archive_compare.py" prepare
"$PY" "$HOME/oyster-shazam-code/archive_compare.py" DRR350416
"$PY" "$HOME/oyster-shazam-code/verify_archive_run.py"
