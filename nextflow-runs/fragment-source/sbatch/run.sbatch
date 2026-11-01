#!/bin/bash
#SBATCH --job-name=nb354-run
#SBATCH --partition=hns,normal
#SBATCH --output=%x-%j.log
# One stage of notebook 354: STAGE=freeze|dev|test|summarise. Resources per stage are set by
# the Makefile on the sbatch line (see its sizing notes).
set -euo pipefail
: "${OUT:?set OUT}" "${STAGE:?set STAGE}" "${BIN:?set BIN}" "${KMERSEEK:?set KMERSEEK}"
# gcc: libstdc++ for kmerseek; python 3.12 for the venv; BLAST+ 2.16.0.
module load gcc/14.2.0 python/3.12.1 biology ncbi-blast+/2.16.0
PY=$HOME/venvs/nb354/bin/python
$PY -c "import parasail, polars, numpy"
blastp -version | head -1
"$KMERSEEK" --version
WORKERS=${SLURM_CPUS_PER_TASK:-1}
case "$STAGE" in
  freeze)    $PY "$BIN/freeze.py" --raw "$OUT/raw" --out "$OUT/frozen" ;;
  dev)       $PY "$BIN/nb354.py" dev --frozen "$OUT/frozen" --work "$OUT/work" --kmerseek "$KMERSEEK" --workers "$WORKERS"
             $PY "$BIN/nb354.py" choose --frozen "$OUT/frozen" --work "$OUT/work" ;;
  test)      $PY "$BIN/nb354.py" test --frozen "$OUT/frozen" --work "$OUT/work" --kmerseek "$KMERSEEK" --workers "$WORKERS" ;;
  summarise) $PY "$BIN/nb354.py" summarise --frozen "$OUT/frozen" --work "$OUT/work" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
echo "nb354 $STAGE done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed,TotalCPU"
