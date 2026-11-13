#!/bin/bash
#SBATCH --job-name=nb300-retry-index
#SBATCH --partition=hns,bigmem
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.log
set -euo pipefail
: "${OUT:?}" "${KMERSEEK_SHA:?}" "${ALPHABET:?}" "${KSIZE:?}" "${PREVIOUS_JOB_ID:?}" "${INDEX_SCRIPT_DIR:?}"
case "$ALPHABET" in
  protein20|dayhoff6|funcgroups8|hsdm17|mmseqs12|sdm12|wass14) ;;
  *) echo "Refusing to restart protected or unknown setting: $ALPHABET" >&2; exit 1 ;;
esac
BASE=$OUT/step4/full/${ALPHABET}.k${KSIZE}.s1.C2.X8.masknone
test ! -e "$BASE.complete"
test -d "$BASE.rocksdb"
ARCHIVE=$OUT/step4/failed_attempts/$PREVIOUS_JOB_ID
mkdir -p "$OUT/step4/failed_attempts"
mkdir "$ARCHIVE"
# Move, never delete, the old index and logs. Completed settings cannot reach this code.
shopt -s nullglob
files=("$BASE".*)
test "${#files[@]}" -gt 0
for f in "${files[@]}"; do
  mv -- "$f" "$ARCHIVE/"
done
exec bash "$INDEX_SCRIPT_DIR/full_index.sbatch"
