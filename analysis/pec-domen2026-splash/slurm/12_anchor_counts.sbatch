#!/bin/bash
#SBATCH --job-name=nb361-anchorcounts
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --output=%x.%j.out
# Anchor counts per sample for every tested anchor, from the SATC bins, then PyDESeq2.
set -euo pipefail
: "${PEC_ROOT:?}" "${CONTRAST:?}"
D=$PEC_ROOT/contrasts/$CONTRAST
cd "$D"
export PATH=$PEC_ROOT/env/bin:$PATH
PT=$PEC_ROOT/pec-tools/target/release/pec-tools
[ -e splash.done ] || { echo "splash not done"; exit 2; }
if [ ! -e anchor_counts.done ]; then
  ALL=splash_out/result.after_correction.all_anchors.tsv
  MAP=splash_out/sample_name_to_id.mapping.txt
  [ -s "$MAP" ] || MAP=$(ls splash_out/*mapping*.txt | head -1)
  rm -f anchor_counts.tsv.part
  start=$(date +%s)
  for b in splash_out/result_satc/bin*.satc; do
    "$PEC_ROOT/splash/bin/satc_dump" --sample_names "$MAP" "$b" /dev/stdout \
      | "$PT" anchor-counts samples.txt "$ALL" >> anchor_counts.tsv.part 2>> anchor_counts.log
  done
  echo "seconds $(( $(date +%s) - start ))" >> anchor_counts.log
  N=$(wc -l < anchor_counts.tsv.part); W=$(( $(wc -l < "$ALL") - 1 ))
  echo "anchors with counts $N, tested anchors $W" >> anchor_counts.log
  [ "$N" = "$W" ] || { echo "anchor count rows $N != tested anchors $W"; exit 3; }
  mv anchor_counts.tsv.part anchor_counts.tsv
  touch anchor_counts.done
fi
if [ ! -e deseq.done ]; then
  /usr/bin/time -v -o deseq.time python "$PEC_CODE/scripts/anchor_deseq.py" anchor_counts.tsv samples.txt \
    design.tsv deseq "$SLURM_CPUS_PER_TASK" > deseq.log 2>&1
  touch deseq.done
fi
