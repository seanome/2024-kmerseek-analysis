#!/bin/bash
#SBATCH --job-name=nb361-fastp
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%A_%a.out
# fastp on one run (line N+1 of meta/runs.txt); interleaved output named by sample, so
# SPLASH sees one file per sample. Writes trimmed/<sample>.done after the record count
# in the output equals fastp's own count of reads passing filters.
set -euo pipefail
: "${PEC_ROOT:?}"
cd "$PEC_ROOT"
export PATH=$PEC_ROOT/env/bin:$PATH
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" meta/runs.txt)
SAMPLE=$(awk -F'\t' -v r="$RUN" '$2==r{print $1}' design/samples.tsv)
[ -n "$SAMPLE" ] || { echo "no sample for $RUN"; exit 1; }
mkdir -p trimmed logs
OUT=trimmed/$SAMPLE.il.fastq.gz
if [ -e "trimmed/$SAMPLE.done" ]; then
  if [ -s "$OUT" ] && [ -s "trimmed/$SAMPLE.fastp.json" ]; then
    N=$(pigz -dc "$OUT" | awk 'END{print NR/4}')
    WANT=$(python -c "import json;print(json.load(open('trimmed/$SAMPLE.fastp.json'))['summary']['after_filtering']['total_reads'])")
    [ "$N" = "$WANT" ] && { echo "$SAMPLE existing output reverified"; exit 0; }
  fi
  rm -f "trimmed/$SAMPLE.done"
fi
[ -e "reads/$RUN.done" ] || { echo "reads for $RUN not verified"; exit 2; }
rm -f "$OUT.part"
/usr/bin/time -v -o "logs/fastp_$SAMPLE.time" bash -c "set -o pipefail; fastp -i reads/${RUN}_1.fastq.gz -I reads/${RUN}_2.fastq.gz --stdout \
  --detect_adapter_for_pe --trim_poly_g --length_required 50 --thread 6 \
  -j trimmed/$SAMPLE.fastp.json -h trimmed/$SAMPLE.fastp.html 2> trimmed/$SAMPLE.fastp.log \
  | pigz -p 2 -1 > $OUT.part"
N=$(pigz -dc "$OUT.part" | awk 'END{print NR/4}')
WANT=$(python -c "import json;print(json.load(open('trimmed/$SAMPLE.fastp.json'))['summary']['after_filtering']['total_reads'])")
[ "$N" = "$WANT" ] || { echo "records $N != fastp after_filtering $WANT"; exit 3; }
mv "$OUT.part" "$OUT"
echo "$N" > "trimmed/$SAMPLE.done"
