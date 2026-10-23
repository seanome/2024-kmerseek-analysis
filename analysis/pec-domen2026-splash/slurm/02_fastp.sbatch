#!/bin/bash
#SBATCH --job-name=nb361-fastp
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%A_%a.out
# fastp on one run (line N+1 of meta/runs.txt); paired output named by sample (for DE-kupl)
# plus an interleaved copy (SPLASH sees one file per sample). Writes trimmed/<sample>.done after the record count
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
P1=trimmed/${SAMPLE}_1.fastq.gz; P2=trimmed/${SAMPLE}_2.fastq.gz
# fastp chooses gzip from the extension, so temporary names must end in .fastq.gz
T1=trimmed/${SAMPLE}_1.tmp.fastq.gz; T2=trimmed/${SAMPLE}_2.tmp.fastq.gz
rm -f "$OUT.part" "$T1" "$T2"
# paired output for DE-kupl (it reads <sample>_1/_2.fastq.gz)
/usr/bin/time -v -o "logs/fastp_$SAMPLE.time" fastp -i reads/${RUN}_1.fastq.gz -I reads/${RUN}_2.fastq.gz \
  -o "$T1" -O "$T2" --compression 4 \
  --detect_adapter_for_pe --trim_poly_g --length_required 50 --thread 8 \
  -j trimmed/$SAMPLE.fastp.json -h trimmed/$SAMPLE.fastp.html 2> trimmed/$SAMPLE.fastp.log
WANT=$(python -c "import json;print(json.load(open('trimmed/$SAMPLE.fastp.json'))['summary']['after_filtering']['total_reads'])")
N1=$(pigz -dc "$T1" | awk 'END{print NR/4}'); N2=$(pigz -dc "$T2" | awk 'END{print NR/4}')
[ "$N1" = "$N2" ] && [ $(( N1 + N2 )) = "$WANT" ] || { echo "R1 $N1 R2 $N2 fastp $WANT"; exit 3; }
mv "$T1" "$P1"; mv "$T2" "$P2"
# interleaved copy for SPLASH, which wants one file per sample name: R1 record then its R2 record
paste <(pigz -dc "$P1" | paste - - - -) <(pigz -dc "$P2" | paste - - - -) | tr '\t' '\n' \
  | pigz -p 8 -1 > "$OUT.part"
N=$(pigz -dc "$OUT.part" | awk 'END{print NR/4}')
[ "$N" = "$WANT" ] || { echo "interleaved records $N != fastp after_filtering $WANT"; exit 3; }
mv "$OUT.part" "$OUT"
echo "$N" > "trimmed/$SAMPLE.done"
