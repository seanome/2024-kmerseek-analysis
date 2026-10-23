#!/bin/bash
#SBATCH --job-name=nb361-control-sim
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
# Known-answer control: simulate 6 control + 6 PEC samples (1 M pairs each, seed 11) from the
# planted design, trim them with the same fastp settings as the real reads, and lay out the
# contrast folder contrasts/control/ exactly like a real contrast (paired files for DE-kupl,
# interleaved copy for SPLASH).
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
PT=$PEC_ROOT/pec-tools/target/release/pec-tools
C=$PEC_ROOT/control
mkdir -p "$C/reads" "$C/trimmed"
cd "$C"
if [ ! -e simulate.done ]; then
  /usr/bin/time -v -o simulate.time "$PT" simulate "$PEC_CODE/control/control_transcripts.fa" \
    "$PEC_CODE/control/design.tsv" reads 1000000 11 2> simulate.log
  pigz -p 8 reads/*.fastq
  touch simulate.done
fi
SAMPLES="ctrl1 ctrl2 ctrl3 ctrl4 ctrl5 ctrl6 pec1 pec2 pec3 pec4 pec5 pec6"
for s in $SAMPLES; do
  [ -e "trimmed/$s.done" ] && continue
  P1=trimmed/${s}_1.fastq.gz; P2=trimmed/${s}_2.fastq.gz
  # fastp chooses gzip from the extension, so temporary names must end in .fastq.gz
  T1=trimmed/${s}_1.tmp.fastq.gz; T2=trimmed/${s}_2.tmp.fastq.gz
  fastp -i "reads/${s}_1.fastq.gz" -I "reads/${s}_2.fastq.gz" -o "$T1" -O "$T2" --compression 4 \
    --detect_adapter_for_pe --trim_poly_g --length_required 50 --thread 8 \
    -j "trimmed/$s.fastp.json" -h "trimmed/$s.fastp.html" 2> "trimmed/$s.fastp.log"
  WANT=$(python -c "import json;print(json.load(open('trimmed/$s.fastp.json'))['summary']['after_filtering']['total_reads'])")
  N1=$(pigz -dc "$T1" | awk 'END{print NR/4}'); N2=$(pigz -dc "$T2" | awk 'END{print NR/4}')
  [ "$N1" = "$N2" ] && [ $(( N1 + N2 )) = "$WANT" ] || { echo "$s: R1 $N1 R2 $N2 fastp $WANT"; exit 3; }
  mv "$T1" "$P1"; mv "$T2" "$P2"
  paste <(pigz -dc "$P1" | paste - - - -) <(pigz -dc "$P2" | paste - - - -) | tr '\t' '\n' \
    | pigz -p 8 -1 > "trimmed/$s.il.fastq.gz.part"
  N=$(pigz -dc "trimmed/$s.il.fastq.gz.part" | awk 'END{print NR/4}')
  [ "$N" = "$WANT" ] || { echo "$s: interleaved $N != fastp $WANT"; exit 3; }
  mv "trimmed/$s.il.fastq.gz.part" "trimmed/$s.il.fastq.gz"
  echo "$N" > "trimmed/$s.done"
done
D=$PEC_ROOT/contrasts/control
mkdir -p "$D"
: > "$D/input.txt"; : > "$D/cjs.csv"; : > "$D/samples.txt"
printf 'sample\tcondition\tpair\tgenotype\n' > "$D/design.tsv"
for s in $SAMPLES; do
  echo "$s $C/trimmed/$s.il.fastq.gz" >> "$D/input.txt"
  case $s in ctrl*) w=-1; cond=control;; *) w=1; cond=PEC;; esac
  echo "$s,$w" >> "$D/cjs.csv"
  echo "$s" >> "$D/samples.txt"
  printf '%s\t%s\tp%s\tsim\n' "$s" "$cond" "${s: -1}" >> "$D/design.tsv"
done
echo "control contrast ready: $(wc -l < "$D/input.txt") samples"
