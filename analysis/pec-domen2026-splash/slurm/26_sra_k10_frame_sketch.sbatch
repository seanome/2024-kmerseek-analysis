#!/bin/bash
#SBATCH --job-name=nb361-k10-sketch
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x.%A_%a.out
# One run of runs.tsv (array index = row, 0-based, header excluded; or RUNS_FILE lists runs).
# Replaces the Swiss-Prot 7-mer frame choice of 22_sra_sketch_run.sbatch (Olga, 2026-11-03).
# 1. pec-tools best-frame, rule orpheum, protein k = 10: a low-complexity read gets no frame; the
#    allowed frames have no stop codon and are not low complexity; the allowed frame with the most
#    protein 10-mers in the reference wins. Reference: the species' own proteins, the new
#    Botryllus proteome for PRJNA1219309 and GENCODE human translations for the human runs.
#    Cutoff 0: a read is kept when its chosen frame has at least one matching 10-mer (best-frame
#    drops reads with no match before applying the cutoff). No share cutoff.
# 2. Windows per kept read at HP k = 24, 27, 30 (residues - k + 1, or 0), from the kept peptides.
# 3. branchwater manysketch: one signature per run at HP k = 24, 27, 30, scaled 10.
# Writes to $PEC_ROOT/sra_search/k10_frames/<run>/; the earlier runs/ folder is not touched.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
W=$PEC_ROOT/sra_search
BOT_REF=${BOT_REF:-$PEC_ROOT/db/new_genome/protein.fasta}
HUMAN_REF=${HUMAN_REF:-$PEC_ROOT/db/gencode_v50/gencode.v50.pc_translations.fa}
cd "$W"
if [ -n "${RUNS_FILE:-}" ]; then
  run=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
else
  run=$(awk -F'\t' 'NR>1{print $1}' runs.tsv | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
fi
dataset=$(awk -F'\t' -v r="$run" '$1==r{print $3}' runs.tsv)
files=$(awk -F'\t' -v r="$run" '$1==r{print $9}' runs.tsv | tr ';' ' ')
[ -n "$files" ] || { echo "run $run not in runs.tsv"; exit 1; }
for f in $files; do
  [ -s "$f" ] || { echo "missing reads $f for $run"; exit 1; }
  case $f in "$W"/reads/*) [ -e "$f.done" ] || { echo "download of $f not verified"; exit 1; };; esac
done
if [ "$dataset" = PRJNA1219309 ]; then ref=$BOT_REF; else ref=$HUMAN_REF; fi
O=$W/k10_frames/$run
mkdir -p "$O"
[ -e "$O/sketch.done" ] && { echo "$run done"; exit 0; }
echo "$ref" > "$O/reference.txt"
L=${L_SCRATCH:-/tmp}/nb361_k10_$run
mkdir -p "$L"
trap 'command rm -rf "$L"' EXIT
PT=$PEC_CODE/pec-tools/target/release/pec-tools

/usr/bin/time -v -o "$O/best_frame.time" bash -c "set -o pipefail; zcat $files | $PT best-frame $ref protein 10 $L/$run 0 orpheum 6" \
  2> "$O/best_frame.log"
command cp "$L/$run.hist.tsv" "$L/$run.summary.tsv" "$O/"
faa=$L/$run.ge0.faa
wc -c < "$faa" > "$O/ge0.faa_bytes"
# windows per kept peptide at each HP k (sourmash hp sketches every residue window of length k)
awk '!/^>/{L=length($0); n++; res+=L; for(i=1;i<=3;i++){k=(i==1?24:(i==2?27:30)); if(L>=k){w[i]+=L-k+1; r[i]++}}}
  END{print "peptides\tresidues\twindows_k24\twindows_k27\twindows_k30\tpeptides_with_window_k24\tpeptides_with_window_k27\tpeptides_with_window_k30";
      printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", n, res, w[1], w[2], w[3], r[1], r[2], r[3]}' "$faa" > "$O/windows.tsv"
printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$run" "$faa" > "$L/ms.csv"
/usr/bin/time -v -o "$O/manysketch.time" sourmash scripts manysketch "$L/ms.csv" -c 1 \
  -p hp,k=24,scaled=10 -p hp,k=27,scaled=10 -p hp,k=30,scaled=10 \
  -o "$O/$run.ge0.sig.zip" > "$O/manysketch.log" 2>&1
touch "$O/sketch.done"
