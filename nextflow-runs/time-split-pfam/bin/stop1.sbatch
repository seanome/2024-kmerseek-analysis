#!/bin/bash
#SBATCH --job-name=280-stop1
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.log
# Stop 1 of the time-split test (notebook 280): build the query set and count every step,
# then the identity of 500 random instances to their closest t0 Swiss-Prot domain.
# Submit with `make stop1` (runs in RUN_DIR, reads the scripts from this checkout).
set -euo pipefail
RUN=$PWD  # sbatch --chdir=RUN_DIR (make stop1)
BIN=${TIME_SPLIT_BIN:-$RUN/bin}
DATA=$RUN/data
QS=$RUN/query_set
ID=$RUN/stop1_identity
IMG_DIR=$RUN/images
mkdir -p "$IMG_DIR" "$ID/m8"
# One frozen kmerseek build for the whole test: 0.4.0-rc5, pinned by digest.
KMERSEEK_IMG=$IMG_DIR/kmerseek-0.4.0-rc5.sif
FASTA_IMG=$IMG_DIR/fasta3-36.3.8i.sif
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache
[ -s "$KMERSEEK_IMG" ] || apptainer pull "$KMERSEEK_IMG" \
    docker://docker.io/olgabot/kmerseek@sha256:7855826572644933e0c71926102b90174e650c144ae90b3641b760bd9f463afb
[ -s "$FASTA_IMG" ] || apptainer pull "$FASTA_IMG" \
    docker://quay.io/biocontainers/fasta3:36.3.8i--h7b50bb2_3
PY="apptainer exec --bind /scratch $KMERSEEK_IMG python3"

$PY "$BIN/build_query_set.py" --data "$DATA" --out "$QS"
$PY "$BIN/stop1_identity.py" prepare --qs "$QS" --work "$ID"
# BLOSUM62, gap open 11, extend 1 (BLAST's defaults), report every alignment (-E 1e6),
# best 5 library sequences per query; collect keeps the top bit score.
while IFS=$'\t' read -r i fam; do
    [ -s "$ID/m8/$i.m8" ] && continue
    apptainer exec --bind /scratch "$FASTA_IMG" ssearch36 -q -m 8CB -s BL62 -f -11 -g -1 \
        -E 1000000 -b 5 -d 0 -T "$SLURM_CPUS_PER_TASK" \
        "$ID/q/$i.fa" "$ID/lib/$fam.fa" > "$ID/m8/$i.m8.tmp"
    mv "$ID/m8/$i.m8.tmp" "$ID/m8/$i.m8"
done < "$ID/jobs.tsv"
$PY "$BIN/stop1_identity.py" collect --qs "$QS" --work "$ID"
