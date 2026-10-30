#!/bin/bash
#SBATCH --job-name=nb361-dekupl
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --output=%x.%j.out
# dekupl-run 1.3.5 on one contrast, same trimmed pairs and samples as SPLASH:
#   $PEC_ROOT/contrasts/$CONTRAST/{design.tsv,input.txt}
# Paired files are found next to each interleaved file listed in input.txt (<sample>_1/_2.fastq.gz).
# Resources are set at submission.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}" "${CONTRAST:?}"
# limma is the primary k-mer test (fast); DESeq2 1.18.1 in this image is the slow alternative
METHOD=${DEKUPL_METHOD:-limma}
D=$PEC_ROOT/contrasts/$CONTRAST
W=dekupl_$METHOD
mkdir -p "$D/$W/fastq"
cd "$D/$W"
[ -e ../$W.done ] && { echo "$W already done"; exit 0; }
while read -r s p; do
  dir=$(dirname "$p")
  for m in 1 2; do
    [ -s "$dir/${s}_$m.fastq.gz" ] || { echo "missing $dir/${s}_$m.fastq.gz"; exit 2; }
    ln -sfn "$dir/${s}_$m.fastq.gz" "fastq/${s}_$m.fastq.gz"
  done
done < ../input.txt
TMP=${L_SCRATCH:-$D/$W}/dekupl_tmp_$SLURM_JOB_ID
mkdir -p "$TMP"
"$PEC_ROOT/env/bin/python" "$PEC_CODE/scripts/dekupl_config.py" ../design.tsv "$D/$W/fastq" "$TMP" config.json "$METHOD"
IMG=$PEC_ROOT/images/dekupl-run_1.3.5.sif
# The image's limma script loads the R package pryr (never used) but the image lacks it.
# pryr 0.1.4 (CRAN archive, 2018) is installed beside the image and put on R's library path;
# DE-kupl's own code is not changed.
export APPTAINERENV_R_LIBS=$PEC_ROOT/Rlib_dekupl
[ -d "$APPTAINERENV_R_LIBS/pryr" ] || { echo "pryr side library missing"; exit 2; }
# SLURM_MEM_PER_NODE is unset with --mem-per-cpu or a partition default; fall back to per-cpu memory
MEM_MB_TOTAL=${SLURM_MEM_PER_NODE:-$(( ${SLURM_MEM_PER_CPU:-4096} * ${SLURM_CPUS_PER_TASK:-1} ))}
MEM_MB=$(( MEM_MB_TOTAL * 9 / 10 ))
echo "memory used for limits: SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} -> ${MEM_GB:-${MEM_MB:-}}"
/usr/bin/time -v -o ../$W.time apptainer run --bind /scratch --bind "$TMP" "$IMG" \
  --configfile config.json -j "$SLURM_CPUS_PER_TASK" --resources ram=$MEM_MB -p > ../$W.log 2>&1
rm -rf "$TMP"
M=DEkupl_result/control_vs_PEC_kmer_counts/merged-diff-counts.tsv.gz
[ -s "$M" ] || { echo "missing $M"; exit 3; }
zcat "$M" | awk 'END{print NR-1}' > ../$W.n_contigs
zcat DEkupl_result/control_vs_PEC_kmer_counts/diff-counts.tsv.gz | awk 'END{print NR-1}' > ../$W.n_de_kmers
du -sb DEkupl_result > ../$W.disk_bytes
echo ok > ../$W.done
