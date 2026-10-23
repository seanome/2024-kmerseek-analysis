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
D=$PEC_ROOT/contrasts/$CONTRAST
mkdir -p "$D/dekupl/fastq"
cd "$D/dekupl"
[ -e ../dekupl.done ] && { echo "dekupl already done"; exit 0; }
while read -r s p; do
  dir=$(dirname "$p")
  for m in 1 2; do
    [ -s "$dir/${s}_$m.fastq.gz" ] || { echo "missing $dir/${s}_$m.fastq.gz"; exit 2; }
    ln -sfn "$dir/${s}_$m.fastq.gz" "fastq/${s}_$m.fastq.gz"
  done
done < ../input.txt
TMP=${L_SCRATCH:-$D/dekupl}/dekupl_tmp_$SLURM_JOB_ID
mkdir -p "$TMP"
"$PEC_ROOT/env/bin/python" "$PEC_CODE/scripts/dekupl_config.py" ../design.tsv "$D/dekupl/fastq" "$TMP" config.json
IMG=$PEC_ROOT/images/dekupl-run_1.3.5.sif
MEM_MB=$(( SLURM_MEM_PER_NODE * 9 / 10 ))
/usr/bin/time -v -o ../dekupl.time apptainer run --bind /scratch --bind "$TMP" "$IMG" \
  --configfile config.json -j "$SLURM_CPUS_PER_TASK" --resources ram=$MEM_MB -p > ../dekupl.log 2>&1
rm -rf "$TMP"
M=DEkupl_result/control_vs_PEC_kmer_counts/merged-diff-counts.tsv.gz
[ -s "$M" ] || { echo "missing $M"; exit 3; }
zcat "$M" | awk 'END{print NR-1}' > ../dekupl.n_contigs
zcat DEkupl_result/control_vs_PEC_kmer_counts/diff-counts.tsv.gz | awk 'END{print NR-1}' > ../dekupl.n_de_kmers
du -sb DEkupl_result > ../dekupl.disk_bytes
echo ok > ../dekupl.done
