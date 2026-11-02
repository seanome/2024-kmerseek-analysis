#!/bin/bash
# Submit the frame-method comparison: stops, k-mer scores and DIAMOND blastx in parallel, then the
# summary once all three succeed. Run on a Sherlock login node:
#   PEC_ROOT=/scratch/users/olgabot/nb361-pec CODE=<folder with frame_stops.py, frame_methods.py,
#   22_frame_methods.sbatch> BZ=<old gene-model proteins fasta> bash 22_frame_methods_submit.sh
# Resources come from the earlier frame_test/*.time files: frame-scores took 18-24 s and at most
# 1.8 GB for 1.04 M reads, so 8 settings per reference fit in 2 h.
set -euo pipefail
: "${PEC_ROOT:?}" "${CODE:?}" "${BZ:?}"
cd "$PEC_ROOT/sra_search/frame_test" && mkdir -p methods && cd methods
SB="sbatch --parsable"
a=$($SB --export=ALL,STEP=stops,PEC_ROOT=$PEC_ROOT,CODE=$CODE,BZ=$BZ -c 1 --mem=4G -t 01:00:00 "$CODE/22_frame_methods.sbatch")
# one k-mer job per reference, so they run side by side; the held-out split is made by the job
# that needs it. Protein 10-mers and Dayhoff 18-mers are new: about 200 M distinct Swiss-Prot
# k-mers in a u64 hash set is near 2.5 GB, doubling briefly while the set grows, so 16 GB.
b=""
for db in swissprot botznik swissprot_botznik newgenome newgenome_halfA; do
  j=$($SB --export=ALL,STEP=kmers,DB=$db,PEC_ROOT=$PEC_ROOT,CODE=$CODE,BZ=$BZ -c 1 --mem=16G -t 02:00:00 \
      -J nb361-frame-kmers-$db "$CODE/22_frame_methods.sbatch")
  b=$b${b:+:}$j
done
c=$($SB --export=ALL,STEP=blastx,PEC_ROOT=$PEC_ROOT,CODE=$CODE,BZ=$BZ -c 16 --mem=32G -t 04:00:00 "$CODE/22_frame_methods.sbatch")
d=$($SB --export=ALL,STEP=summary,PEC_ROOT=$PEC_ROOT,CODE=$CODE,BZ=$BZ -c 1 --mem=24G -t 01:00:00 \
  --dependency=afterok:$a:$b:$c "$CODE/22_frame_methods.sbatch")
echo "stops $a  kmers $b  blastx $c  summary $d"
