#!/bin/bash
#SBATCH --job-name=nb361-compactors
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --output=%x.%j.out
# Upstream compactors on EVERY significant anchor of the supervised channel, with all reads of
# the contrast (both mates, interleaved). Settings are the ones SPLASH 2.11.9 would pass
# (max_length = 6 x anchor length = 186, epsilon 0.001, beta 0.5, num_kmers 2).
set -euo pipefail
: "${PEC_ROOT:?}" "${CONTRAST:?}"
D=$PEC_ROOT/contrasts/$CONTRAST
cd "$D"
[ -e compactors.done ] && { echo "compactors already done"; exit 0; }
[ -e splash.done ] || { echo "splash not done"; exit 2; }
mkdir -p compactors
cut -d' ' -f2 input.txt > compactors/input_list.txt
SIG=splash_out/result.after_correction.scores.tsv
echo "significant anchors: $(( $(wc -l < $SIG) - 1 ))" > compactors/n_input_anchors.txt
cd compactors
/usr/bin/time -v -o ../compactors.time "$PEC_ROOT/splash/compactors" --keep_temp \
  --max_length 186 --num_threads "$SLURM_CPUS_PER_TASK" --epsilon 0.001 --beta 0.5 --num_kmers 2 \
  --input_format fastq input_list.txt "../$SIG" compactors.tsv > compactors.log 2>&1
[ -s compactors.tsv ] || { echo "no compactors output"; exit 3; }
cd ..
echo ok > compactors.done
