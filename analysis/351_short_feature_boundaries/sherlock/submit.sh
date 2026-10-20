#!/bin/bash
# Submit notebook 351's whole run on Sherlock as one chain of jobs. Run from anywhere
# inside the repository checkout on a Sherlock login node:
#   bash analysis/351_short_feature_boundaries/sherlock/submit.sh
#
# Chain: prepare -> three index builds -> plan (sizes and submits the kmerseek searches)
#                \-> hmmscan array, blastp array
#
# First asks (no earlier run of these exact jobs; numbers below are modelled, not measured):
#   index memory = 45 GB build + the E-value fit, times 1.5. The fit's memory was modelled
#   in the dark-set pipeline as 300 GB at 12 bits per k-mer and scaled 1 (extrapolated,
#   not measured), halving every 2.5 bits. Bits per k-mer: protein20 k5 21.6, dayhoff6 k7
#   18.1, hp_lehninger2 k12 12.0. So: about 21, 55 and 300 GB of fit, asked as 96, 128 and
#   512 GB.
#   search memory: set by the plan step from each index's own k-mer spectrum.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
S=$REPO/analysis/351_short_feature_boundaries/sherlock
D=${NB351_DIR:-$SCRATCH/351_short_feature_boundaries}
mkdir -p "$D/logs"
cd "$REPO"
echo "repo commit $(git rev-parse --short HEAD) on branch $(git rev-parse --abbrev-ref HEAD)"

SETTINGS="protein20_k5:protein20:5:96:16 dayhoff6_k7:dayhoff6:7:128:16 hp_lehninger2_k12:hp_lehninger2:12:512:32"
printf '%-20s %8s %5s %8s\n' job memory cpus time
printf '%-20s %8s %5s %8s\n' prepare 64G 16 8h
for spec in $SETTINGS; do IFS=: read -r n a k m c <<< "$spec"; printf '%-20s %8s %5s %8s\n' "index $n" "${m}G" "$c" 24h; done
printf '%-20s %8s %5s %8s\n' "hmmscan x8" 16G 16 12h "blastp x8" 32G 16 12h "kmerseek x8 per run" "from plan" 32 24h

# Refuse early if the largest ask cannot be met by any node in these partitions.
sbatch --test-only --account=ayeletv --partition=hns,normal --mem=512G --cpus-per-task=32 \
    --time=24:00:00 --wrap true 2>&1 | tail -1

EXP=ALL,NB351_REPO=$REPO,NB351_DIR=$D
O="--output=$D/logs/%x-%j.out"
prep=$(sbatch --parsable $O --export=$EXP "$S/01_prepare.sbatch")
deps=()
for spec in $SETTINGS; do
    IFS=: read -r n a k m c <<< "$spec"
    j=$(sbatch --parsable $O --dependency=afterok:$prep --mem=${m}G --cpus-per-task=$c \
        --job-name=351-index-$n \
        --export=$EXP,NB351_SETTING=$n,NB351_ALPHABET=$a,NB351_KSIZE=$k "$S/02_index.sbatch")
    deps+=("$j"); echo "index $n: $j"
done
# name:alphabet:ksize for each setting, passed through the environment (--export=ALL).
NB351_SETTINGS=$(for s in $SETTINGS; do printf '%s ' "${s%:*:*}"; done)
export NB351_SETTINGS
plan=$(sbatch --parsable $O --dependency=afterany:$(IFS=:; echo "${deps[*]}") \
    --export=$EXP "$S/03_plan_searches.sbatch")
AO="--output=$D/logs/%x-%A_%a.out"
hmm=$(sbatch --parsable $AO --dependency=afterok:$prep --array=0-7 --export=$EXP "$S/05_hmmscan.sbatch")
bl=$(sbatch --parsable $AO --dependency=afterok:$prep --array=0-7 --export=$EXP "$S/06_blastp.sbatch")
echo "prepare $prep, plan $plan, hmmscan $hmm, blastp $bl"
echo "Watch: squeue -u \$USER -o '%.12i %.24j %.8T %.10M %.6m %R' | grep 351"
