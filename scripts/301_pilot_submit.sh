#!/bin/bash
# Notebook 301: submit the build job, then three pilot settings that start only if the
# build succeeds. Run from this branch's clone on Sherlock:
#
#   cd $SCRATCH/2024-kmerseek-analysis-301 && git pull && bash scripts/301_pilot_submit.sh
#
# The three settings span the cost range of notebook 241's 152: the lowest k of the
# largest alphabet, a middle one, and a two-letter alphabet at a k earlier work used.
# Memory per setting, from the dark-set runs on 572,000-protein references (project note
# kmerseek_search_memory_spectrum_load): index 51-83 GB; searches up to 200-312 GB for the
# low-k settings of large alphabets. protein20 k5 is the worst case, so it asks 256 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
KMERSEEK=$SCRATCH/301-kmerseek-v0.4.0/target/release/kmerseek
build=$(sbatch --parsable scripts/301_build.sbatch)
echo "build $build"
while read -r alphabet k penalty xdrop mem; do
    setting=$alphabet:$k:$penalty:$xdrop
    j=$(sbatch --parsable --dependency=afterok:"$build" --mem="$mem" \
        --job-name="301-pilot-$alphabet-k$k" \
        --export=ALL,KMERSEEK="$KMERSEEK",SETTING="$setting" scripts/301_pilot.sbatch)
    echo "pilot $setting ($mem): $j"
done <<'EOF'
protein20 5 0.14 0.57 256G
dayhoff6 12 0.41 1.65 128G
hp_lehninger2 19 1.51 6.03 128G
EOF
