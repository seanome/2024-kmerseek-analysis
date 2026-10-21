#!/bin/bash
# Notebook 301, step 5: build the query subset, then one kmerseek job per alphabet of
# notebook 241's list, each starting only if the subset job succeeds. Run from this
# branch's clone on Sherlock:
#
#   cd $SCRATCH/2024-kmerseek-analysis-301 && git pull && bash scripts/301_kmerseek_submit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}
KMERSEEK=$SCRATCH/301-kmerseek-v0.4.0/target/release/kmerseek
"$KMERSEEK" --version | grep -q '^kmerseek 0.4.0$'
mkdir -p "$BASE/step5"
subset=$(sbatch --parsable -J 301-subset -A ayeletv -p hns -c 1 --mem=4G -t 00:30:00 \
    -o 301-subset-%j.out --wrap "module load python/3.12.1 && python3 -I scripts/build_301_step5_subset.py \
    --step2 $BASE/step2 --step3 $BASE/step3 --out-fasta $BASE/step5/subset_queries.fasta \
    --out-ids $BASE/step5/subset_query_ids.txt")
echo "subset $subset"
module load python/3.12.1
for a in $(python3 -I -c "import json; print(' '.join(sorted({x['alphabet'] for x in json.load(open('notebooks/241_plan.json'))})))"); do
    j=$(sbatch --parsable --dependency=afterok:"$subset" -J "301-$a" \
        --export=ALL,ALPHABET="$a",KMERSEEK="$KMERSEEK" scripts/301_kmerseek_alphabet.sbatch)
    echo "$a $j"
done
