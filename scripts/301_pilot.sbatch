#!/bin/bash
#SBATCH --job-name=301-pilot
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=301-pilot-%x-%j.out
#
# Notebook 301 pilot, one kmerseek setting per job: index the step 2 Swiss-Prot database
# (568,962 proteins, every query's UniRef90 cluster left out), search a few queries and
# their decoys, and check that region_evalue is finite. Memory is passed with --mem at
# submission (see 301_pilot_submit.sh), since it differs by setting.
#
# Needs: KMERSEEK (the v0.4.0 binary from 301_build.sbatch), SETTING
# ("alphabet:k:penalty:xdrop", from notebooks/241_plan.json).
set -euo pipefail
: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}" "${KMERSEEK:?}" "${SETTING:?}"
IFS=: read -r ALPHABET K PENALTY XDROP <<< "$SETTING"
REPO=$SLURM_SUBMIT_DIR
BASE=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}
DB=$BASE/step2/swissprot_minus_query_clusters.fasta
TAG=${ALPHABET}.k${K}.s1.lctrue.c${PENALTY}.x${XDROP}
OUT=$BASE/pilot/$TAG
mkdir -p "$OUT"
module load gcc/14.2.0 python/3.12.1
"$KMERSEEK" --version

timed() {  # name, then the command; wall time and peak RSS from GNU time
    local name=$1; shift
    /usr/bin/time -v -o "$OUT/$name.time" "$@" > "$OUT/$name.log" 2>&1
}

[ -f "$OUT/idx/CURRENT" ] || timed index "$KMERSEEK" index -i "$DB" -o "$OUT/idx" \
    -k "$K" -s 1 -a "$ALPHABET" --remove-low-complexity \
    --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" \
    --ka-queries 200 --ka-survival-out "$OUT/ka_survival.csv"

timed search "$KMERSEEK" search -q "$BASE/pilot/pilot_queries.fasta" -t "$OUT/idx" \
    -k "$K" -a "$ALPHABET" --threshold 0 --min-shared-kmers 1 --max-query-pvalue 1 \
    --min-region-score 0 --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" \
    -o "$OUT/search.csv"

python3 -I "$REPO/scripts/pilot_301_kmerseek.py" check \
    --queries "$BASE/pilot/pilot_queries.fasta" --search-csv "$OUT/search.csv" \
    | tee "$OUT/check.tsv"
grep -hE "Elapsed|Maximum resident" "$OUT"/index.time "$OUT"/search.time 2>/dev/null || true
du -sh "$OUT/idx"
echo "301 pilot done: $TAG"
