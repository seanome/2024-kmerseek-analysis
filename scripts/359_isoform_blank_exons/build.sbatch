#!/bin/bash
# Notebook 359, steps 2, 4 and 5: exon table, blank set, decoys, queries. Then it plans
# the kmerseek searches from the query count, and submits them and the scoring job, or
# stops with the estimate if the searches would take more than 48 job-hours (the two
# compute days the prompt allows).
#SBATCH --job-name=359-build
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.log

set -euo pipefail
WORK=$(pwd)
IMG=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.1-rc1.img
SEED=20261020
py() { apptainer exec --bind /scratch "$IMG" python3 "$@"; }

py code/scripts/359_isoform_blank_exons/build_exons.py \
    --sqanti-dir "$WORK/sqanti/qc" --filter-dir "$WORK/sqanti/filter" --prefix wtc11 \
    --interproscan "$WORK/ips/pfam.tsv" \
    --gencode-gtf "$WORK/data/gencode.v39.primary_assembly.annotation.gtf.gz" \
    --seed "$SEED" --out-dir "$WORK/build"

# Stop here if the coordinate mapping is broken (sanity gate below 90%).
py -c "import json,sys; g=json.load(open('$WORK/build/gate.json')); print(g); sys.exit(0 if g['passes'] else 'sanity gate failed: stop and report, the coordinate mapping is wrong')"

NBLANK=$(py -c "import json; print(json.load(open('$WORK/build/build_summary.json'))['blank_exons'])")
if [ "$NBLANK" -eq 0 ]; then
    echo "STOP: the blank set is empty; nothing to search. See build/build_summary.json." >&2
    exit 1
fi
[ "$NBLANK" -lt 200 ] && echo "NOTE: the blank set has $NBLANK exons, fewer than 200; the rule stays as written."

# Chunks of 1,000 queries, real and decoy kept apart.
mkdir -p "$WORK/kmerseek/chunks"
command rm -f "$WORK"/kmerseek/chunks/*.fasta
for side in real decoy; do
    awk -v d="$WORK/kmerseek/chunks" -v s=$side '/^>/{if (n % 1000 == 0) f = sprintf("%s/%s_%04d.fasta", d, s, n / 1000); n++} {print > f}' \
        "$WORK/build/queries_$side.fasta"
done
NQ=$(cat "$WORK"/build/queries_*.fasta | grep -c '^>')
NCHUNK=$(ls "$WORK"/kmerseek/chunks/*.fasta | wc -l)

# Time per setting = index + queries x seconds per query + 10 min per chunk to load the
# index, x 1.5, at least 1 h (settings.tsv and settings.README.md give the sources).
PLAN=$WORK/kmerseek/plan.tsv
awk -F'\t' -v nq="$NQ" -v nc="$NCHUNK" 'NR == 1 {print $0 "\thours"; next}
    {h = 1.5 * ($7 + nq * $8 / 3600 + nc / 6); if (h < 1) h = 1; printf "%s\t%.1f\n", $0, h}' \
    code/scripts/359_isoform_blank_exons/settings.tsv > "$PLAN"
TOTAL=$(awk -F'\t' 'NR > 1 {t += $9} END {printf "%.1f", t}' "$PLAN")
echo "queries $NQ in $NCHUNK chunks; planned job-hours $TOTAL"
cat "$PLAN"
if awk -v t="$TOTAL" 'BEGIN {exit !(t > 48)}'; then
    echo "STOP: the six settings would take $TOTAL job-hours, more than 48. Report the estimate; nothing submitted." | tee "$WORK/kmerseek/ESTIMATE_TOO_LONG.txt"
    exit 1
fi

# Submit one job per setting, then scoring after all six. A job started from inside a
# job inherits SLURM_* variables that would shrink it to this job's CPUs and memory.
env_clean() { env $(env | awk -F= '/^(SLURM|SBATCH)_/ {printf "-u %s ", $1}') "$@"; }
deps=""
while IFS=$'\t' read -r setting alphabet k pen xdrop mem idxh spq hours; do
    [ "$setting" = setting ] && continue
    mins=$(awk -v h="$hours" 'BEGIN {printf "%d", h * 60 + 0.5}')
    id=$(env_clean sbatch --parsable --job-name="359-km-$setting" --mem="${mem}G" --time="$mins" \
        code/scripts/359_isoform_blank_exons/kmerseek.sbatch "$setting" "$alphabet" "$k" "$pen" "$xdrop")
    echo "$setting -> job $id ($mem GB, $mins min)"
    deps="$deps:$id"
done < "$PLAN"
id=$(env_clean sbatch --parsable --dependency="afterok$deps" code/scripts/359_isoform_blank_exons/score.sbatch)
echo "score -> job $id, after$deps"
