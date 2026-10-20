#!/bin/bash
#SBATCH --job-name=353-kmerseek
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# Notebook 353, step 2 (array, one task per line of kmerseek_settings.tsv): index the
# evidence proteins and search every query and shuffle against them.
# Sizing: no earlier run on this database. It holds 2,883 proteins (about 1.7 million
# residues), about 1/120 of Swiss-Prot; the E-value fit searches 200 of its own sequences.
# 16 GB and 2 h are a generous first ask for a database this small; check afterwards with
#   sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State
# and record the measured numbers in report.md.
source "$NB353_REPO/analysis/353_measurement_retrieval/sherlock/common.sh"
line=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR == n' "$D/in/kmerseek_settings.tsv")
IFS=$'\t' read -r SETTING ALPHABET KSIZE <<< "$line"
NB353_STEP=kmerseek-$SETTING
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$D/index" "$D/kmerseek_out"
IDX=$D/index/$SETTING.kmerseek
rm -rf "$IDX"
/usr/bin/time -v "$KMERSEEK" index -i "$D/in/evidence_proteins.fasta" -o "$IDX" -k "$KSIZE" \
    -a "$ALPHABET" -s 1 2> "$D/index/$SETTING.index.log"
tail -25 "$D/index/$SETTING.index.log"
/usr/bin/time -v "$KMERSEEK" search -q "$D/in/queries_and_controls.fasta" -t "$IDX" -a "$ALPHABET" \
    --extend-mismatch-penalty 2 --extend-xdrop 8 --min-shared-kmers 1 --max-query-pvalue 1.0 \
    -o "$D/kmerseek_out/$SETTING.csv" 2> "$D/kmerseek_out/$SETTING.search.log"
tail -25 "$D/kmerseek_out/$SETTING.search.log"
du -sh "$IDX" "$D/kmerseek_out/$SETTING.csv"
echo "353 kmerseek $SETTING done"
