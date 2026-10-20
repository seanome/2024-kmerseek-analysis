#!/bin/bash
#SBATCH --job-name=351-index
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --time=24:00:00
#
# Notebook 351, step 2: one kmerseek v0.4.0 index of the excluded Swiss-Prot for one
# setting (NB351_SETTING = name, NB351_ALPHABET, NB351_KSIZE). Scaled 1 and every other
# option at its default, as preregistered; the default fits the E-value constants for
# mismatch penalty 2 and X-drop 8 on 200 database sequences. --mem and --cpus-per-task
# come from submit.sh, which explains them.
NB351_STEP=index-${NB351_SETTING:?}
source "$NB351_REPO/analysis/351_short_feature_boundaries/sherlock/common.sh"
: "${NB351_ALPHABET:?}" "${NB351_KSIZE:?}"
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$D/index"
IDX=$D/index/$NB351_SETTING.kmerseek
rm -rf "$IDX"
"$KMERSEEK" --version
"$KMERSEEK" index -i "$D/db/swissprot_excluded.fasta" -o "$IDX" -k "$NB351_KSIZE" \
    -a "$NB351_ALPHABET" -s 1 --kmer-stats-out "$D/index/$NB351_SETTING.spectrum.csv.gz" \
    --ka-survival-out "$D/index/$NB351_SETTING.ka_survival.csv" 2> "$D/index/$NB351_SETTING.index.log"
tail -5 "$D/index/$NB351_SETTING.index.log"
echo "351 index $NB351_SETTING done"
