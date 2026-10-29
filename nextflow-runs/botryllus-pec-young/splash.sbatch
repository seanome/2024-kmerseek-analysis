#!/bin/bash
#SBATCH --job-name=pec-young-splash
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
# SPLASH 2.11.9 on one time window, without a genome: which 31-base anchors are followed by
# different sequences in treated than in control libraries. Sample weights are fixed
# (control -1, treated +1); the p-value corrected is the one for those weights,
# Cjs_0_pval. Everything else is SPLASH's default, including compactors, which extend each
# significant anchor into the longest sequence the reads support.
#
# Sizing: notebook 361 ran SPLASH on these libraries with the same binary. 1.1 GB of reads
# took 97 s and 2.3 GB of memory; the largest window here (24 h) is 23 GB, so about 20x
# that. KMC gets a third of the memory.
# Usage (from submit_all.sh): sbatch --chdir <run dir> splash.sbatch <repo dir> <window>
set -euo pipefail
REPO=$1; WINDOW=$2
SPLASH_DIR=/scratch/users/olgabot/nb361-pec/splash   # SPLASH 2.11.9 release, installed by PR 201
D=$SCRATCH/botryllus-pec-young/splash/$WINDOW
mkdir -p "$D"
cd "$D"
[ -e splash.done ] && { echo "splash already done for $WINDOW"; exit 0; }
cp "$REPO/splash/$WINDOW/input.txt" "$REPO/splash/$WINDOW/cjs.csv" .
while read -r s p; do [ -s "$p" ] || { echo "missing input $p"; exit 2; }; done < input.txt
rm -rf splash_out && mkdir splash_out && cd splash_out
TMP=$D/splash_tmp_$SLURM_JOB_ID
rm -rf "$TMP"   # SPLASH makes this folder itself and stops if it exists
N=$SLURM_CPUS_PER_TASK
KMC_GB=$(( SLURM_MEM_PER_NODE / 1024 / 3 ))
/usr/bin/time -v -o ../splash.time "$SPLASH_DIR/splash" \
  --bin_path "$SPLASH_DIR" \
  --Cjs_samplesheet ../cjs.csv \
  --pvals_correction_col_name Cjs_0_pval \
  --n_threads_stage_1 4 --n_threads_stage_1_internal $(( N / 4 )) --n_threads_stage_2 "$N" \
  --kmc_max_mem_GB "$KMC_GB" \
  --tmp_dir "$TMP" --logs_dir logs \
  --outname_prefix result ../input.txt > ../splash.stdout 2> ../splash.stderr
rm -rf "$TMP"
cd ..
for f in splash_out/result.after_correction.all_anchors.tsv splash_out/result.after_correction.scores.tsv; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }
done
echo ok > splash.done
