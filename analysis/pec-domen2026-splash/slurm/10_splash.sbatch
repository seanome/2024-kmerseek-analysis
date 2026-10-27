#!/bin/bash
#SBATCH --job-name=nb361-splash
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --output=%x.%j.out
# SPLASH, supervised by fixed sample weights (control -1, PEC +1), on one contrast folder:
#   $PEC_ROOT/contrasts/$CONTRAST/{samples.txt,cjs.csv,input.txt}
# Compactors are run separately (11_compactors.sbatch) on every significant anchor.
# Resources are set at submission (sbatch --cpus-per-task --mem --time).
set -euo pipefail
: "${PEC_ROOT:?}" "${CONTRAST:?}" "${PVAL_COL:?name of the supervised p-value column}"
D=$PEC_ROOT/contrasts/$CONTRAST
cd "$D"
[ -e splash.done ] && { echo "splash already done"; exit 0; }
export PATH=$PEC_ROOT/env/bin:$PATH
while read -r s p; do [ -s "$p" ] || { echo "missing input $p"; exit 2; }; done < input.txt
rm -rf splash_out && mkdir splash_out && cd splash_out
TMP=${L_SCRATCH:-$D/splash_out}/splash_tmp_$SLURM_JOB_ID
rm -rf "$TMP"  # SPLASH makes this folder itself and stops if it exists
N=$SLURM_CPUS_PER_TASK
# SLURM_MEM_PER_NODE is unset with --mem-per-cpu or a partition default; fall back to per-cpu memory
MEM_MB_TOTAL=${SLURM_MEM_PER_NODE:-$(( ${SLURM_MEM_PER_CPU:-4096} * ${SLURM_CPUS_PER_TASK:-1} ))}
MEM_GB=$(( MEM_MB_TOTAL / 1024 ))
KMC_GB=$(( MEM_GB / 3 > 1 ? MEM_GB / 3 : 1 ))
/usr/bin/time -v -o ../splash.time "$PEC_ROOT/splash/splash" \
  --bin_path "$PEC_ROOT/splash" \
  --Cjs_samplesheet ../cjs.csv \
  --pvals_correction_col_name "$PVAL_COL" \
  --with_effect_size_cts --n_most_freq_targets 4 \
  --dump_sample_anchor_target_count_binary \
  --without_compactors \
  --n_threads_stage_1 4 --n_threads_stage_1_internal $(( N / 4 > 0 ? N / 4 : 1 )) --n_threads_stage_2 "$N" \
  --kmc_max_mem_GB "$KMC_GB" \
  --tmp_dir "$TMP" --logs_dir logs \
  --outname_prefix result ../input.txt > ../splash.stdout 2> ../splash.stderr
rm -rf "$TMP"
cd ..
for f in splash_out/result.after_correction.all_anchors.tsv splash_out/result.after_correction.scores.tsv; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }
done
ls splash_out/result_satc/bin*.satc | wc -l > splash.n_satc_bins
du -sb splash_out > splash.disk_bytes
echo ok > splash.done
