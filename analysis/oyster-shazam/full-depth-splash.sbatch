#!/bin/bash
#SBATCH --job-name=oyster-full-splash-corrected
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
# SPLASH 2.11.9 with Compactors (its default) at one depth (argument: 1m, 5m
# or full). Read 1 of each trimmed pair only (duplicate sample names are an
# error in SPLASH, so mates cannot go in as extra samples). Anchor 31,
# target 31, gap 0; c_j +1 for breed and -1 for wild. SPLASH tests anchor
# target composition, not abundance.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
depth=full
cpus=${SLURM_CPUS_PER_TASK:-16}
kmc_gb=32  # Per KMC process, two processes at once; not the whole job budget.
reads="$ROOT/trimmed/$depth"
out="$ROOT/splash/full-memory-corrected"
for s in "${SAMPLES[@]}"; do
  [ -f "$reads/$s.validated.json" ] || { echo "trimmed $s $depth not validated" >&2; exit 1; }
done
[ ! -e "$out" ] || { echo "Output already exists; inspect before retry" >&2; exit 1; }
mkdir -p "$out"; cd "$out"
: > input.txt; : > Cjs.csv
for s in "${SAMPLES[@]}"; do
  printf '%s %s\n' "$s" "$reads/${s}_val_1.fq.gz" >> input.txt
  if [ "${CONDITION[$s]}" = breed ]; then cj=+1; else cj=-1; fi
  printf '%s,%s\n' "$s" "$cj" >> Cjs.csv  # comma-separated, as in SPLASH's example sheet
done
# SPLASH creates its temp folder itself and stops if it already exists.
timed "splash-$depth" "$SPLASH_DIR/splash" --bin_path "$SPLASH_DIR" \
  --anchor_len 31 --target_len 31 --gap_len 0 \
  --kmc_max_mem_GB "$kmc_gb" --n_threads_stage_1 2 --n_threads_stage_1_internal 16 --n_threads_stage_2 "$cpus" \
  --tmp_dir "$L_SCRATCH_JOB/splash-tmp" \
  --Cjs_samplesheet Cjs.csv --dump_sample_anchor_target_count_txt --dump_Cjs \
  --outname_prefix oyster input.txt 2>&1 | tee splash.stdout.log
test -s oyster.after_correction.scores.tsv
touch complete
