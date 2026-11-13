#!/usr/bin/env bash
# Run one treated-versus-control contrast of the Domen et al. 2026 PEC data end to end:
#   fastp trim -> SPLASH 2 (supervised Cjs = PEC/control) + compactors -> Rust six-frame
#   translation -> sourmash sketches (protein k7, Dayhoff k12, HP k30; scaled 5/10/100)
#   -> manysearch against SwissProt and the new Botryllus proteome.
# Usage: run_contrast.sh <timepoint_inferred value, e.g. 2h> [threads]
set -euo pipefail
TP=$1; THREADS=${2:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
ENV=/home/claude/work/tools/env
PT=$HERE/rust/target/release/pec-tools
OUT=$HERE/contrast_$TP; mkdir -p "$OUT"; cd "$OUT"
log(){ echo "$(date +%T) $*" | tee -a "$OUT/run.log"; }

# 1. samples of this contrast
awk -F'\t' -v tp="$TP" 'NR==1||($8==tp && $13=="yes")' "$HERE/samples_all64.tsv" > samples.tsv
N=$(($(wc -l < samples.tsv)-1)); log "contrast $TP: $N samples"

# 2. trim (fastp, paired, adapter auto-detect, NextSeq poly-G), interleaved output = one file per sample
mkdir -p "$HERE/trimmed"
tail -n +2 samples.tsv | while IFS=$'\t' read -r sample rest; do
  f="$HERE/trimmed/$sample.il.fastq.gz"
  if [ ! -s "$f.done" ]; then
    r1="$HERE/reads/${sample}_1.fastq.gz"; r2="$HERE/reads/${sample}_2.fastq.gz"
    [ -s "$r1.done" ] && [ -s "$r2.done" ] || { log "reads for $sample not complete"; exit 2; }
    /home/claude/work/tools/env/bin/time -f "%e s %M KB" -o "$HERE/trimmed/$sample.fastp.time" \
      $ENV/bin/fastp -i "$r1" -I "$r2" --stdout --detect_adapter_for_pe --trim_poly_g --length_required 50 \
        --thread $THREADS -j "$HERE/trimmed/$sample.fastp.json" -h "$HERE/trimmed/$sample.fastp.html" 2> "$HERE/trimmed/$sample.fastp.log" \
      | gzip -1 > "$f.part" && mv "$f.part" "$f" && echo ok > "$f.done"
    log "trimmed $sample $(cat $HERE/trimmed/$sample.fastp.time)"
  fi
done

# 3. SPLASH input list + predefined Cjs (control = -1, PEC = +1); correction applied to the supervised p-value
tail -n +2 samples.tsv | awk -F'\t' -v d="$HERE/trimmed" '{print $1" "d"/"$1".il.fastq.gz"}' > splash_input.txt
tail -n +2 samples.tsv | awk -F'\t' '{print $1","($3=="PEC"?1:-1)}' > cjs.csv
if [ ! -s splash.done ]; then
  rm -rf splash_tmp
  /home/claude/work/tools/env/bin/time -f "%e s %M KB" -o splash.time \
   $ENV/bin/splash --n_threads_stage_1 1 --n_threads_stage_1_internal $THREADS --n_threads_stage_2 $THREADS \
    --kmc_max_mem_GB 4 --Cjs_samplesheet cjs.csv --pvals_correction_col_name Cjs_0_pval \
    --with_effect_size_cts --n_most_freq_targets 4 --dump_sample_anchor_target_count_binary \
    --compactors_config "$HERE/compactors_config.json" --tmp_dir splash_tmp --logs_dir splash_logs \
    --outname_prefix result splash_input.txt > splash.stdout 2> splash.stderr && echo ok > splash.done
  log "splash $(cat splash.time)"
fi

# 4. six-frame translation of compactors (Rust), min 20 aa
if [ ! -s translate.done ]; then
  $PT translate compactors/after_correction.scores.tsv pep 20 compactor 2> translate.log && echo ok > translate.done
  log "translate: $(cat translate.log)"
fi

# 5. sourmash sketches of peptides and searches
if [ ! -s search.done ]; then
  sourmash sketch protein --singleton -p protein,k=7,scaled=5,abund -p dayhoff,k=12,scaled=5,abund -p hp,k=30,scaled=5,abund \
      -o pep.scaled5.zip pep.peptides.faa 2> sketch.log
  for s in 5 10 100; do for db in proteome swissprot; do
    dbz=$([ $db = proteome ] && echo /home/claude/work/proteome/botryllus_new_proteome.scaled5.zip || echo /home/claude/work/swissprot/swissprot.scaled5.zip)
    for enc in protein dayhoff hp; do
      k=$([ $enc = protein ] && echo 7 || ([ $enc = dayhoff ] && echo 12 || echo 30))
      [ -s "$dbz" ] || { log "missing database $dbz; skipping $db"; continue; }
      # multisearch: containment of each peptide sketch in each protein sketch, plus the
      # probability that the overlap is chance given all queries and matches (-p).
      sourmash scripts multisearch pep.scaled5.zip "$dbz" --moltype $enc -k $k --scaled $s --threshold 0.0 -p -c $THREADS \
        -o search_${db}_${enc}_s${s}.csv 2>> search.log || log "multisearch $db $enc scaled $s failed"
    done; done; done
  echo ok > search.done; log "search done"
fi
log "contrast $TP finished"
