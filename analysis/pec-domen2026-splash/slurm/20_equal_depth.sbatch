#!/bin/bash
#SBATCH --job-name=nb361-eqdepth
#SBATCH --account=ayeletv
#SBATCH --output=%x.%j.out
# Equal-depth test (notebook 361): does SPLASH still call PEC vs control differences when every
# sample has the same number of read pairs? One stage per call, chosen by STAGE:
#
#   subsample   SAMPLE, CONTRAST, SRC (folder with <sample>_{1,2}.fastq.gz), DEPTHS (read pairs,
#               largest first, joined by ':'). seqtk sample -s 361 draws the same pairs from R1 and R2; each
#               smaller depth is drawn from the one above it, with the same seed.
#   splash      CONTRAST, DEPTH. Same SPLASH settings as main.nf, minus the satc count dump
#               (only anchor counts read it; it changes no test).
#   compactors  CONTRAST, DEPTH. Same settings as main.nf, every anchor in scores.tsv.
#   search      CONTRAST, DEPTH. Same peptides and sourmash settings as main.nf, against the old
#               gene models (botznik) only: that is the database the gene mapping reads.
#   control     DEPTH. control_truth.py on the subsampled control (SPLASH channel only).
#
# Everything is written under $PEC_ROOT/equal_depth/<contrast>/<depth>/. DEPTH=full links the
# full-depth trimmed reads of the kept samples instead of subsampling them.
# Resources are set at submission (scripts/equal_depth_submit.sh).
set -euo pipefail
: "${PEC_ROOT:?}" "${CODE_DIR:?}" "${STAGE:?}"
ENV=$PEC_ROOT/env/bin
SEQTK=$PEC_ROOT/equal_depth/seqtk_env/bin/seqtk
SPLASH=$PEC_ROOT/splash
PECTOOLS=$PEC_ROOT/pec-tools/target/release/pec-tools
DB=$PEC_ROOT/db
export PATH=$ENV:$PATH
N=${SLURM_CPUS_PER_TASK:-1}

case "$STAGE" in
subsample)
  : "${SAMPLE:?}" "${CONTRAST:?}" "${SRC:?}" "${DEPTHS:?}"
  in1=$SRC/${SAMPLE}_1.fastq.gz; in2=$SRC/${SAMPLE}_2.fastq.gz
  for d in ${DEPTHS//:/ }; do
    out=$PEC_ROOT/equal_depth/$CONTRAST/$d/reads
    mkdir -p "$out"
    if [ ! -e "$out/$SAMPLE.done" ]; then
      # seqtk keeps a reservoir of d records; with the same seed and the same number of records
      # it picks the same positions in R1 and R2
      $SEQTK sample -s 361 "$in1" "$d" | pigz -p 2 -1 > "$out/${SAMPLE}_1.tmp.fastq.gz" &
      p1=$!
      $SEQTK sample -s 361 "$in2" "$d" | pigz -p 2 -1 > "$out/${SAMPLE}_2.tmp.fastq.gz" &
      p2=$!
      wait $p1; wait $p2
      # interleave (R1 record, then its R2 record) and refuse any pair whose names differ
      paste <(pigz -dc "$out/${SAMPLE}_1.tmp.fastq.gz" | paste - - - -) \
            <(pigz -dc "$out/${SAMPLE}_2.tmp.fastq.gz" | paste - - - -) \
        | awk -F'\t' -v OFS='\n' '{ a = $1; b = $5; sub(/ .*/, "", a); sub(/ .*/, "", b);
                                    sub(/\/[12]$/, "", a); sub(/\/[12]$/, "", b);
                                    if (a != b) { print "pair names differ at pair " NR ": " a " " b > "/dev/stderr"; exit 3 }
                                    print $1, $2, $3, $4, $5, $6, $7, $8 }' \
        | pigz -p "$N" -1 > "$out/$SAMPLE.il.tmp.fastq.gz"
      got=$(pigz -dc "$out/$SAMPLE.il.tmp.fastq.gz" | awk 'END { print NR / 8 }')
      [ "$got" = "$d" ] || { echo "$SAMPLE depth $d: got $got pairs"; exit 3; }
      mv "$out/${SAMPLE}_1.tmp.fastq.gz" "$out/${SAMPLE}_1.fastq.gz"
      mv "$out/${SAMPLE}_2.tmp.fastq.gz" "$out/${SAMPLE}_2.fastq.gz"
      mv "$out/$SAMPLE.il.tmp.fastq.gz" "$out/$SAMPLE.il.fastq.gz"
      echo "$got" > "$out/$SAMPLE.done"
    fi
    in1=$out/${SAMPLE}_1.fastq.gz; in2=$out/${SAMPLE}_2.fastq.gz
  done
  ;;

splash)
  : "${CONTRAST:?}" "${DEPTH:?}"
  D=$PEC_ROOT/equal_depth/$CONTRAST/$DEPTH
  cd "$D"
  [ -e splash.done ] && { echo "splash already done"; exit 0; }
  while read -r s p; do [ -s "$p" ] || { echo "missing input $p"; exit 2; }; done < input.txt
  [ "$(wc -l < input.txt)" = "$(wc -l < cjs.csv)" ] || { echo "samples and cjs differ"; exit 2; }
  rm -rf splash_out && mkdir splash_out && cd splash_out
  TMP=${L_SCRATCH:-$D}/splash_tmp_$SLURM_JOB_ID
  rm -rf "$TMP"
  MEM_GB=$(( ${SLURM_MEM_PER_NODE:-4096} / 1024 ))
  /usr/bin/time -v -o ../splash.time "$SPLASH/splash" --bin_path "$SPLASH" --Cjs_samplesheet ../cjs.csv \
    --pvals_correction_col_name Cjs_0_pval --with_effect_size_cts --n_most_freq_targets 4 \
    --without_compactors \
    --n_threads_stage_1 4 --n_threads_stage_1_internal $(( N / 4 > 0 ? N / 4 : 1 )) \
    --n_threads_stage_2 "$N" --kmc_max_mem_GB $(( MEM_GB / 3 )) \
    --tmp_dir "$TMP" --logs_dir logs --outname_prefix result ../input.txt \
    > ../splash.stdout 2> ../splash.stderr
  rm -rf "$TMP"
  cd ..
  test -s splash_out/result.after_correction.all_anchors.tsv
  test -s splash_out/result.after_correction.scores.tsv
  touch splash.done
  ;;

compactors)
  : "${CONTRAST:?}" "${DEPTH:?}"
  D=$PEC_ROOT/equal_depth/$CONTRAST/$DEPTH
  cd "$D"
  [ -e compactors.done ] && { echo "compactors already done"; exit 0; }
  rm -rf compactors && mkdir compactors && cd compactors
  cut -d' ' -f2 ../input.txt > input_list.txt
  echo "significant anchors: $(( $(wc -l < ../splash_out/result.after_correction.scores.tsv) - 1 ))" > n_input_anchors.txt
  /usr/bin/time -v -o ../compactors.time "$SPLASH/compactors" --max_length 186 --num_threads "$N" \
    --epsilon 0.001 --beta 0.5 --num_kmers 2 --input_format fastq input_list.txt \
    ../splash_out/result.after_correction.scores.tsv compactors.tsv > compactors.log 2>&1
  test -s compactors.tsv
  cd .. && touch compactors.done
  ;;

search)
  : "${CONTRAST:?}" "${DEPTH:?}"
  D=$PEC_ROOT/equal_depth/$CONTRAST/$DEPTH
  cd "$D"
  [ -e search.done ] && { echo "search already done"; exit 0; }
  rm -rf search_compactors && mkdir search_compactors && cd search_compactors
  ln -s ../compactors/compactors.tsv seqs.tsv
  $PECTOOLS translate seqs.tsv pep 20 compactor 2> translate.log
  if [ -s pep.peptides.faa ]; then
    seqkit split2 -s 500000 -O chunks pep.peptides.faa 2> split.log
    for fa in chunks/*.faa; do
      c=$(basename "$fa" .faa); c=${c##*.}
      sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$c" "$PWD/$fa") \
        --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c "$N" -o "$c.scaled5.zip"
      for spec in protein:7 dayhoff:12 hp:30; do
        m=${spec%:*}; k=${spec#*:}
        for s in 10 100 5; do
          sourmash scripts manysearch "$c.scaled5.zip" "$DB/botznik.$m.k$k.s$s.rocksdb" -m "$m" -k "$k" -s "$s" \
            -t 0.000001 -N -c "$N" -o "ms.$c.botznik.$m.k$k.s$s.csv" 2>> manysearch.log
        done
      done
    done
    python "$CODE_DIR/scripts/make_pairs.py" pairs.tsv hits.parquet ms.*.csv
    python "$CODE_DIR/scripts/pair_align.py" pairs.botznik.tsv pep.peptides.faa "$DB/botznik.faa" align.botznik.parquet "$N"
    rm -f ms.*.csv ./*.scaled5.zip
  fi
  rm -f seqs.tsv
  cd .. && touch search.done
  ;;

control)
  : "${DEPTH:?}"
  D=$PEC_ROOT/equal_depth/control/$DEPTH
  cd "$D"
  python "$CODE_DIR/scripts/control_truth.py" "$CODE_DIR/control/control_transcripts.fa" \
    "$CODE_DIR/control/truth_design.tsv" . control_eval > control_truth.log 2>&1
  test -s control_eval/control_truth_summary.json
  ;;

*)
  echo "unknown STAGE $STAGE"; exit 2 ;;
esac
