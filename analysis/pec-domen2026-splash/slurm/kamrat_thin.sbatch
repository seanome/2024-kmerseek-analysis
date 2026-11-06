#!/bin/bash
# KaMRaT, t-test and DESeq2, on the first 1/10, 1/100 and 1/1000 of each sample's trimmed read pairs,
# at least one pair (notebook 361). Same commands as 13_dekupl.sbatch, the kamrat process of
# nextflow-runs/pec-domen2026/main.nf and 18_kamrat_dekupl_p05.sbatch. Each thinned set has its own
# root, $S/thin_kamrat/f<F>, shaped like $S, so paper_recovery.py and nfcore_recovery.py can read it.
# DE-kupl's own test may fail on a thin table; only its k-mer count table is needed, so that does not
# stop the task. A step that finds nothing leaves that route with no search folder.
# Searches the old gene models and the new proteome, which the two sets of lists are scored on.
# Every step is timed into <contrast>/time/. Making the thin reads is not timed: a real cheap run
# would take the start of each file.
# Resources: full-depth DE-kupl and KaMRaT peaked at 25.2 GB (exports/paper_recovery/resources.tsv).
#SBATCH --job-name=nb361-kamrat-thin
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs_kamrat_reads/%x.%A_%a.out
set -uo pipefail
S=/scratch/users/olgabot/nb361-pec
export PEC_CODE=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
CONTRASTS=(young_2h young_24h old_24h young_13_49d young_132d old_41_93d)
FRACTIONS=(10 100 1000)
F=${FRACTIONS[$((SLURM_ARRAY_TASK_ID / 6))]}
C=${CONTRASTS[$((SLURM_ARRAY_TASK_ID % 6))]}
export CONTRAST=$C PEC_ROOT=$S/thin_kamrat/f$F DEKUPL_METHOD=limma
D=$PEC_ROOT/contrasts/$C
mkdir -p "$D/reads" "$D/time"
for x in env images Rlib_dekupl; do ln -sfn "$S/$x" "$PEC_ROOT/$x"; done
export PATH=$S/env/bin:$PATH
PT=$S/pec-tools/target/release/pec-tools
N=$SLURM_CPUS_PER_TASK
T="/usr/bin/time -v -o $D/time"
DSG=$PEC_CODE/design/contrast_$C.design.tsv
DESIGN=pair_condition; [ "$C" = young_132d ] && DESIGN=genotype_condition
cp -f "$S/contrasts/$C/design.tsv" "$D/design.tsv"
: > "$D/input.txt"
for s in $(cut -d' ' -f1 "$S/contrasts/$C/input.txt"); do
  pairs=$(python -c "import json;print(json.load(open('$S/trimmed/$s.fastp.json'))['summary']['after_filtering']['total_reads']//2)")
  n=$(( pairs / F )); [ "$n" -lt 1 ] && n=1
  for m in 1 2; do
    [ -s "$D/reads/${s}_$m.fastq.gz" ] || pigz -dc "$S/trimmed/${s}_$m.fastq.gz" 2>/dev/null | head -n $(( n * 4 )) | pigz -p 2 -1 > "$D/reads/${s}_$m.fastq.gz"
  done
  echo "$s $D/reads/$s.il.fastq.gz" >> "$D/input.txt"
done
RAW=$D/dekupl_limma/DEkupl_result/kmer_counts/raw-counts.tsv.gz
[ -s "$RAW" ] || bash "$PEC_CODE/slurm/13_dekupl.sbatch"
[ -s "$RAW" ] || { echo "$C f$F: no k-mer table"; exit 0; }
[ -e "$D/dekupl_limma.time" ] && cp -f "$D/dekupl_limma.time" "$D/time/dekupl.time"
set -e
K="apptainer exec --bind /scratch $S/images/kamrat_v1.2.sif kamrat"
mkdir -p "$D/kamrat_ttest"
cd "$D/kamrat_ttest"
if [ ! -s contigs_all.tsv.gz ]; then
  awk -F'\t' 'NR > 1 { print $1 "\t" $2 }' "$DSG" > design.txt
  rm -rf idx && mkdir idx
  $T/kamrat_index.time $K index -intab "$RAW" -outdir idx -klen 31 -unstrand -nfbase 1000000000 > kamrat.log 2>&1 || { echo "$C f$F: index failed"; exit 0; }
  $T/kamrat_score.time $K score -idxdir idx -scoreby ttest.padj -design design.txt -outfmt bin -outpath score.bin >> kamrat.log 2>&1
  $T/kamrat_merge.time $K merge -idxdir idx -overlap 30-15 -with score.bin:min -interv pearson:0.20 -outfmt tab -counts mean:float -outpath contigs_all.tsv >> kamrat.log 2>&1
  gzip contigs_all.tsv
  rm -rf idx score.bin
fi
cd "$D"
search() {
  src=$1
  [ -e "search_$src/search.done" ] && return 0
  rm -rf "search_$src" && mkdir "search_$src" && cd "search_$src"
  ln -s "../$src/contigs.tsv" seqs.tsv
  "$PT" translate seqs.tsv pep 20 contig 2> translate.log
  if [ -s pep.peptides.faa ]; then
    seqkit split2 -s 500000 -O chunks pep.peptides.faa 2> split.log
    for fa in chunks/*.faa; do
      c=$(basename "$fa" .faa); c=${c##*.}
      sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$c" "$PWD/$fa") \
        --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c "$N" -o "$c.scaled5.zip"
      for db in proteome botznik; do
        for spec in protein:7 dayhoff:12 hp:30; do
          m=${spec%:*}; k=${spec#*:}
          for s in 10 100 5; do
            sourmash scripts manysearch "$c.scaled5.zip" "$S/db/$db.$m.k$k.s$s.rocksdb" -m $m -k $k -s $s \
              -t 0.000001 -N -c "$N" -o "ms.$c.$db.$m.k$k.s$s.csv" 2>> manysearch.log
          done
        done
      done
      rm "$c.scaled5.zip"
    done
    python "$PEC_CODE/scripts/make_pairs.py" pairs.tsv hits.parquet ms.*.csv
    rm ms.*.csv
    for db in proteome botznik; do
      python "$PEC_CODE/scripts/pair_align.py" pairs.$db.tsv pep.peptides.faa "$S/db/$db.faa" align.$db.parquet "$N"
    done
  fi
  touch search.done
  cd "$D"
}
export -f search
export D PT N S PEC_CODE
if [ ! -s kamrat_p05/contigs.tsv ]; then
  mkdir -p kamrat_p05
  $T/select_ttest.time python "$PEC_CODE/scripts/select_kamrat_nominal.py" kamrat_ttest/contigs_all.tsv.gz "$DSG" kamrat_p05 0 > kamrat_p05/select.log
fi
$T/search_ttest.time bash -c "set -euo pipefail; search kamrat_p05" || echo "$C f$F: t-test search failed"
if [ ! -s kamrat_deseq2_p05/contigs.tsv ]; then
  mkdir -p kamrat_deseq2 kamrat_deseq2_p05
  if $T/counts.time python "$PEC_CODE/scripts/kamrat_contig_counts.py" kamrat_ttest/contigs_all.tsv.gz "$RAW" "$DSG" kamrat_deseq2 > kamrat_deseq2/counts.log \
    && (cd "$PEC_CODE/scripts" && $T/deseq.time python anchor_deseq.py "$D/kamrat_deseq2/contig_counts.tsv" "$D/kamrat_deseq2/samples.txt" "$DSG" "$D/kamrat_deseq2/deseq" "$N" "$DESIGN" > "$D/kamrat_deseq2/deseq.log" 2>&1) \
    && $T/select_deseq.time python "$PEC_CODE/scripts/select_kamrat_deseq.py" "kamrat_deseq2/deseq/deseq_$DESIGN.parquet" kamrat_deseq2_p05 > kamrat_deseq2_p05/select.log; then
    rm -f kamrat_deseq2/contig_counts.tsv
    $T/search_deseq.time bash -c "set -euo pipefail; search kamrat_deseq2_p05" || echo "$C f$F: DESeq2 search failed"
  else
    echo "$C f$F: DESeq2 step failed"
  fi
fi
rm -rf "$D/dekupl_limma/DEkupl_result/kmer_counts/"*.tmp 2>/dev/null || true
