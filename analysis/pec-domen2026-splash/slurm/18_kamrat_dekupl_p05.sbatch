#!/bin/bash
#SBATCH --job-name=nb361-p05-routes
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=%x.%A_%a.out
# KaMRaT and DE-kupl at uncorrected p < 0.05, no fold-change cutoff (added 2026-11-02, Olga: "use
# nominal p < 0.05 for kamrat and de-kupl" and "do KAMRAT + DeSeq2 not just a ttest"). One
# contrast per array task; task 6 is the simulated control, which gets the selections but no
# protein search (control_truth.py scores it on k-mers).
#   kamrat_p05          KaMRaT contigs, Welch t-test, p < 0.05     select_kamrat_nominal.py ... 0
#   kamrat_deseq2       KaMRaT contig counts back on the read scale + PyDESeq2 (one design:
#                       ~ genotype + condition for young_132d, ~ pair + condition otherwise)
#   kamrat_deseq2_p05   its contigs with p < 0.05                   select_kamrat_deseq.py
#   dekupl_limma_p05    DE-kupl k-mers with limma p < 0.05 (raw_pvals.txt.gz), split by
#                       direction and merged into contigs with DE-kupl's own mergeTags -k 31 -n
# then the protein search of each (same commands as the search process of main.nf).
# Every step is timed with /usr/bin/time -v into <folder>/<step>.time, for the resources table.
# Resources: the largest earlier steps on these inputs were the young_24h compactor search
# (66.5 GB, 58 min, 16 CPUs) and young_2h anchor counts + PyDESeq2 on 4.6 M anchors, three
# designs (26.9 GB, 2 h 2 min); here PyDESeq2 fits one design on up to 6.4 M contigs.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
CONTRASTS=(young_2h young_24h old_24h young_13_49d young_132d old_41_93d control)
C=${CONTRASTS[$SLURM_ARRAY_TASK_ID]}
D=$PEC_ROOT/contrasts/$C
if [ "$C" = control ]; then DSG=$PEC_CODE/control/control.design.tsv; else DSG=$PEC_CODE/design/contrast_$C.design.tsv; fi
DESIGN=pair_condition; [ "$C" = young_132d ] && DESIGN=genotype_condition
export PATH=$PEC_ROOT/env/bin:$PATH
PT=$PEC_ROOT/pec-tools/target/release/pec-tools
N=$SLURM_CPUS_PER_TASK
T="/usr/bin/time -v -o"
W=$(mktemp -d "$PEC_ROOT/tmp_p05_${C}_XXXX")
cd "$W"

# --- KaMRaT t-test, p < 0.05
if [ ! -s "$D/kamrat_p05/contigs.tsv" ]; then
  mkdir -p kamrat_p05
  $T kamrat_p05/select.time python "$PEC_CODE/scripts/select_kamrat_nominal.py" \
    "$D/kamrat_ttest/contigs_all.tsv.gz" "$DSG" kamrat_p05 0 > kamrat_p05/select.log
  test -s kamrat_p05/contigs.tsv
  mv kamrat_p05 "$D/"
fi
cat "$D/kamrat_p05/summary.json"

# --- KaMRaT contigs + PyDESeq2, p < 0.05
if [ ! -s "$D/kamrat_deseq2_p05/contigs.tsv" ]; then
  mkdir -p kamrat_deseq2
  $T kamrat_deseq2/counts.time python "$PEC_CODE/scripts/kamrat_contig_counts.py" \
    "$D/kamrat_ttest/contigs_all.tsv.gz" "$D/dekupl_limma/DEkupl_result/kmer_counts/raw-counts.tsv.gz" \
    "$DSG" kamrat_deseq2 > kamrat_deseq2/counts.log
  (cd "$PEC_CODE/scripts" && $T "$W/kamrat_deseq2/deseq.time" python anchor_deseq.py \
    "$W/kamrat_deseq2/contig_counts.tsv" "$W/kamrat_deseq2/samples.txt" "$DSG" "$W/kamrat_deseq2/deseq" "$N" "$DESIGN" \
    > "$W/kamrat_deseq2/deseq.log" 2>&1)
  gzip kamrat_deseq2/contig_counts.tsv
  mkdir -p kamrat_deseq2_p05
  $T kamrat_deseq2_p05/select.time python "$PEC_CODE/scripts/select_kamrat_deseq.py" \
    "kamrat_deseq2/deseq/deseq_$DESIGN.parquet" kamrat_deseq2_p05 > kamrat_deseq2_p05/select.log
  test -s kamrat_deseq2_p05/contigs.tsv
  mv kamrat_deseq2 kamrat_deseq2_p05 "$D/"
fi
cat "$D/kamrat_deseq2_p05/summary.json"

# --- DE-kupl limma, p < 0.05, merged per direction
if [ ! -s "$D/dekupl_limma_p05/contigs.tsv" ]; then
  mkdir -p dekupl_limma_p05
  $T dekupl_limma_p05/select.time python "$PEC_CODE/scripts/select_dekupl_nominal.py" \
    "$D/dekupl_limma/DEkupl_result" "$DSG" dekupl_limma_p05 > dekupl_limma_p05/select.log
  cd dekupl_limma_p05
  for dir in up down; do
    gzip "$dir.diff-counts.tsv"
    $T "merge_$dir.time" apptainer exec --bind /scratch "$PEC_ROOT/images/dekupl-run_1.3.5.sif" \
      /dekupl/bin/mergeTags -k 31 -n "$dir.diff-counts.tsv.gz" 2> "merge_$dir.log" | gzip -c > "$dir.merged.tsv.gz"
  done
  # one table with a direction column; translate carries every column into pep.peptides.tsv
  python - <<'EOF'
import polars as pl
parts = [pl.read_csv(f"{d}.merged.tsv.gz", separator="\t").select("contig", "nb_merged_kmers", "pvalue", "log2FC")
         .with_columns(pl.lit(d).alias("direction")) for d in ("up", "down")]
t = pl.concat(parts)
bad = t.filter(((pl.col("direction") == "up") & (pl.col("log2FC") <= 0)) | ((pl.col("direction") == "down") & (pl.col("log2FC") >= 0)))
assert bad.height == 0, bad.head()
t.write_csv("contigs.tsv", separator="\t")
print("contigs", t.group_by("direction").len().sort("direction").rows())
EOF
  cd "$W"
  test -s dekupl_limma_p05/contigs.tsv
  mv dekupl_limma_p05 "$D/"
fi
cat "$D/dekupl_limma_p05/summary.json"

[ "$C" = control ] && { cd "$PEC_ROOT" && rm -rf "$W"; exit 0; }

# --- protein search of each, as the search process of main.nf
search_src() {  # <src name>
  local src=$1 out=$D/search_$1
  [ -e "$out/search.done" ] && return 0
  mkdir "$W/search_$src" && cd "$W/search_$src"
  ln -s "$D/$src/contigs.tsv" seqs.tsv
  "$PT" translate seqs.tsv pep 20 contig 2> translate.log
  cat translate.log
  if [ -s pep.peptides.faa ]; then
    seqkit split2 -s 500000 -O chunks pep.peptides.faa 2> split.log
    for fa in chunks/*.faa; do
      c=$(basename "$fa" .faa); c=${c##*.}
      sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$c" "$PWD/$fa") \
        --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c "$N" -o "$c.scaled5.zip"
      for db in proteome swissprot botznik; do
        for spec in protein:7 dayhoff:12 hp:30; do
          m=${spec%:*}; k=${spec#*:}
          for s in 10 100 5; do
            sourmash scripts manysearch "$c.scaled5.zip" "$PEC_ROOT/db/$db.$m.k$k.s$s.rocksdb" -m $m -k $k -s $s \
              -t 0.000001 -N -c "$N" -o "ms.$c.$db.$m.k$k.s$s.csv" 2>> manysearch.log
          done
        done
      done
    done
    python "$PEC_CODE/scripts/make_pairs.py" pairs.tsv hits.parquet ms.*.csv
    for db in proteome swissprot botznik; do
      python "$PEC_CODE/scripts/pair_align.py" pairs.$db.tsv pep.peptides.faa "$PEC_ROOT/db/$db.faa" align.$db.parquet "$N"
    done
  fi
  rm -f seqs.tsv
  touch search.done
  cd "$W"
  mv "search_$src" "$D/"
}
export -f search_src
export D W PT N PEC_ROOT PEC_CODE
for src in kamrat_p05 kamrat_deseq2_p05 dekupl_limma_p05; do
  if [ ! -e "$D/search_$src/search.done" ]; then
    $T "$W/search_$src.time" bash -c "set -euo pipefail; search_src $src"
    mv "$W/search_$src.time" "$D/search_$src/search.time"
  fi
  echo "$src peptides: $(grep -c '>' "$D/search_$src/pep.peptides.faa" || true)"
done
cd "$PEC_ROOT" && rm -rf "$W"
