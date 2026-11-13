#!/bin/bash
#SBATCH --job-name=nb361-kamrat-nominal
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=%x.%A_%a.out
# KaMRaT contigs at uncorrected p <= 0.05 and 2-fold, then the protein search, one contrast per
# array task. The same commands as the kamrat_nominal and search processes of
# nextflow-runs/pec-domen2026/main.nf, writing to the same storeDir folders
# ($PEC_ROOT/contrasts/<c>/kamrat_nominal and search_kamrat_nominal), so a later pipeline run
# finds them. Run as a standalone job (2026-11-02) so only these two steps run.
# Resources: the search of young_24h's 3.19 M compactor peptides needed 128 GB (64 GB was
# killed, exit 137) and took 58 min with 16 CPUs.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
CONTRASTS=(young_2h young_24h old_24h young_13_49d young_132d old_41_93d)
C=${CONTRASTS[$SLURM_ARRAY_TASK_ID]}
D=$PEC_ROOT/contrasts/$C
export PATH=$PEC_ROOT/env/bin:$PATH
PT=$PEC_ROOT/pec-tools/target/release/pec-tools
N=$SLURM_CPUS_PER_TASK
W=$(mktemp -d "$PEC_ROOT/tmp_kamrat_nominal_${C}_XXXX")
cd "$W"

# --- kamrat_nominal
if [ ! -s "$D/kamrat_nominal/contigs.tsv" ]; then
  python "$PEC_CODE/scripts/select_kamrat_nominal.py" "$D/kamrat_ttest/contigs_all.tsv.gz" \
    "$PEC_CODE/design/contrast_$C.design.tsv" kamrat_nominal > kamrat_nominal.log
  mv kamrat_nominal.log kamrat_nominal/
  test -s kamrat_nominal/contigs.tsv
  mv kamrat_nominal "$D/"
fi
cat "$D/kamrat_nominal/summary.json"

# --- search, source kamrat_nominal
if [ ! -e "$D/search_kamrat_nominal/search.done" ]; then
  mkdir search_kamrat_nominal && cd search_kamrat_nominal
  ln -s "$D/kamrat_nominal/contigs.tsv" seqs.tsv; COL=contig
  "$PT" translate seqs.tsv pep 20 $COL 2> translate.log
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
  mv search_kamrat_nominal "$D/"
fi
echo "peptides: $(grep -c '>' "$D/search_kamrat_nominal/pep.peptides.faa" || true)"
cd "$PEC_ROOT" && rm -rf "$W"
