#!/bin/bash
#SBATCH --job-name=nb361-db-sketch
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
# Protein databases for the sketch search: the new Botryllus proteome (stable IDs BSP######)
# and UniProtKB/Swiss-Prot 2026_03 (stable IDs SPR######). One sketch per protein
# (--singleton) at scaled 5 for each alignment-free setting; RocksDB indexes at scaled 5, 10, 100.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
cd "$PEC_ROOT/db"
N=$SLURM_CPUS_PER_TASK
for db in proteome swissprot; do
  case $db in
    proteome)  src=botryllus_new_genome_protein.fasta; pre=BSP;;
    swissprot) src=uniprot_sprot.fasta.gz; pre=SPR;;
  esac
  [ -s $db.faa ] || python "$PEC_CODE/scripts/stable_ids.py" $src $pre $db.faa $db.id_map.tsv
  if [ ! -e $db.sketch.done ]; then
    printf 'name,genome_filename,protein_filename\n%s,,%s\n' $db $db.faa > $db.manysketch.csv
    /usr/bin/time -v -o $db.manysketch.time sourmash scripts manysketch $db.manysketch.csv --singleton \
      -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c $N -o $db.scaled5.zip
    touch $db.sketch.done
  fi
  for spec in protein:7 dayhoff:12 hp:30; do
    m=${spec%:*}; k=${spec#*:}
    for s in 5 10 100; do
      idx=$db.$m.k$k.s$s.rocksdb
      [ -e $idx.done ] && continue
      rm -rf $idx
      /usr/bin/time -v -o $idx.time sourmash scripts index $db.scaled5.zip -m $m -k $k -s $s -c $N -o $idx
      touch $idx.done
    done
  done
done
sourmash sig summarize proteome.scaled5.zip > proteome.scaled5.summary.txt 2>&1 || true
