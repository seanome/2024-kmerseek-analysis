#!/bin/bash
#SBATCH --job-name=nb361-db-sketch
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
# Protein databases for the sketch search: the new Botryllus proteome (stable IDs BSP######)
# UniProtKB/Swiss-Prot 2026_03 (stable IDs SPR######), and the old gene-model proteins whose
# g##### IDs the paper's DE tables use (stable IDs BZP######). One sketch per protein
# (--singleton) at scaled 5 for each setting; searches downsample to scaled 10 and 100.
# Searches use multisearch -p (in memory), so no RocksDB index is built.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
cd "$PEC_ROOT/db"
N=$SLURM_CPUS_PER_TASK
for db in proteome swissprot botznik; do
  case $db in
    proteome)  src=botryllus_new_genome_protein.fasta; pre=BSP;;
    swissprot) src=uniprot_sprot.fasta.gz; pre=SPR;;
    botznik)   src=botznik_proteins.fa; pre=BZP;;
  esac
  [ -s $db.faa ] || python "$PEC_CODE/scripts/stable_ids.py" $src $pre $db.faa $db.id_map.tsv
  if [ ! -e $db.sketch.done ]; then
    printf 'name,genome_filename,protein_filename\n%s,,%s\n' $db $db.faa > $db.manysketch.csv
    /usr/bin/time -v -o $db.manysketch.time sourmash scripts manysketch $db.manysketch.csv --singleton \
      -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c $N -o $db.scaled5.zip
    touch $db.sketch.done
  fi
done
sourmash sig summarize proteome.scaled5.zip > proteome.scaled5.summary.txt 2>&1 || true
