#!/bin/bash
#SBATCH --job-name=nb361-search
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --output=%x.%j.out
# Translate one sequence source of a contrast (compactors or DE-kupl contigs), sketch the
# peptides, search the new proteome and Swiss-Prot at every setting (multisearch -p), then align
# each proposed pair (parasail). SRC=compactors|dekupl. Resources are set at submission.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}" "${CONTRAST:?}" "${SRC:?}"
D=$PEC_ROOT/contrasts/$CONTRAST
cd "$D"
export PATH=$PEC_ROOT/env/bin:$PATH
PT=$PEC_ROOT/pec-tools/target/release/pec-tools
N=$SLURM_CPUS_PER_TASK
mkdir -p search_$SRC && cd search_$SRC
[ -e search.done ] && { echo "search $SRC already done"; exit 0; }
case $SRC in
  compactors) IN=../compactors/compactors.tsv; COL=${SEQ_COL:-compactor};;
  dekupl) zcat ../dekupl/DEkupl_result/control_vs_PEC_kmer_counts/merged-diff-counts.tsv.gz > dekupl_contigs.tsv
          IN=dekupl_contigs.tsv; COL=contig;;
esac
"$PT" translate "$IN" pep 20 "$COL" 2> translate.log
cat translate.log
if [ -s pep.peptides.faa ]; then
  sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\npep,,%s\n' "$PWD/pep.peptides.faa") \
    --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c "$N" -o pep.scaled5.zip
  for db in proteome swissprot botznik; do
    for spec in protein:7 dayhoff:12 hp:30; do
      m=${spec%:*}; k=${spec#*:}
      for s in 10 100 5; do
        out=ms.$db.$m.k$k.s$s.csv
        [ -s "$out" ] && continue
        sourmash scripts multisearch pep.scaled5.zip "$PEC_ROOT/db/$db.scaled5.zip" -m $m -k $k -s $s \
          -t 0.000001 -p -c "$N" -o "$out.part" 2>> multisearch.log
        mv "$out.part" "$out"
      done
    done
  done
  python "$PEC_CODE/scripts/make_pairs.py" pairs.tsv hits.parquet ms.*.csv
  for db in proteome swissprot botznik; do
    python "$PEC_CODE/scripts/pair_align.py" pairs.$db.tsv pep.peptides.faa "$PEC_ROOT/db/$db.faa" \
      align.$db.parquet "$N"
  done
fi
touch search.done
