#!/bin/bash
#SBATCH --job-name=nb361-ctl-rightprot
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
# The control's "right protein": for each old (botznik) transcript, its longest stop-free peptide
# is searched against the new proteome (protein k7, scaled 10) and the proposed pairs are aligned.
# Right protein = the supported pair with the smallest E-value, if E <= 1e-10 (analysis plan).
# Then each SPLASH compactor and DE-kupl contig is checked against the right protein of the gene
# its sequence comes from.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
PT=$PEC_ROOT/pec-tools/target/release/pec-tools
W=$PEC_ROOT/control_eval/right_protein
mkdir -p "$W" && cd "$W"
N=$SLURM_CPUS_PER_TASK
if [ ! -e search.done ]; then
  "$PT" translate "$PEC_CODE/control/control_transcripts.fa" tx 20 2> translate.log
  python "$PEC_CODE/scripts/longest_peptide.py" tx.peptides.tsv tx_longest.faa
  sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\ntx,,%s\n' "$PWD/tx_longest.faa") \
    --singleton -p protein,k=7,scaled=5 -c "$N" -o tx.scaled5.zip
  sourmash scripts manysearch tx.scaled5.zip "$PEC_ROOT/db/proteome.protein.k7.s10.rocksdb" -m protein -k 7 -s 10 \
    -t 0.000001 -N -c "$N" -o ms.proteome.protein.k7.s10.csv
  python "$PEC_CODE/scripts/make_pairs.py" pairs.tsv hits.parquet ms.proteome.protein.k7.s10.csv
  python "$PEC_CODE/scripts/pair_align.py" pairs.proteome.tsv tx_longest.faa "$PEC_ROOT/db/proteome.faa" align.proteome.parquet "$N"
  touch search.done
fi
python "$PEC_CODE/scripts/control_right_protein.py" "$PEC_CODE/control/control_transcripts.fa" \
  "$PEC_CODE/control/truth_design.tsv" align.proteome.parquet "$PEC_ROOT/contrasts/control" .
