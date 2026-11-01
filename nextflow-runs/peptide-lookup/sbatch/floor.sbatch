#!/bin/bash
#SBATCH --job-name=nb300-floor
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Step 0 table: the E-value floor of one exact window, every setting x database x length.
set -euo pipefail
: "${OUT:?set OUT}" "${IMG:?set IMG}" "${BIN:?set BIN}" "${KA_STORE:?set KA_STORE}"
cd "$BIN"
apptainer exec --bind /scratch "$IMG" python3 collect_ka_fits.py "$KA_STORE" "$OUT/step0/ka_fits.tsv"
apptainer exec --bind /scratch "$IMG" python3 evalue_floor.py \
  --swissprot "$OUT/db/uniprot_sprot.fasta.gz" \
  --ka-fits "$OUT/step0/ka_fits.tsv" \
  --composition "swissprot=$OUT/step0/composition_swissprot.json" \
  --composition "uniref90=$OUT/step0/composition_uniref90.json" \
  --composition "uhgp90=$OUT/step0/composition_uhgp90.json" \
  --out-dir "$OUT/step0"
