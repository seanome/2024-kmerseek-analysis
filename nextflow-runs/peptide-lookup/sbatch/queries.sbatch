#!/bin/bash
#SBATCH --job-name=nb300-queries
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# Steps 1-3 of notebook 300: spiked and real peptides, decoys, and the search database.
# Measured 2026-10-21: 49 min, of which the human/cRAP check in select took 36.
# The select (36 min, 2026-10-21) and spike steps are skipped when their outputs exist; delete
# real_peptide_pool.tsv or spiked_peptides.tsv to redo them.
set -euo pipefail
: "${OUT:?set OUT}" "${IMG:?set IMG}" "${BIN:?set BIN}"
DB=$OUT/db
S1=$OUT/step1
mkdir -p "$S1"
export APPTAINERENV_PYTHONPATH=$OUT/pylib
py() { apptainer exec --bind /scratch "$IMG" python3 "$@"; }

# 1a. real peptides from PXD069517, minus any found in human or cRAP
[ -s "$S1/real_peptide_pool.tsv" ] || py "$BIN/real_peptides.py" select "$S1/PXD069517/MGDB_peptides_intensities_tax-funct_annotations.xlsx" \
  "$DB/human_UP000005640_9606.fasta.gz" "$DB/human_UP000005640_9606_additional.fasta.gz" \
  --crap "$DB/crap.fasta" --out-dir "$S1"

# 1b. 1,000 spiked peptides from UHGP-90, lengths drawn from the real peptides
[ -s "$S1/spiked_peptides.tsv" ] || py "$BIN/spike_peptides.py" "$DB/uhgp-90.faa.gz" "$S1/spiked_peptides.tsv" \
  --length-pool "$S1/real_peptide_pool.tsv"

# UHGP-90 proteins holding any pool peptide, I and L read as one letter. Records are split on
# header lines, not on ">": some descriptions hold one, as in "(1->5)-L-arabinanase".
gzip -dc "$DB/uhgp-90.faa.gz" \
  | awk '/^>/ {if (id != "") {gsub(/I/,"L",seq); print id "\t" seq}; split(substr($0,2),a," ");
               id=a[1]; seq=""; next}
         {seq = seq $0}
         END {gsub(/I/,"L",seq); print id "\t" seq}' \
  | { grep -F -f "$S1/real_peptide_patterns_il.txt" || [ $? -eq 1 ]; } > "$S1/uhgp90_lines_with_pool_peptide.tsv"
py "$BIN/real_peptides.py" hide "$S1/uhgp90_lines_with_pool_peptide.tsv" "$S1/spiked_peptides.tsv" \
  --out-dir "$S1"

# 2. decoys
py "$BIN/make_queries.py" "$S1/real_peptides.tsv" "$S1/spiked_peptides.tsv" "$S1"

# 3. search database: UHGP-90 minus the hidden proteins, human with isoforms, cRAP
gzip -dc "$DB/uhgp-90.faa.gz" \
  | awk 'NR==FNR {h[$1]; next} /^>/ {split(substr($0,2),a," "); skip=(a[1] in h)} !skip' \
      "$S1/uhgp90_hidden_proteins.txt" - > "$S1/search_db.fasta"
gzip -dc "$DB/human_UP000005640_9606.fasta.gz" "$DB/human_UP000005640_9606_additional.fasta.gz" >> "$S1/search_db.fasta"
cat "$DB/crap.fasta" >> "$S1/search_db.fasta"
printf 'part\tn_sequences\n' > "$S1/search_db_counts.tsv"
printf 'uhgp90_kept\t%s\n' "$(gzip -dc "$DB/uhgp-90.faa.gz" | grep -c '^>' | awk -v h="$(grep -c . "$S1/uhgp90_hidden_proteins.txt" || true)" '{print $1-h}')" >> "$S1/search_db_counts.tsv"
printf 'uhgp90_hidden\t%s\n' "$(grep -c . "$S1/uhgp90_hidden_proteins.txt" || true)" >> "$S1/search_db_counts.tsv"
printf 'human_with_isoforms\t%s\n' "$(gzip -dc "$DB/human_UP000005640_9606.fasta.gz" "$DB/human_UP000005640_9606_additional.fasta.gz" | grep -c '^>')" >> "$S1/search_db_counts.tsv"
printf 'crap\t%s\n' "$(grep -c '^>' "$DB/crap.fasta")" >> "$S1/search_db_counts.tsv"
printf 'total\t%s\n' "$(grep -c '^>' "$S1/search_db.fasta")" >> "$S1/search_db_counts.tsv"
cat "$S1/search_db_counts.tsv"

# The database must hold exactly the UHGP-90 proteins not hidden: every hidden id is a real
# accession, and none is left in.
n_kept=$(awk '$1=="uhgp90_kept"{print $2}' "$S1/search_db_counts.tsv")
n_db=$(grep -c '^>MGYG' "$S1/search_db.fasta")
if [ "$n_kept" != "$n_db" ] || grep -qv -e '^MGYG' -e '^$' "$S1/uhgp90_hidden_proteins.txt"; then
  echo "database check failed: $n_db UHGP-90 proteins in the database, expected $n_kept" >&2
  exit 1
fi
echo "database check passed: $n_db UHGP-90 proteins"
