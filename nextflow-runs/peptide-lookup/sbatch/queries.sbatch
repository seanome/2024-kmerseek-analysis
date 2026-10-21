#!/bin/bash
#SBATCH --job-name=nb300-queries
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# Steps 1-3 of notebook 300: spiked and real peptides, decoys, and the search database.
# Two reads of UHGP-90 (4.4 G residues) for the spike, one awk + grep pass for the hide list,
# one awk pass for the database: about 1 h on one core.
set -euo pipefail
: "${OUT:?set OUT}" "${IMG:?set IMG}" "${BIN:?set BIN}"
DB=$OUT/db
S1=$OUT/step1
mkdir -p "$S1"
export APPTAINERENV_PYTHONPATH=$OUT/pylib
py() { apptainer exec --bind /scratch "$IMG" python3 "$@"; }

# 1a. real peptides from PXD069517, minus any found in human or cRAP
py "$BIN/real_peptides.py" select "$S1/PXD069517/MGDB_peptides_intensities_tax-funct_annotations.xlsx" \
  "$DB/human_UP000005640_9606.fasta.gz" "$DB/human_UP000005640_9606_additional.fasta.gz" \
  --crap "$DB/crap.fasta" --out-dir "$S1"

# 1b. 1,000 spiked peptides from UHGP-90, lengths drawn from the real peptides
py "$BIN/spike_peptides.py" "$DB/uhgp-90.faa.gz" "$S1/spiked_peptides.tsv" \
  --length-pool "$S1/real_peptide_pool.tsv"

# UHGP-90 proteins holding any pool peptide, I and L read as one letter
gzip -dc "$DB/uhgp-90.faa.gz" \
  | awk 'BEGIN{RS=">"; ORS=""} NR>1 {n=index($0,"\n"); id=substr($0,1,n-1); split(id,a," ");
         seq=substr($0,n+1); gsub(/\n/,"",seq); gsub(/I/,"L",seq); print a[1] "\t" seq "\n"}' \
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
printf 'uhgp90_kept\t%s\n' "$(gzip -dc "$DB/uhgp-90.faa.gz" | grep -c '^>' | awk -v h="$(wc -l < "$S1/uhgp90_hidden_proteins.txt")" '{print $1-h}')" >> "$S1/search_db_counts.tsv"
printf 'uhgp90_hidden\t%s\n' "$(wc -l < "$S1/uhgp90_hidden_proteins.txt")" >> "$S1/search_db_counts.tsv"
printf 'human_with_isoforms\t%s\n' "$(gzip -dc "$DB/human_UP000005640_9606.fasta.gz" "$DB/human_UP000005640_9606_additional.fasta.gz" | grep -c '^>')" >> "$S1/search_db_counts.tsv"
printf 'crap\t%s\n' "$(grep -c '^>' "$DB/crap.fasta")" >> "$S1/search_db_counts.tsv"
printf 'total\t%s\n' "$(grep -c '^>' "$S1/search_db.fasta")" >> "$S1/search_db_counts.tsv"
cat "$S1/search_db_counts.tsv"
