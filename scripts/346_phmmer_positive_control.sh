#!/bin/bash
# Notebook 346 positive control: phmmer (HMMER 3.4, 20-letter scoring) on the same queries
# and databases as the HP scans, so the notebook can show whether the task is solvable at
# all with full sequence information. One query per search, about a minute each on the
# laptop (no kmerseek, no Nextflow).
#
# Writes tables/346_phmmer_<query>_vs_<database>.tbl (phmmer --max --tblout: speed filters off, every target with
# E <= 1000, so the partner's rank can be read even when it is far down the list).
#
# Run from the repository root:
#   bash scripts/346_phmmer_positive_control.sh
set -euo pipefail

PHMMER=${PHMMER:-/Users/olga/anaconda3/envs/hmmer/bin/phmmer}
OUT=tables
TMP=$(mktemp -d)
trap 'command rm -rf "$TMP"' EXIT
"$PHMMER" -h | sed -n 2p

pick() {  # pick <name in tables/346_sequences.fasta> <out fasta>
    awk -v n=">$1" '/^>/ { keep = ($1 == n) } keep' tables/346_sequences.fasta > "$2"
    [ "$(grep -c '>' "$2")" = 1 ] || { echo "346: $1 not found" >&2; exit 2; }
}

search() {  # search <query> <database file> <database slug>
    pick "$1" "$TMP/q.fa"
    db=$2
    case $db in *.gz) gzip -dc "$db" > "$TMP/db.fa"; db=$TMP/db.fa ;; esac
    "$PHMMER" --cpu 4 --max -E 1000 --domE 1000 --noali --tblout "$OUT/346_phmmer_${1}_vs_$3.tbl" "$TMP/q.fa" "$db" > /dev/null
    echo "$1 vs $3: $(grep -vc '^#' "$OUT/346_phmmer_${1}_vs_$3.tbl") targets at E <= 1000"
}

search CED-9 /Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa human_gencode_v49
search d1ohua_ /Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa human_gencode_v49
search BCL-2 /Users/olga/data/uniprot/reference_proteomes/UP000001940_6239.fasta.gz celegans_uniprot_UP000001940
search P66_mature /Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa human_gencode_v49
search CD47 /Users/olga/data/uniprot/reference_proteomes/UP000001807_224326.fasta.gz borrelia_uniprot_UP000001807
