#!/bin/bash
# relabel_for_mmseqs.sh PROTEOME FASTA [PROTEOME FASTA ...] > clade.fasta
#
# Writes every protein as '>PROTEOME__ACCESSION', so an MMseqs2 cluster member names its
# proteome. No '|' in the label: MMseqs2 cuts 'sp|ACC|NAME' down to ACC (memory note
# mmseqs_uniprot_header_accession_trap), which would drop the proteome.
set -euo pipefail
while [ $# -gt 0 ]; do
    awk -v p="$1" '/^>/ { split(substr($0, 2), a, " "); n = split(a[1], b, "|");
                          print ">" p "__" (n >= 2 ? b[2] : a[1]); next } { print }' "$2"
    shift 2
done
