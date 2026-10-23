#!/bin/bash
# Sequence baselines for Ced9, P66 and BHF against GENCODE v49 canonical human proteins,
# with the settings of nextflow-runs/qfo-pfam-region-benchmark (E <= 10; jackhmmer -N 3;
# MMseqs2 -s 7, --max-seqs 1000, iterative with 3 rounds) plus blastp at E <= 10.
set -euo pipefail
Q=/Users/olga/data/botryllus/alphabet-ranking-three-cases/queries.fa
H=/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa
HM=/Users/olga/anaconda3/envs/hmmer/bin; MM=/Users/olga/anaconda3/envs/orthofinder/bin
FMT='NF >= 22 {print $4 "\t" $1 "\t" $16 "\t" $17 "\t" $20 "\t" $21 "\t" $14 "\t" $13}'
$HM/phmmer --domtblout phmmer.domtbl --tblout /dev/null -o /dev/null --noali -E 10 --cpu 8 $Q $H
grep -v '^#' phmmer.domtbl | awk "$FMT" > phmmer.tsv
$HM/jackhmmer -N 3 --domtblout jackhmmer.domtbl --tblout /dev/null -o /dev/null --noali -E 10 --cpu 8 $Q $H
grep -v '^#' jackhmmer.domtbl | awk "$FMT" > jackhmmer.tsv
mkdir -p mm && $MM/mmseqs createdb $Q mm/q >/dev/null && $MM/mmseqs createdb $H mm/t >/dev/null
for it in 1 3; do
  rm -rf mm/tmp mm/res$it*; f=""; [ $it -gt 1 ] && f="--num-iterations $it"
  $MM/mmseqs search mm/q mm/t mm/res$it mm/tmp --threads 8 -s 7 $f --max-seqs 1000 -e 10 >/dev/null
  $MM/mmseqs convertalis mm/q mm/t mm/res$it mmseqs_it$it.tsv --format-output "query,target,qstart,qend,tstart,tend,bits,evalue" >/dev/null
done
/Users/olga/anaconda3/envs/orthofinder/bin/makeblastdb -in $H -dbtype prot -out mm/blastdb >/dev/null
/Users/olga/anaconda3/envs/orthofinder/bin/blastp -query $Q -db mm/blastdb -evalue 10 -num_threads 8 -max_target_seqs 1000 \
  -outfmt "6 qseqid sseqid qstart qend sstart send bitscore evalue" > blastp.tsv
for t in $HM/phmmer $HM/jackhmmer $MM/mmseqs /Users/olga/anaconda3/envs/orthofinder/bin/blastp; do echo "$t: $($t -h 2>&1 | grep -m1 -iE 'version|HMMER [0-9]|BLAST' )"; done > versions.txt 2>&1 || true
$MM/mmseqs version >> versions.txt; /Users/olga/anaconda3/envs/orthofinder/bin/blastp -version >> versions.txt
echo done
