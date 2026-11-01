#!/bin/bash
# Inputs for notebook 279's BHF section, made on Sherlock on 2026-10-12. Run there, then copy
# $OUT to /Users/olga/data/invertebrate-dark-set/bhf-view/ through sherlock-dtn.
#
# BHF (Botryllus histocompatibility factor) is FUN008084_FUN008084 in the staged Botryllus
# proteome. Query chunks hold 2_000 proteins in file order; BHF is protein 8_256 (0-based),
# so chunk_0004.
set -euo pipefail
OUT=${OUT:-$SCRATCH/bhf-view}
RESULTS=/scratch/users/olgabot/invertebrate-dark-set/results-0.4/botryllus/kmerseek
HITS=/scratch/users/olgabot/invertebrate-dark-set/results/botryllus/hits
QUERY=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/qfo/Eukaryota/BOTSCH2026_30301.fasta
REF=/scratch/users/olgabot/invertebrate-dark-set/reference/minus_Ascidiacea/reference.fasta
IMG=$SCRATCH/apptainer-cache/quay.io-biocontainers-hmmer@sha256-7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c.img
mkdir -p "$OUT"

# BHF's sequence
awk '/^>/{p=($1==">FUN008084_FUN008084")} p' "$QUERY" > "$OUT/bhf.fasta"

# Every kmerseek region on BHF under hp_lehninger2 k17, scaled 1, mask off, extension at C_opt
f=$(ls "$RESULTS"/chunk_0004.hp_lehninger2.k17.s1.lcfalse.*.regions.csv.zst)
srun -p hns,normal -A ayeletv -c 2 --mem 4G -t 30 bash -c \
    "zstdcat $f | awk -F, 'NR==1 || \$1==\"FUN008084_FUN008084\"' > $OUT/bhf_hp_lehninger2_k17_s1.csv"

# The run's stored phmmer, jackhmmer and mmseqs2 hits for BHF (target coordinates only)
for t in phmmer jackhmmer mmseqs2; do
    zcat "$HITS/botryllus.chunk_0004.$t.tsv.gz" \
        | awk -v t=$t -v OFS='\t' '$1=="FUN008084_FUN008084"{print t, $0}'
done > "$OUT/bhf_tool_hits.tsv"

# phmmer again for BHF alone, with the run's image, reference and -E 10, to get each hit's
# position on BHF (domtblout columns 16-17) and the alignments
srun -p hns,normal -A ayeletv -c 8 --mem 8G -t 60 apptainer exec --bind /scratch "$IMG" \
    phmmer --domtblout "$OUT/bhf_phmmer.domtbl" -o "$OUT/bhf_phmmer.out" -E 10 --cpu 8 "$OUT/bhf.fasta" "$REF"

# Swiss-Prot names of the tool targets and the 200 kmerseek targets with the smallest E-value
# (sprot_descriptions from nextflow-runs/invertebrate-dark-set/tools/dark-regions, PR 57)
