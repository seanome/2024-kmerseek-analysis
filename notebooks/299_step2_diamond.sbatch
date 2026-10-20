#!/bin/bash
#SBATCH --job-name=299-diamond
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=299-diamond-%j.out
#
# Notebook 299, step 2: DIAMOND blastp of all 2_289 queries (109 real, 2_180 shuffled
# copies) against the OM-RGC v2 proteins. The shuffled copies go through DIAMOND too, so
# both methods get the same control. Submitted by 299_gate.sbatch only if step 1 says the
# catalog fits.
#
# Settings, fixed before the run: --very-sensitive, report E <= 1e-3, every target
# (--max-target-seqs 0); a true hit is E < 1e-5 (rule in notebooks/299_data/rules.md).
#
# Sizing (arithmetic from measured history, not this catalog): DIAMOND took 16.6 s for 1000
# queries against Swiss-Prot's 575_748 proteins at default sensitivity on the laptop
# (tables/239_laptop_scale_swissprot_k15_s10.txt). OM-RGC has about 82 times as many
# targets, and --very-sensitive is roughly 10-20 times slower, so a few hours on 32 threads.
# 12 h is the ask. Memory: --block-size 4 with 4 index chunks uses about 4 x 6 = 24 GB by
# DIAMOND's manual; 128 G leaves room. Read the real use with sacct afterwards.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
D=$WORK/diamond
echo "host $(hostname); $(lscpu | grep 'Model name'); $("$E/diamond" version)"

cat "$REPO/notebooks/299_data/queries_real.faa" "$REPO/notebooks/299_data/queries_shuffled.faa" > "$D/queries.faa"
if [ ! -s "$D/omrgc_v2.dmnd" ]; then
    /usr/bin/time -v -o "$D/makedb.time" \
        "$E/diamond" makedb --threads 32 --in "$WORK/catalog/OM-RGC_v2_proteins.faa" -d "$D/omrgc_v2"
fi
CMD=("$E/diamond" blastp --threads 32 --very-sensitive --evalue 1e-3 --max-target-seqs 0
     --block-size 4 --index-chunks 4 --tmpdir "${L_SCRATCH:-$D}"
     --query "$D/queries.faa" --db "$D/omrgc_v2.dmnd" --out "$D/hits.tsv"
     --outfmt 6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen)
printf '%q ' "${CMD[@]}" > "$D/blastp_command.txt"; echo >> "$D/blastp_command.txt"
/usr/bin/time -v -o "$D/blastp.time" "${CMD[@]}"
wc -l "$D/hits.tsv"
tail -n 25 "$D/blastp.time"
# Every gene DIAMOND hit (any query, E <= 1e-3), as DNA, in 32 parts: step 3 sketches only
# these to get recall exactly, since whether a pair shares a hash ignores all other genes.
cut -f2 "$D/hits.tsv" | sort -u > "$D/hit_genes.txt"
"$E/seqkit" grep -j 32 -f "$D/hit_genes.txt" "$WORK/catalog/OM-RGC_v2_genes_from_tsv.fna" > "$D/hit_genes.fna"
mkdir -p "$D/hit_gene_parts"
"$E/seqkit" split2 -j 32 -p 32 -O "$D/hit_gene_parts" "$D/hit_genes.fna"
echo "genes DIAMOND hit: $(wc -l < "$D/hit_genes.txt")"
mkdir -p "$REPO/notebooks/299_data/sherlock"
cp "$D/blastp_command.txt" "$D/blastp.time" "$REPO/notebooks/299_data/sherlock/"
