#!/bin/bash
# Notebook 359, step 1: SQANTI3 QC with ORF prediction and IsoAnnotLite (the competitor's
# own pipeline, with tappAS's GENCODE v39 GFF3), then the default rules filter, then the
# ORF FASTA InterProScan reads.
#
# No measured history for SQANTI3 on this cluster. The ask is a first estimate: 16 CPUs,
# 128 GB, 24 h for the union of three TALON replicates (expect 1-3 x 10^5 transcripts).
# The log records the elapsed time and peak, to size any rerun from data.
#SBATCH --job-name=359-sqanti
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.log

set -euo pipefail
WORK=$(pwd)
D=$WORK/data
SQ=/scratch/users/olgabot/apptainer-cache/docker.io-anaconesalab-sqanti3-v6.0.1.img
GFF3=$(find "$D/tappas" -name '*.gff3' | head -1)
mkdir -p "$WORK/sqanti/qc" "$WORK/sqanti/filter"
run() { apptainer exec --bind /scratch "$SQ" /conda/miniconda3/bin/conda run --no-capture-output -n sqanti3 "$@"; }
SQDIR=$(apptainer exec "$SQ" bash -c 'dirname /opt2/sqanti3/*/SQANTI3-*/sqanti3_qc.py')

run python "$SQDIR/sqanti3_qc.py" \
    --isoforms "$D/wtc11_lrgasp_pacbio_union.gtf" \
    --refGTF "$D/gencode.v39.primary_assembly.annotation.gtf" \
    --refFasta "$D/GRCh38.primary_assembly.genome.fa" \
    --include_ORF --isoAnnotLite --gff3 "$GFF3" \
    -t 16 -n 16 --report skip \
    -d "$WORK/sqanti/qc" -o wtc11

run python "$SQDIR/sqanti3_filter.py" rules \
    --sqanti_class "$WORK/sqanti/qc/wtc11_classification.txt" \
    --isoAnnotGFF3 "$WORK/sqanti/qc/wtc11.gff3" \
    --filter_gtf "$WORK/sqanti/qc/wtc11_corrected.gtf" \
    --filter_faa "$WORK/sqanti/qc/wtc11_corrected.faa" \
    -c 16 --skip_report \
    -d "$WORK/sqanti/filter" -o wtc11

python3 code/scripts/359_isoform_blank_exons/write_orfs.py \
    "$WORK/sqanti/qc/wtc11_classification.txt" \
    "$WORK/sqanti/filter/wtc11_pass_isoforms.txt" \
    "$WORK/sqanti/orfs_pass_coding.fasta"
run python "$SQDIR/sqanti3_qc.py" --version || true
