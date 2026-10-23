#!/bin/bash
# Reduce one kmerseek hit file of the cys_hp3 phylum run (run-phylum, finished 2026-10-17)
# to one row per query and target, plus the top-10 score groups and per-query counts that
# notebook 297 reads. HITS_FILE empty: write only the references and truth lists.
#
# Memory: the pilot's reduction (job 48246909) peaked at 7.8 GB with 2.8 GB as its largest
# file, so about 2.8x the file. Asked here at 4x the file, at least 32 GB:
#   k11: Mollusca 64 GB file -> 256G, Arthropoda 42 -> 192G, Cnidaria 36 -> 160G
#   k13: Mollusca 9.5 -> 40G; every other file 32G
# Time: the pilot read 6 GB in 30 s; the largest file is 11x that, asked 2 h.
# Measured (jobs 48306661-48306672, 2026-10-17): about 1.5x the file. Mollusca k11 64 GB
# peaked at 92.6 GB in 5 min 12 s, Arthropoda k11 61 GB in 3 min 22 s, Cnidaria k11 45 GB
# in 2 min 13 s; every k13 and k15 file under 15 GB and 1 min.
#
# Submit from a Sherlock login node, one job per file, e.g.:
#   sbatch --mem=256G --export=ALL,HITS_FILE=phylum.Mollusca.cys_hp3.k11.hits.csv \
#       nextflow-runs/cys-frameworks/reduce_phylum_hits.sbatch
#SBATCH --job-name=cys-reduce
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cys-reduce.%j.log

set -euo pipefail
RUN=$SCRATCH/2024-kmerseek-analysis-cys-frameworks/nextflow-runs/cys-frameworks/run-phylum
IMG=$SCRATCH/apptainer_cache/docker.io-olgabot-kmerseek-2026-10-15-cys-frameworks-c39370c.img
SCRIPT=${SCRIPT:-$RUN/reduced/reduce_cys_framework_hits.py}
mkdir -p "$RUN/reduced"

if [ -n "${HITS_FILE:-}" ]; then
    mode=(--hits-file "$HITS_FILE" --top-n 10)
else
    mode=(--references-only)
fi
/usr/bin/time -v apptainer exec --bind /scratch "$IMG" python3 "$SCRIPT" \
    --results "$RUN/results" --work "$RUN/work" --out "$RUN/reduced" "${mode[@]}"
