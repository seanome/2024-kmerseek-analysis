#!/bin/bash
#SBATCH --job-name=nb361-call-types
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out
# What kind of sequence difference each SPLASH call is (target distances), where reached peptides
# sit along the protein, and how many reached genes rest only on peptides shared with other genes.
# Reads existing SPLASH, compactor and search outputs; see scripts/call_types.py.
# The all_anchors.tsv files (1-6 GB) are scanned for the anchor-count anchors and a random sample.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
export POLARS_MAX_THREADS=$SLURM_CPUS_PER_TASK
python "$PEC_CODE/scripts/call_types.py" "$PEC_ROOT/contrasts" "$PEC_ROOT/db/botznik.id_map.tsv" "$PEC_ROOT/call_types" "$PEC_CODE/control/truth_design.tsv"
