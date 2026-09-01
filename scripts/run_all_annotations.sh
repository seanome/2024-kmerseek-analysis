#!/usr/bin/env bash
# run_all_annotations.sh
#
# Run the full annotation + pair construction pipeline for all QfO species.
# Steps:
#   1. build_pfam_architectures.py  — fetch Pfam from UniProt, save per-species parquet
#   2. construct_subdomain_pairs.py — build human-vs-X ground truth pairs
#
# Usage (from anywhere):
#   bash scripts/run_all_annotations.sh
#   bash scripts/run_all_annotations.sh --force   # rerun everything
#
# The 2025-kmerseek-analysis conda environment must be active, or this script
# will activate it via the absolute Python path.

set -euo pipefail

# The two Python steps live in notebooks/ next to the notebooks that drive them
# (125), while their inputs and outputs stay under results/pfam_benchmark/ —
# so resolve everything from the repo root rather than from this script's dir.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK_DIR="${REPO_ROOT}/notebooks"
BENCHMARK_DIR="${REPO_ROOT}/results/pfam_benchmark"
PYTHON="/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3"

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
    echo "=== --force mode: reprocessing all species ==="
fi

echo "=== Step 1: Fetch Pfam annotations for all species ==="
echo "Output: ${BENCHMARK_DIR}/annotations/"
echo ""

"$PYTHON" "${NOTEBOOK_DIR}/build_pfam_architectures.py" \
    --species all \
    --outdir  "${BENCHMARK_DIR}/annotations" \
    --qfo-dir "${HOME}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143" \
    $FORCE_FLAG

echo ""
echo "=== Step 2: Construct human-vs-species benchmark pairs ==="
echo "Output: ${BENCHMARK_DIR}/pairs/"
echo ""

"$PYTHON" "${NOTEBOOK_DIR}/construct_subdomain_pairs.py" \
    --species all \
    --annot-dir "${BENCHMARK_DIR}/annotations" \
    --pairs-dir "${BENCHMARK_DIR}/pairs" \
    $FORCE_FLAG

echo ""
echo "=== Done! ==="
echo ""
echo "Annotation files:"
ls -lh "${BENCHMARK_DIR}/annotations/"*.parquet 2>/dev/null || echo "  (none yet)"

echo ""
echo "Pair files:"
ls -lh "${BENCHMARK_DIR}/pairs/"*.parquet 2>/dev/null || echo "  (none yet)"
