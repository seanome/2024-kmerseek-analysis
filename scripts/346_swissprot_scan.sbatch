#!/bin/bash
#SBATCH --job-name=346-swissprot-scan
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --array=0-15
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/olgabot/346_linked_pairs/logs/%x-%A_%a.out
#
# Notebook 346, pairs 3 to 6: each pair's query (a Pfam seed domain) against every
# Swiss-Prot protein, a uniform shuffle of it and a dipeptide-preserving shuffle of it,
# scored with B2, the chosen linked HP word pair, its randstrobe and B3 on every diagonal
# (scripts/346_database_scan.py, linked_pairs_utils.diagonal_scan). Python and numpy only;
# no kmerseek. Sixteen array tasks each take every 16th protein; a last step joins them.
#
# Why Sherlock: on the 16-core laptop a 1,200-protein sample projected 232 minutes for the
# whole of Swiss-Prot (805 us of worker time per Swiss-Prot residue, 207.6 million
# residues, so about 46 core-hours), over the notebook's 2-hour laptop limit.
#
# Sizing (no Sherlock history for this script): 46 core-hours / 16 tasks = 2.9 core-hours
# per task, about 11 minutes on 16 cores if a Sherlock core runs as fast as a laptop core.
# 3 hours covers a core 10 times slower than that and the largest proteins arriving last.
# Memory: one protein pair of a 272-residue query against titin (35,991 residues) peaks
# near 2 GB per worker (the int32 and int64 score matrices of diagonal_scan), so 16
# workers fit in 32 GB; 64 GB is twice that. Afterwards:
#   sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State
#
# Inputs: this repository on the branch olgabot/linked-pairs-known-pairs (git pull), the
# image below, and the same Swiss-Prot FASTA the laptop runs used, copied to
# $SCRATCH/346_linked_pairs/inputs/uniprot_sprot.fasta.gz. The script refuses a file whose
# protein count differs from tables/346_databases.tsv (572,970).
#
# Submit with `make run` in nextflow-runs/linked-pairs-known-pairs (on Sherlock): it submits
# the 16 tasks and a merge task that waits for them (NB346_MERGE=1, --array=0).
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_ARRAY_TASK_ID:?}"
D=${NB346_SHERLOCK_DIR:-$SCRATCH/346_linked_pairs}
IMG=${NB346_IMG:-$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img}
SPROT=$D/inputs/uniprot_sprot.fasta.gz
REPO=${SLURM_SUBMIT_DIR:?}
NPARTS=16
mkdir -p "$D/logs" "$D/results"
for f in "$IMG" "$SPROT" "$REPO/scripts/346_database_scan.py" "$REPO/tables/346_databases.tsv"; do
    [ -s "$f" ] || { echo "346: missing input $f" >&2; exit 2; }
done
py() { apptainer exec --bind /scratch "$IMG" env NB346_DIR="$D/results" OMP_NUM_THREADS=1 python3 "$@"; }
py -c "import numpy, polars; print('numpy', numpy.__version__, 'polars', polars.__version__)"

cd "$REPO"
if [ "${NB346_MERGE:-0}" = 1 ]; then
    py scripts/346_database_scan.py --only Swiss-Prot --merge
    echo "346: copy back with: rsync -av sherlock-dtn:$D/results/346_scan_swissprot.parquet ~/data/linked-pairs/"
    exit 0
fi
echo "346 task $SLURM_ARRAY_TASK_ID of $NPARTS on $(hostname), commit $(git rev-parse --short HEAD)"
py scripts/346_database_scan.py --only Swiss-Prot --db-file "$SPROT" \
    --part "$SLURM_ARRAY_TASK_ID" --nparts "$NPARTS" --workers "${SLURM_CPUS_PER_TASK:-16}"
echo "346 task $SLURM_ARRAY_TASK_ID done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
