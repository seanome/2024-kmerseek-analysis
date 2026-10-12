#!/bin/bash
#SBATCH --job-name=280-download
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.log
# Compute nodes have outbound internet; the dtn accepts only scp/rsync.
bash "${TIME_SPLIT_BIN:-$SLURM_SUBMIT_DIR/bin}/download.sh"
