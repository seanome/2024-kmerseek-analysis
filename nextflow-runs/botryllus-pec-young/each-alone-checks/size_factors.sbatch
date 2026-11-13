#!/bin/bash
#SBATCH --job-name=pec-size-factors
# Sized from job 51911916: 36 fits took 12 min 48 s and 2.1 GB on 1 CPU.
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=size_factors.%j.log
set -euo pipefail
cd /scratch/users/olgabot/botryllus-pec-young/size-factors
IMG=/scratch/users/olgabot/apptainer-cache/quay.io-biocontainers-bioconductor-deseq2@sha256-2d299c5f045c65879e8da189bba17857963d50703af5410003e7e84883deef13.img
apptainer exec --bind /scratch "$IMG" Rscript size_factors.R > size_factors.out 2>&1
