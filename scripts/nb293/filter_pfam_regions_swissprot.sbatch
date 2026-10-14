#!/bin/bash
#SBATCH --job-name=nb293-pfam-sprot
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=%x-%j.out
# Keep the Swiss-Prot rows of Pfam 38.2 Pfam-A.regions.tsv.gz (notebook 293).
# Inputs in $NB293_DIR: swissprot_2026_03_accessions.txt, filter_pfam_regions_swissprot.py
# Expected md5 of the Pfam file, from the release's md5_checksums: f22fb1efefd1e52d6d9c64a8f10003ba
set -euo pipefail
PFAM=${PFAM_REGIONS:-$SCRATCH/data/pfam/regions/Pfam-A.regions.tsv.gz}
cd "$NB293_DIR"
md5sum "$PFAM" > pfam_regions_md5.txt
gzip -dc "$PFAM" | python3 filter_pfam_regions_swissprot.py swissprot_2026_03_accessions.txt Pfam-A.regions.swissprot_2026_03.tsv
gzip -f Pfam-A.regions.swissprot_2026_03.tsv
md5sum Pfam-A.regions.swissprot_2026_03.tsv.gz >> pfam_regions_md5.txt
