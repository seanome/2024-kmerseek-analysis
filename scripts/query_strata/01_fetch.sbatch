#!/bin/bash
#SBATCH --job-name=qs-fetch
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/olgabot/leaderboard-query-strata/logs/%x-%j.log
#
# Pull the four pinned images and download the three reference files the other jobs read.
# Ask: 4 CPUs, 32 GB, 2 h. No measured peak exists for these pulls; the pipeline's own
# prefetch-images target asks 48 GB and 2 h for one pull (nextflow-runs/
# qfo-pfam-region-benchmark/Makefile PREFETCH_MEM/PREFETCH_TIME). The DeepTMHMM image is
# 4.7 GB compressed, the largest of the four.
set -euo pipefail
source /scratch/users/olgabot/leaderboard-query-strata/repo/scripts/query_strata/paths.sh
mkdir -p "$IMAGES" "$WD/ref" "$WD/logs"
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache APPTAINER_TMPDIR=$SCRATCH/apptainer-tmp
mkdir -p "$APPTAINER_TMPDIR"

pull() {  # pull <out.sif> <ref>
  if [ -s "$1" ]; then echo "have $1"; return; fi
  apptainer pull --name "$1" "docker://$2"
}
pull "$MMSEQS_IMG" "$MMSEQS_REF"
pull "$BLAST_IMG" "$BLAST_REF"
pull "$BUSCO_IMG" "$BUSCO_REF"
pull "$DEEPTMHMM_IMG" "$DEEPTMHMM_REF"

cd "$WD/ref"
# UniProt species codes: mnemonic -> kingdom letter and NCBI taxid (release 2026_03).
[ -s speclist.txt ] || curl -fsSL -o speclist.txt \
  https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/speclist.txt
# NCBI taxonomy with each node's ranked lineage (phylum, superkingdom/domain).
[ -s new_taxdump.tar.gz ] || curl -fsSL -o new_taxdump.tar.gz \
  https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz
tar -xzf new_taxdump.tar.gz rankedlineage.dmp
# BUSCO lineage, downloaded by BUSCO itself so the version file comes with it.
apptainer exec --bind "$WD" "$BUSCO_IMG" busco --download "$BUSCO_LINEAGE" \
  --download_path "$WD/ref/busco_downloads"
sha256sum speclist.txt new_taxdump.tar.gz rankedlineage.dmp "$IMAGES"/*.sif > "$WD/ref/sha256.txt"
cat "$WD/ref/sha256.txt"
