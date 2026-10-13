#!/bin/bash
#SBATCH --job-name=282-orthofinder
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=282-orthofinder-%j.out
#
# Notebook 282, Stage 1: gene families for Botryllus, Ciona, amphioxus (Branchiostoma
# lanceolatum) and human with OrthoFinder 3.1.5, one protein per gene (written by
# scripts/prepare_282_proteomes.py on the laptop, 54 MB).
#
# First time only, from the laptop (data goes through the data transfer node):
#
#   rsync -av /Users/olga/data/282-ohnolog-families/proteomes/ sherlock-dtn:/scratch/users/olgabot/282-ohnolog-families/proteomes/
#
# On Sherlock, from a checkout of this branch:
#
#   git clone --branch olgabot/ohnolog-specialization-282 https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/282-ohnolog-families-repo
#   cd $SCRATCH/282-ohnolog-families-repo && sbatch notebooks/282_orthofinder.sbatch
#
# Sizing. No run of this exact input exists yet. The closest one: OrthoFinder 3.1.0 on the
# laptop (16 cores) grouped 318,152 proteins of two species (Botryllus with isoforms, all
# GENCODE v49 isoforms) in 1 h 43 min, 2026-03-03, Log.txt in
# ~/data/botryllus/data-for-orthofinder/OrthoFinder/Results_Mar03_1. This input is
# 115,111 proteins in four species. The all-against-all DIAMOND search grows with the
# square of the protein count, so about (115/318)^2 = 0.13 of that search, but four
# species add gene trees for more families. 12 h covers Sherlock cores up to 5x slower than
# the laptop. Peak memory was never measured; 64 GB is a guess with room. After the run,
# read `sacct -j <jobid> -o JobID,MaxRSS,Elapsed` and shrink both.
#
# The notebook needs only the family tables. Copy them back with:
#
#   rsync -av sherlock-dtn:/scratch/users/olgabot/282-ohnolog-families/orthofinder/Results_282/{Orthogroups,Phylogenetic_Hierarchical_Orthogroups,Species_Tree,Log.txt,Citation.txt} \
#       /Users/olga/data/282-ohnolog-families/orthofinder/Results_282/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
DIR=${NB282_DIR:-$SCRATCH/282-ohnolog-families}
IN=$DIR/proteomes
OUT=$DIR/orthofinder
IMG=$SCRATCH/apptainer-cache/quay.io-biocontainers-orthofinder-3.1.5--hdfd78af_0.sif
TREE=$REPO/notebooks/282_species_tree.nwk

for f in "$IN/botryllus.fa" "$IN/human.fa" "$IN/amphioxus.fa" "$IN/ciona.fa" "$TREE"; do
    [ -s "$f" ] || { echo "282: missing input $f" >&2; exit 2; }
done
# OrthoFinder refuses to write into a results folder that already exists.
[ ! -e "$OUT/Results_282" ] || { echo "282: $OUT/Results_282 exists; move it first" >&2; exit 2; }

mkdir -p "$(dirname "$IMG")" "$OUT"
[ -s "$IMG" ] || apptainer pull "$IMG" docker://quay.io/biocontainers/orthofinder:3.1.5--hdfd78af_0

# OrthoFinder reads every FASTA in -f; give it a folder with only the four proteomes.
FA=$DIR/orthofinder-input
mkdir -p "$FA"
for s in botryllus human amphioxus ciona; do ln -sf "$IN/$s.fa" "$FA/$s.fa"; done

# -s: the species tree is fixed, not inferred (four species are too few to root it
# reliably). It roots the families at the chordate ancestor, with amphioxus outside
# tunicates + vertebrates. -S diamond is the default search (DIAMOND --more-sensitive).
apptainer exec --bind "$DIR","$REPO" "$IMG" \
    orthofinder -f "$FA" -s "$TREE" -t "$SLURM_CPUS_PER_TASK" -a 4 -S diamond \
    -o "$OUT" -n 282

# OrthoFinder 3.1 writes the families rooted at N0 (the chordate ancestor) to
# Orthogroups/Orthogroups.tsv and leaves no N0.tsv (a 4-species test run, 2026-10-12).
# N1.tsv holds the families cut at the tunicate + vertebrate ancestor.
ls -la "$OUT/Results_282/Orthogroups/Orthogroups.tsv" "$OUT/Results_282/Phylogenetic_Hierarchical_Orthogroups/N1.tsv"
