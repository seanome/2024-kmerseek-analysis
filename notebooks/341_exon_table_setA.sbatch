#!/bin/bash
#SBATCH --job-name=341-exon-table-A
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=341-exon-table-A-%j.out
#
# Notebook 341, step 1 for set A (Patowary et al. 2024): every isoform pair that differs by one
# internal exon, with both proteins (longest ORF of each isoform) and Ψ per donor in the
# germinal zone and cortical plate. Runs scripts/build_341_exon_table.py.
#
# Inputs, under $SCRATCH/condensate_exons_341/:
#   patowary2024_dryad/DataS1.tar.gz, DataS2.xlsx.zip   copied from the laptop (see below)
#   grch37/GRCh37.primary_assembly.genome.fa.gz         downloaded by this job the first time,
#                                                        GENCODE release 33 GRCh37 mapping, md5 checked
#
# Copy the Dryad files from the laptop first (140 MB):
#
#   rsync -av /Users/olga/data/condensate_exons_341/patowary2024_dryad/DataS1.tar.gz /Users/olga/data/condensate_exons_341/patowary2024_dryad/DataS2.xlsx.zip sherlock-dtn:/scratch/users/olgabot/condensate_exons_341/patowary2024_dryad/
#
# Then, on Sherlock:
#
#   cd $SCRATCH/2024-kmerseek-analysis-341 && git pull && sbatch notebooks/341_exon_table_setA.sbatch
#
# Python: Sherlock's python/3.12.1 module with openpyxl 3.1.5 (the laptop env's version) in a
# venv under $SCRATCH/venvs/nb341, made the first time and checked by importing it.
#
# Sizing: no earlier run of this script. Tested on the laptop on ZNF207 (31 isoforms, 10 pairs,
# under a second). The whole set is 214,516 isoforms: reading the 1.3 GB GTF, the 870 MB
# compressed genome one chromosome at a time (chr1 is about 250 MB as text), and the 63 MB
# workbook. Memory is set by the largest chromosome plus the isoform tables, a few GB; 16 G
# leaves room. Time is mostly the pure-Python ORF scan, roughly 30-60 min; 4 h covers a slow
# node. Read the real numbers after the run with `sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State`.
#
# Copy the event table back to the laptop worktree (a few MB):
#
#   rsync -av sherlock-dtn:/scratch/users/olgabot/condensate_exons_341/setA_exon_table/341_setA_exon_events.tsv /Users/olga/code/2024-kmerseek-analysis-341/data/341_condensate_exons/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
cd "$SLURM_SUBMIT_DIR"
BASE=${CONDENSATE_341_DATA:-$SCRATCH/condensate_exons_341}
DRYAD=$BASE/patowary2024_dryad
GENOME_DIR=$BASE/grch37
OUT=$BASE/setA_exon_table
GENOME_URL=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_33/GRCh37_mapping/GRCh37.primary_assembly.genome.fa.gz
GENOME_MD5=c9f86b6083dd05b9799eca02c79f651d
mkdir -p "$GENOME_DIR" "$OUT"

# The two Dryad files, checked against Dryad's SHA-256 checksums.
(cd "$DRYAD" && sha256sum -c - <<'SUMS'
7bffff510a453a2bff7f6683bdad0140b3e5ca4e31ecda5b0ecefad498807eab  DataS1.tar.gz
43230bda4631add5e4257c70ef239ad6f1e57598ac8e6d042dcb5de8ccecf19f  DataS2.xlsx.zip
SUMS
)
[[ -f "$DRYAD/Bulk.gtf" ]] || tar xzf "$DRYAD/DataS1.tar.gz" -C "$DRYAD" Bulk.gtf
[[ -f "$DRYAD/DataS2.xlsx" ]] || unzip -o -q "$DRYAD/DataS2.xlsx.zip" DataS2.xlsx -d "$DRYAD"

GENOME=$GENOME_DIR/GRCh37.primary_assembly.genome.fa.gz
if [[ ! -f "$GENOME.md5ok" ]]; then
    for attempt in 1 2 3 4 5; do
        set +e
        curl -fsSL -C - --retry 5 --retry-delay 30 --speed-limit 100000 --speed-time 120 -o "$GENOME" "$GENOME_URL"
        status=$?
        set -e
        [[ $status -eq 0 ]] && break
        echo "genome download: curl exit $status, attempt $attempt"
        sleep 60
    done
    [[ "$(md5sum "$GENOME" | cut -d' ' -f1)" == "$GENOME_MD5" ]] || { rm -f "$GENOME"; echo "genome md5 mismatch; deleted, resubmit"; exit 1; }
    touch "$GENOME.md5ok"
fi

module load python/3.12.1
VENV=$SCRATCH/venvs/nb341
if ! "$VENV/bin/python" -c 'import openpyxl' 2>/dev/null; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet --only-binary=:all: openpyxl==3.1.5
    "$VENV/bin/python" -c 'import openpyxl; print("openpyxl", openpyxl.__version__)'
fi

"$VENV/bin/python" scripts/build_341_exon_table.py \
    --gtf "$DRYAD/Bulk.gtf" \
    --counts "$DRYAD/DataS2.xlsx" \
    --genome "$GENOME" \
    --out-dir "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
ls -l "$OUT"
