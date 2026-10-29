#!/bin/bash
#SBATCH --job-name=301-step2-queries
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=301-step2-%j.out
#
# Notebook 301, step 2: download Swiss-Prot, build the 2,000 real queries and 500
# chimeras, look up every query's UniRef90 cluster, and write Swiss-Prot without those
# clusters (the step 4 database). Standard-library Python only.
#
# Submit from this branch's own clone on Sherlock:
#
#   git clone --branch olgabot/nb301-short-feature-boundaries https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/2024-kmerseek-analysis-301
#   cd $SCRATCH/2024-kmerseek-analysis-301 && git pull && sbatch scripts/301_step2.sbatch
#
# Sizing (no history for this job): parsing all of Swiss-Prot keeps about 573,000 entries
# in memory, a few GB in Python; 16 GB leaves room. Time: the download is about 650 MB,
# the parse a few minutes, and the UniRef90 lookups about 2,500 calls to rest.uniprot.org
# at under a second each. 3 h covers a slow server. After the run, read the real use with
# `sacct -j <jobid> -o JobID,MaxRSS,Elapsed`.
#
# Outputs go to $SCRATCH/301-short-feature-boundaries/step2. The notebook needs only the
# small tables; copy them back with
#
#   rsync -av sherlock-dtn:/scratch/users/olgabot/301-short-feature-boundaries/step2/{truth.tsv,chimeras.tsv,feature_type_counts.tsv,filter_counts.tsv,uniref90_clusters.tsv,uniref90_counts.tsv,sprot_release.txt} data/301_short_feature_boundaries/step2/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
OUT=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}/step2
SPROT_DIR=$SCRATCH/data/swissprot
mkdir -p "$OUT" "$SPROT_DIR"
module load python/3.12.1

URL=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete
if [ ! -s "$SPROT_DIR/uniprot_sprot.dat.gz" ]; then
    # Resumable, and gives up on a stalled connection after a minute.
    curl -fL -C - --retry 5 --speed-limit 10000 --speed-time 60 \
        -o "$SPROT_DIR/uniprot_sprot.dat.gz.part" "$URL/uniprot_sprot.dat.gz"
    mv "$SPROT_DIR/uniprot_sprot.dat.gz.part" "$SPROT_DIR/uniprot_sprot.dat.gz"
    curl -fsSL -o "$SPROT_DIR/reldate.txt" "$URL/reldate.txt"
fi
gzip -t "$SPROT_DIR/uniprot_sprot.dat.gz"
{
    cat "$SPROT_DIR/reldate.txt"
    echo "downloaded: $(stat -c %y "$SPROT_DIR/uniprot_sprot.dat.gz")"
    sha256sum "$SPROT_DIR/uniprot_sprot.dat.gz"
} > "$OUT/sprot_release.txt"

python3 -I "$REPO/scripts/build_301_step2_queries.py" \
    --sprot-dat "$SPROT_DIR/uniprot_sprot.dat.gz" \
    --elm "$REPO/data/301_short_feature_boundaries/step2_elm_true_positive_instances_under60.tsv" \
    --out "$OUT"

python3 -I "$REPO/scripts/fetch_301_uniref90.py" \
    --accessions "$OUT/accessions_for_uniref90.txt" \
    --sprot-dat "$SPROT_DIR/uniprot_sprot.dat.gz" \
    --out "$OUT"

echo "done: $OUT"
