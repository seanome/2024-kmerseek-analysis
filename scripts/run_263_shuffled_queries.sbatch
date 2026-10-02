#!/bin/bash
# Shuffled-query twin of the four zebrafish searches notebooks 260 to 263 read.
#
# The real searches are the random-alphabet control (PR 76, run 2026-09-30 from
# /scratch/users/olgabot/2024-kmerseek-random-alphabets): the 998 midi-plus human queries
# against zebrafish, hp_pbotc_1st_ed2 at k19, built in and encoded as two letters run as
# protein20, each at extension penalty C 1.63 and 2. This job repeats them with one
# dipeptide shuffle of each query (scripts/make_decoy_queries.py, seed 270, accession
# prefixed DECOY_), against the same indexes and their stored Karlin-Altschul fits
# (--ka-queries 0), with the same flags, in the same image. Then it cuts every result to
# region_evalue < 10 with notebook 260's reduce, which is what comes back to the laptop.
#
# Before searching, it encodes the REAL human queries with the same encoder and partition
# and checks the md5 equals the encoded file the real run searched, so the encoded shuffles
# are made the way the real encoded queries were.
#
# Run from an empty work directory holding code/ (a clone of this branch) and logs/:
#   sbatch code/scripts/run_263_shuffled_queries.sbatch
#
# Resources, from the real run's trace (qfo_pfam_region.2026-09-30.trace.txt): each search
# 3-4 min realtime on 4 CPUs, peak RSS 8.7 GB, peak vmem 10.8 GB. Four searches one after
# another, plus the shuffle, encode and reduce: about 20 min. Asked: 4 CPUs, 24 GB (2x the
# largest peak vmem), 2 h.
#SBATCH --job-name=263-shuffled
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.log

set -euo pipefail
B=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark
RA=/scratch/users/olgabot/2024-kmerseek-random-alphabets
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
IDX=$B/results/kmerseek_index_0.4/ka_fit
REAL_ENCODED=$B/results/encoded_fasta/human-38fdd87d.encoded_hp_pbotc_1st_ed2.fasta
ENCODER=$RA/nextflow-runs/qfo-pfam-region-benchmark/bin/encode_partition.py
PARTITION=$RA/data/random_alphabets/encoded_hp_pbotc_1st_ed2.tsv
run() { apptainer exec --bind /scratch "$IMAGE" "$@"; }
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "host $(hostname), job ${SLURM_JOB_ID:-none}, code $(cd code && git rev-parse HEAD)"
run kmerseek --version

# 1. One dipeptide shuffle of each human query.
run python3 code/scripts/make_decoy_queries.py --qfo-dir "$B/data/midi-plus/qfo" --out-dir qfo-decoy
HUMAN_DECOY=qfo-decoy/Eukaryota/UP000005640_9606.fasta
echo "shuffled queries: $(grep -c '>' "$HUMAN_DECOY")"

# 2. Encode: first the real queries, which must reproduce the real run's encoded file.
run python3 "$ENCODER" --partition "$PARTITION" --input "$B/data/midi-plus/qfo/Eukaryota/UP000005640_9606.fasta" \
    --output real_check.encoded_hp_pbotc_1st_ed2.fasta
a=$(md5sum < real_check.encoded_hp_pbotc_1st_ed2.fasta | cut -c1-32)
b=$(md5sum < "$REAL_ENCODED" | cut -c1-32)
echo "encoder check: re-encoded real queries $a, real run's encoded file $b"
[ "$a" = "$b" ] || { echo "encoder does not reproduce the real run's encoded queries" >&2; exit 1; }
run python3 "$ENCODER" --partition "$PARTITION" --input "$HUMAN_DECOY" \
    --output human_decoy.encoded_hp_pbotc_1st_ed2.fasta

# 3. Four searches, the real run's flags (main.nf kmerseekExtendSearch, PR 76).
mkdir -p kmerseek
for label in hp_pbotc_1st_ed2 encoded_hp_pbotc_1st_ed2; do
  if [ "$label" = hp_pbotc_1st_ed2 ]; then flag=hp_pbotc_1st_ed2; q=$HUMAN_DECOY
  else flag=protein20; q=human_decoy.encoded_hp_pbotc_1st_ed2.fasta; fi
  for c in 1.63 2; do
    # X-drop = 4 x C, written as the real run wrote it (main.nf penaltyString).
    case $c in 1.63) xd=6.52;; 2) xd=8;; esac
    slug=human_vs_zebrafish.$label.k19.lcfalse.extend-c$c
    t0=$(date +%s)
    run kmerseek search --alphabet "$flag" --ksize 19 \
        --query "$q" --target "$IDX/zebrafish.$label.k19.lcfalse.kmerseek.rocksdb" \
        --extend-mismatch-penalty "$c" --extend-xdrop "$xd" --ka-queries 0 \
        --chain-max-gap 30 --chain-max-shift 10 \
        --threshold 0.0 --min-shared-kmers 2 --max-query-pvalue 0.05 --min-region-score 1.3 \
        2> "kmerseek/$slug.log" | zstd -q -T2 -o "kmerseek/$slug.regions.csv.zst"
    run python3 - "kmerseek/$slug" <<'PYEOF'
import sys, polars as pl
s = sys.argv[1]
DROP = ["query_md5", "target_md5", "region_subseq", "target_subseq", "moltype_seq"]
lf = pl.scan_csv(f"{s}.regions.csv.zst", ignore_errors=True)
lf.select([c for c in lf.collect_schema().names() if c not in DROP]).sink_parquet(
    f"{s}.regions.parquet", compression="zstd", compression_level=9)
PYEOF
    rm "kmerseek/$slug.regions.csv.zst"
    echo "$slug: xdrop $xd, $(( $(date +%s) - t0 )) s"
  done
done

# 4. region_evalue < 10, notebook 260's reduce.
run python3 code/notebooks/region_table_260.py reduce --kmerseek-dir kmerseek --out-dir reduced_decoy
ls -la reduced_decoy
