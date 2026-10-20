#!/bin/bash
# Notebook 359, step 6, one setting: build the kmerseek index of the Swiss-Prot regions
# index (Swiss-Prot 2026_03 minus Mammalia, annotated regions with 21-residue flanks,
# clustered; the regions-index pilot's regions_index.fasta, targets only), fit E-values,
# then search every real and decoy chunk and keep each blank exon's strongest landing call.
# Memory and time are set by build.sbatch from settings.tsv and the query count.
#
#   sbatch kmerseek.sbatch <setting> <alphabet> <ksize> <penalty> <xdrop>
#
# Index flags are the pilot's (kmerseekIndex in nextflow-runs/regions-index-pilot/main.nf,
# --ka-queries 2500 for a regions index). Search flags are the dark-set run's, read from
# its own commands (scripts/359_isoform_blank_exons/settings.README.md).
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x.%j.log

set -uo pipefail
setting=$1 alphabet=$2 k=$3 pen=$4 xdrop=$5
WORK=$(pwd)
IMG=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.1-rc1.img
REGIONS=/scratch/users/olgabot/regions-index-pilot/results/regions_index/regions_index.fasta
REGIONS_MD5=8dca1949648e1e179cde22f5b665a856
IDX=$WORK/kmerseek/index/$setting.kmerseek.rocksdb
BEST=$WORK/kmerseek/best
mkdir -p "$WORK/kmerseek/index" "$BEST/chunks"
ks() { apptainer exec --bind /scratch "$IMG" "$@"; }

[ "$(md5sum < "$REGIONS" | cut -d' ' -f1)" = "$REGIONS_MD5" ] || { echo "regions_index.fasta md5 changed" >&2; exit 1; }
ks kmerseek --version

if [ ! -f "$IDX/.done" ]; then
    command rm -rf "$IDX"
    /usr/bin/time -v apptainer exec --bind /scratch "$IMG" kmerseek index --alphabet "$alphabet" --ksize "$k" --scaled 1 \
        --input "$REGIONS" --output "$IDX" \
        --extend-mismatch-penalty "$pen" --extend-xdrop "$xdrop" \
        --ka-queries 2500 --ka-survival-out "$IDX/ka_survival.C$pen.csv" \
        --kmer-stats-out "$IDX/spectrum.csv.gz" || exit 1
    touch "$IDX/.done"
fi

for chunk in "$WORK"/kmerseek/chunks/*.fasta; do
    name=$(basename "$chunk" .fasta)
    side=${name%%_*}
    out=$BEST/chunks/$setting.$name.best.tsv
    [ -s "$out" ] && continue
    /usr/bin/time -v apptainer exec --bind /scratch "$IMG" kmerseek search --alphabet "$alphabet" --ksize "$k" \
        --query "$chunk" --target "$IDX" \
        --extend-mismatch-penalty "$pen" --extend-xdrop "$xdrop" --ka-queries 0 \
        --chain-max-gap 30 --chain-max-shift 10 \
        --threshold 0.0 --min-shared-kmers 2 --max-query-pvalue 0.05 \
        --min-region-score 1.3 2> "$BEST/chunks/$setting.$name.search.log" \
      | ks python3 code/scripts/359_isoform_blank_exons/reduce_calls.py "$WORK/build/blank_exons.parquet" "$side" > "$out.part"
    status=("${PIPESTATUS[@]}")
    if [ "${status[0]}" -ne 0 ] && grep -q "no Karlin-Altschul fit" "$BEST/chunks/$setting.$name.search.log"; then
        # The index refused the E-value fit: this setting has no region_evalue to threshold.
        echo "no fit" > "$BEST/$setting.nofit"
        command rm -f "$out.part"
        exit 0
    fi
    for s in "${status[@]}"; do [ "$s" -eq 0 ] || { tail -20 "$BEST/chunks/$setting.$name.search.log" >&2; exit 1; }; done
    mv "$out.part" "$out"
done

for side in real decoy; do
    files=("$BEST"/chunks/"$setting".${side}_*.best.tsv)
    { head -1 "${files[0]}"; for f in "${files[@]}"; do tail -n +2 "$f"; done; } > "$BEST/$setting.$side.best.tsv"
done
wc -l "$BEST/$setting".*.best.tsv
