#!/bin/bash
# Rescore the ELM cover run's Reseek arm with corrected target positions (notebook 253).
#
# bin/normalize_reseek.awk adds an offset of (n - 1) * 200 for AlphaFold model fragment F<n>,
# but it reads n from the FIRST "-F<digits>" in the file name. For a UniProt accession that
# starts with F and a digit (F6SXM4, F7B1X4, F8VPJ6) that is the accession, so the positions
# in AF-F6SXM4-F1.cif get 1_000 added. This undoes that offset in a copy of each species'
# Reseek table, (d - 1) * 200 for an accession starting F<d>, and rescores Reseek alone with
# scripts/reduce_elm_cover.py.
#
# Every query is human and none of the human query accessions starts with F and a digit, so
# only the target side (columns 5 and 6) needs the correction. A real
# fragment F2 or later of an F<d> accession (proteins over 2_700 aa) got (d - 1) * 200 instead
# of its own offset; after this it is left at offset 0, not its true offset.
#
# Run on Sherlock inside a job (it reads ~1.5 GB of compressed tables):
#   srun -A ayeletv -p normal -t 60 -c 8 --mem 64G bash scripts/rescore_elm_reseek_offset.sh
# CODE is a checkout holding scripts/reduce_elm_cover.py and the region benchmark's bin/ and assets/.
set -euo pipefail
PIPE=${PIPE:-/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark}
CODE=${CODE:-/scratch/users/olgabot/250-elm-cover/code}
OUT=${OUT:-/scratch/users/olgabot/250-elm-cover/reseek-fix}
IMG=${IMG:-/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img}
SPECIES=${SPECIES:-mouse zebrafish ciona fly worm yeast arabidopsis}
COVER=$PIPE/data/elm-cover
A=$CODE/nextflow-runs/qfo-pfam-region-benchmark/assets

mkdir -p "$OUT/results/regions/reseek" "$OUT/results/kmerseek_top1000"
for sp in $SPECIES; do
  zcat "$COVER/results/regions/reseek/human_vs_$sp.reseek.tsv.gz" \
    | awk -F'\t' 'BEGIN { OFS = "\t" }
        $2 ~ /^F[0-9]/ { off = (substr($2, 2, 1) - 1) * 200; $5 -= off; $6 -= off }
        { print }' \
    | gzip -c > "$OUT/results/regions/reseek/human_vs_$sp.reseek.tsv.gz"
  mkdir -p "$OUT/tmp/$sp"
  (cd "$OUT/tmp/$sp" && apptainer exec --bind /scratch "$IMG" python3 "$CODE/scripts/reduce_elm_cover.py" \
    --results "$OUT/results" --species "$sp" --kmerseek-subdir kmerseek_top1000 \
    --instances "$A/elm_cover_instances.tsv" --orthologs "$A/elm_cover_orthologs.tsv" \
    --projections "$A/elm_cover_projections.tsv" \
    --query-fasta "$COVER/qfo/Eukaryota/UP000005640_9606.fasta" --structures "$COVER/structures/human" \
    --qfo-bin "$CODE/nextflow-runs/qfo-pfam-region-benchmark/bin" \
    --out-dir "$OUT/cover/$sp")
done
