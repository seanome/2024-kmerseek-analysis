#!/bin/bash
#SBATCH --job-name=244-bonus-tools
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=244-bonus-tools.%j.log
#
# The six comparison tools of notebook 244, run for the two bonus queries (Ced-9 P41958 and
# BHF) against the whole human proteome.
#
# In notebook 244 human was the query and nine other species were the targets. Here human is
# the target, so the human databases are built from the full QfO human proteome (sequences)
# and the full AlphaFold v6 human set (structures), the way the pipeline builds any target.
# Every command, flag and container below is copied from
# nextflow-runs/qfo-pfam-region-benchmark/main.nf (phmmerSearch, mmseqsDb, mmseqs2Search,
# foldseekDb, foldseekSearch, reseekConvert, reseekSearch, prostt5Db, prostt5Search), with
# the midi-plus parameters: evalue_report 10, mmseqs2_sensitivity 7, mmseqs2_iterations 3,
# reseek_mode verysensitive, prostt5_max_len 6000, CPU search (gpu_search false). The GPU
# is only for ProstT5's 3Di prediction in createdb, as in the pipeline.
#
# BHF has no AlphaFold model, so Foldseek and Reseek run for Ced-9 only.
#
# Run from a clone of this branch:
#   cd $SCRATCH/244-bonus/code && sbatch scripts/bonus_244_tools.sbatch
# Writes $SCRATCH/244-bonus/out/<tool>.tsv (8 columns: query target qstart qend tstart tend
# score evalue) and versions.txt.
set -euo pipefail

REPO=$(pwd)
DATA=${DATA:-$SCRATCH/2024-kmerseek-midi-plus-all-qfo/nextflow-runs/qfo-pfam-region-benchmark/data}
CACHE=$SCRATCH/apptainer-cache
WORK=${WORK:-$SCRATCH/244-bonus/work}
OUT=${OUT:-$SCRATCH/244-bonus/out}
T=${SLURM_CPUS_PER_TASK:-16}

HUMAN_FASTA=$DATA/qfo/Eukaryota/UP000005640_9606.fasta
HUMAN_STRUCTS=$DATA/structures/human
CED9_CIF=$DATA/structures/worm/AF-P41958-F1.cif
WEIGHTS=$DATA/prostt5/weights
NORMALIZE_RESEEK=$REPO/nextflow-runs/qfo-pfam-region-benchmark/bin/normalize_reseek.awk

img() { echo "$CACHE/quay.io-biocontainers-$1@sha256-$2.img"; }
HMMER=$(img hmmer 7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c)
MMSEQS=$(img mmseqs2 3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef)
FOLDSEEK=$(img foldseek 1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a)
RESEEK=$(img reseek 24f7c37150dd2c2f2f322b1387a08d2d1a4a279f46f98f1051f1745417675752)
run() { local image=$1; shift; apptainer exec --nv -B "$SCRATCH" "$image" "$@"; }

for f in "$HUMAN_FASTA" "$CED9_CIF" "$NORMALIZE_RESEEK" "$HMMER" "$MMSEQS" "$FOLDSEEK" "$RESEEK"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
[ -d "$WEIGHTS" ] || { echo "missing: $WEIGHTS" >&2; exit 1; }

mkdir -p "$WORK" "$OUT"
cd "$WORK"
# Target databases are kept between runs; everything built from the queries is rebuilt.
rm -rf mm_q fs_q fs_result* fs_tmp p5_q p5_result* p5_tmp ced9.bca ced9_cif reseek_raw.tsv

# The two queries, as scripts/export_244_bonus_cases.py kmerseek wrote them. First token = accession.
awk '/^>/ { keep = ($1 == ">P41958" || $1 == ">BHF") } keep' \
    "$REPO/tables/244_bonus_queries.fasta" > queries.fasta
[ "$(grep -c '^>' queries.fasta)" -eq 2 ] || { echo "queries.fasta: expected 2 records" >&2; exit 1; }

{
    echo "phmmer:   $(run "$HMMER" phmmer -h | sed -n 2p)"
    echo "mmseqs:   $(run "$MMSEQS" mmseqs version)"
    echo "foldseek: $(run "$FOLDSEEK" foldseek version)"
    echo "reseek:   $(run "$RESEEK" reseek --version 2>&1 | head -1 || true)"
    echo "human:    $HUMAN_FASTA ($(grep -c '^>' "$HUMAN_FASTA") sequences)"
} > "$OUT/versions.txt"

# ---- phmmer (phmmerSearch) ----
run "$HMMER" phmmer --domtblout /dev/stdout --tblout /dev/null -o /dev/null --noali \
        -E 10 --cpu "$T" queries.fasta "$HUMAN_FASTA" \
    | grep -v '^#' \
    | awk 'NF >= 22 {print $4 "\t" $1 "\t" $16 "\t" $17 "\t" $20 "\t" $21 "\t" $14 "\t" $13}' \
    > "$OUT/phmmer.tsv"
echo "phmmer: $(wc -l < "$OUT/phmmer.tsv") rows"

# ---- MMseqs2 and MMseqs2 iterative (mmseqsDb + mmseqs2Search) ----
mkdir -p mm_q mm_t
[ -e mm_t/db.index ] || run "$MMSEQS" mmseqs createdb "$HUMAN_FASTA" mm_t/db
run "$MMSEQS" mmseqs createdb queries.fasta mm_q/db
for variant in seqseq iterative; do
    iter=""; [ "$variant" = iterative ] && iter="--num-iterations 3"
    rm -rf "mm_$variant" && mkdir -p "mm_$variant/tmp"
    run "$MMSEQS" mmseqs search mm_q/db mm_t/db "mm_$variant/result" "mm_$variant/tmp" \
        --threads "$T" -s 7 $iter --max-seqs 1000 -e 10
    run "$MMSEQS" mmseqs convertalis mm_q/db mm_t/db "mm_$variant/result" "$OUT/mmseqs2_$variant.tsv" \
        --threads "$T" --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"
    echo "mmseqs2 $variant: $(wc -l < "$OUT/mmseqs2_$variant.tsv") rows"
done

# ---- Structures: human target set, .cif only ----
# The human directory holds each model twice (AF-*.cif and the downloaded AF-*.cif.gz);
# every species directory the pipeline used as a target holds .cif only. Link the .cif files
# into one directory so both tools read each model once.
mkdir -p human_cif
if [ ! -e human_cif/.done ]; then
    find "$HUMAN_STRUCTS"/ -maxdepth 1 -name 'AF-*.cif' -exec ln -sf {} human_cif/ \;
    touch human_cif/.done
fi
n_cif=$(find human_cif/ -name 'AF-*.cif' | wc -l)
echo "human structures: $n_cif .cif files"
mkdir -p ced9_cif && ln -sf "$CED9_CIF" ced9_cif/

# ---- Foldseek (foldseekDb + foldseekSearch) ----
mkdir -p fs_t fs_q fs_tmp
[ -e fs_t/db.index ] || run "$FOLDSEEK" foldseek createdb human_cif/ fs_t/db --threads "$T"
n_fs=$(wc -l < fs_t/db.index)
echo "foldseek human db: $n_fs entries"
# The pipeline read real files; these are symlinks. Fail rather than search an empty set.
[ "$n_fs" -ge $(( n_cif * 9 / 10 )) ] || { echo "foldseek read $n_fs of $n_cif models" >&2; exit 1; }
run "$FOLDSEEK" foldseek createdb ced9_cif/ fs_q/db --threads "$T"
run "$FOLDSEEK" foldseek search fs_q/db fs_t/db fs_result fs_tmp --threads "$T" -e 10 --max-seqs 1000
run "$FOLDSEEK" foldseek convertalis fs_q/db fs_t/db fs_result fs_out.tsv --threads "$T" \
    --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"
awk -F'\t' 'BEGIN{OFS="\t"} {
    for (i = 1; i <= 2; i++) {
        if (match($i, /AF-[A-Z0-9]+-F[0-9]+/)) {
            tok = substr($i, RSTART, RLENGTH)
            split(tok, p, "-")
            off = (substr(p[3], 2) + 0 - 1) * 200
            if (off) { $(2 * i + 1) += off; $(2 * i + 2) += off }
            $i = p[2]
        }
    }
    print
}' fs_out.tsv > "$OUT/foldseek.tsv"
echo "foldseek: $(wc -l < "$OUT/foldseek.tsv") rows"

# ---- Reseek (reseekConvert + reseekSearch) ----
if [ ! -e human.mu.fasta ]; then
    find -L human_cif/ \( -name '*.cif' -o -name '*.mmcif' -o -name '*.pdb' \) | sort > human.files
    run "$RESEEK" reseek -convert human.files -bca human.bca -threads "$T" -log human.convert.log
    run "$RESEEK" reseek -convert2mu human.bca -fasta human.mu.fasta -threads "$T" -log human.mu.log
fi
find -L ced9_cif/ -name '*.cif' > ced9.files
run "$RESEEK" reseek -convert ced9.files -bca ced9.bca -threads "$T" -log ced9.convert.log
run "$RESEEK" reseek -search ced9.bca -db human.bca -dbmu human.mu.fasta -verysensitive \
    -evalue 10 -columns query+target+qlo+qhi+tlo+thi+aq+evalue \
    -threads "$T" -log reseek.search.log -output reseek_raw.tsv
awk -f "$NORMALIZE_RESEEK" reseek_raw.tsv > "$OUT/reseek.tsv"
echo "reseek: $(wc -l < "$OUT/reseek.tsv") rows"

# ---- ProstT5 (prostt5Db + prostt5Search) ----
filter_len() {  # the prostt5Db length filter, 6000 aa
    awk -v maxlen=6000 '
        function flush() { if (hdr != "" && length(seq) <= maxlen) { print hdr; print seq } }
        /^>/ { flush(); hdr = $0; seq = ""; next } { seq = seq $0 } END { flush() }' "$1"
}
mkdir -p p5_t p5_q p5_tmp
if [ ! -e p5_t/db.index ]; then
    filter_len "$HUMAN_FASTA" > human.p5.fasta
    run "$FOLDSEEK" foldseek createdb human.p5.fasta p5_t/db --prostt5-model "$WEIGHTS" --gpu 1 --threads "$T"
fi
filter_len queries.fasta > queries.p5.fasta
run "$FOLDSEEK" foldseek createdb queries.p5.fasta p5_q/db --prostt5-model "$WEIGHTS" --gpu 1 --threads "$T"
run "$FOLDSEEK" foldseek search p5_q/db p5_t/db p5_result p5_tmp --threads "$T" -e 10 --max-seqs 1000
run "$FOLDSEEK" foldseek convertalis p5_q/db p5_t/db p5_result p5_out.tsv --threads "$T" \
    --format-output "query,target,qstart,qend,tstart,tend,bits,evalue"
awk -F'\t' 'BEGIN{OFS="\t"} { for (i = 1; i <= 2; i++) { n = split($i, p, "|"); if (n >= 2) $i = p[2] } print }' \
    p5_out.tsv > "$OUT/prostt5.tsv"
echo "prostt5: $(wc -l < "$OUT/prostt5.tsv") rows"

echo "done: $OUT"
ls -la "$OUT"
