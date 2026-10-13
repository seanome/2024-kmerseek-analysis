#!/bin/bash
# Re-run phmmer, jackhmmer and mmseqs2 for notebook 281's Botryllus proteins against the dark
# set's reference (Swiss-Prot minus Ascidiacea), keeping each hit's position on the Botryllus
# protein as well as on the target. The dark-set run (PR 57, main.nf phmmerSearch,
# jackhmmerSearch, mmseqs2Search) saved only the target positions, and notebook 281 needs the
# query positions to place tool hits on the Botryllus protein.
#
# Same images, database and settings as that run: E <= 10 reporting, jackhmmer -N 3,
# mmseqs2 -s 7 --num-iterations 3. This time the full output is kept:
#   <out>/<tool>.chunk_N.domtblout.gz   HMMER per-domain table, every column
#   <out>/mmseqs2.chunk_N.tsv.gz        query,target,qstart,qend,tstart,tend,bits,evalue,qlen,tlen
# scripts/parse_query_position_hits.py turns them into one table.
#
#   sbatch --array=0-7 -c 16 --mem 16G -t 3:00:00  281_tool_search_query_positions.sbatch phmmer    <chunks_dir> <out>
#   sbatch --array=0-7 -c 16 --mem 32G -t 10:00:00 281_tool_search_query_positions.sbatch jackhmmer <chunks_dir> <out>
#   sbatch --array=0-7 -c 16 --mem 8G  -t 1:00:00  281_tool_search_query_positions.sbatch mmseqs2   <chunks_dir> <out>
#
# <chunks_dir> holds chunk_0.fasta ... chunk_7.fasta. Resources from the dark-set run's own
# Botryllus tasks (trace.full.txt, ~1_970 proteins per chunk): phmmer max 1.2 h / 11.6 GB,
# jackhmmer max 3.8 h / 17.5 GB, mmseqs2 max 3 min / 2.6 GB. These chunks hold ~900 proteins,
# but proteins with hits run slower, so the time asks keep about 2x headroom over the old max.
# Added 2026-10-13 (notebook 281, PR 126).
#SBATCH -p hns,normal
#SBATCH -A ayeletv
#SBATCH -J bmhc-qpos
set -euo pipefail
TOOL=$1; CHUNKS=$2; OUT=$3
I=${SLURM_ARRAY_TASK_ID}
REF=/scratch/users/olgabot/invertebrate-dark-set/reference/minus_Ascidiacea/reference.fasta
IMGS=$SCRATCH/apptainer-cache
HMMER=$IMGS/quay.io-biocontainers-hmmer@sha256-7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c.img
MMSEQS=$IMGS/quay.io-biocontainers-mmseqs2@sha256-3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef.img
Q=$CHUNKS/chunk_${I}.fasta
mkdir -p "$OUT"
TMP=$(mktemp -d "$L_SCRATCH/qpos.XXXXXX" 2>/dev/null || mktemp -d)
trap 'rm -rf "$TMP"' EXIT
case "$TOOL" in
  phmmer)
    apptainer exec -B /scratch "$HMMER" phmmer --domtblout "$TMP/out" --tblout /dev/null -o /dev/null --noali \
        -E 10 --cpu "$SLURM_CPUS_PER_TASK" "$Q" "$REF"
    gzip -c "$TMP/out" > "$OUT/phmmer.chunk_${I}.domtblout.gz.tmp" ;;
  jackhmmer)
    apptainer exec -B /scratch "$HMMER" jackhmmer -N 3 --domtblout "$TMP/out" --tblout /dev/null -o /dev/null --noali \
        -E 10 --cpu "$SLURM_CPUS_PER_TASK" "$Q" "$REF"
    gzip -c "$TMP/out" > "$OUT/jackhmmer.chunk_${I}.domtblout.gz.tmp" ;;
  mmseqs2)
    cd "$TMP"
    apptainer exec -B /scratch "$MMSEQS" mmseqs createdb "$Q" qdb
    apptainer exec -B /scratch "$MMSEQS" mmseqs createdb "$REF" tdb
    apptainer exec -B /scratch "$MMSEQS" mmseqs search qdb tdb res tmp -s 7 --num-iterations 3 -e 10 \
        --threads "$SLURM_CPUS_PER_TASK"
    apptainer exec -B /scratch "$MMSEQS" mmseqs convertalis qdb tdb res out.tsv \
        --format-output 'query,target,qstart,qend,tstart,tend,bits,evalue,qlen,tlen'
    gzip -c out.tsv > "$OUT/mmseqs2.chunk_${I}.tsv.gz.tmp" ;;
  *) echo "tool must be phmmer, jackhmmer or mmseqs2" >&2; exit 2 ;;
esac
for f in "$OUT"/"$TOOL".chunk_"$I".*.tmp; do mv "$f" "${f%.tmp}"; done
echo "$TOOL chunk $I done"
