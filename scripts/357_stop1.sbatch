#!/bin/bash
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --output=357-stop1-%x-%A_%a.out
#
# Notebook 357, Stop 1: the nb 342 exon queries against the condensate-region index, one step
# per job. scripts/357_stop1_submit.sh submits every step with its --cpus/--mem/--time from
# tables/357/357_stop1_resources.tsv and chains them with dependencies; run that, not this.
#
#   STEP=index    one condensate index (ALPHABET, KSIZE) of
#                 tables/357/357_condensate_regions.w10.target_decoy.fasta (88 regions and their
#                 10-residue-window decoys), with no Karlin-Altschul fit of its own (--ka-queries 0).
#   STEP=search   one chunk (SLURM_ARRAY_TASK_ID) of nb 342's 9,026 queries against it. E-values
#                 use nb 342's whole-protein index fit for the same alphabet, k and penalty
#                 (--ka-k, --ka-r-database), as the pre-registration says. Keeps regions with
#                 E <= 10, in nb 342's call-file format.
#
# kmerseek is nb 342's build of tag nb342-kmerseek-9ce5865, reused, not rebuilt. Every step
# skips work already done (a finished index has a DONE file, a finished chunk its .tsv.gz).
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}" "${STEP:?}"
REPO=$SLURM_SUBMIT_DIR
WORK=${NB357_WORK:-$SCRATCH/condensate-357}
NB342_WORK=$SCRATCH/exon-function-342
KMERSEEK=$NB342_WORK/kmerseek-src/target/release/kmerseek
TARGETS=$REPO/tables/357/357_condensate_regions.w10.target_decoy.fasta
QUERIES=$REPO/tables/357/inputs/342_setA_exon_queries_with_decoys.fasta
QUERIES_MD5=fb388f253f1026d4e9a77acf37e86870
N_CHUNKS=10
# Search options as nb 342 Stop 3 and PR 114's pilot (BUILD_REPORT.md, "kmerseek settings").
SEARCH_OPTS="--threshold 0.0 --min-shared-kmers 2 --max-query-pvalue 0.05 --min-region-score 1.3 --chain-max-gap 30 --chain-max-shift 10"
MAX_EVALUE=10
mkdir -p "$WORK"
# kmerseek was built with gcc 14.2.0 and needs its libstdc++ (CXXABI_1.3.15).
GCC14_LIB=/share/software/user/open/gcc/14.2.0/lib64
kmerseek() { LD_LIBRARY_PATH=$GCC14_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "$KMERSEEK" "$@"; }

penalty() {  # c_opt and xdrop_4c of an alphabet
    awk -F'\t' -v a="$1" '!/^#/ && $1 == a { print $2, $3 }' "$REPO/tables/357/357_penalties.tsv"
}

# `|| true`: read returns 1 on an empty list, and set -e would skip the message below.
read -r C X < <(penalty "${ALPHABET:?}") || true
[ -n "$C" ] || { echo "357: no penalty for $ALPHABET" >&2; exit 2; }
slug=${ALPHABET}_k${KSIZE:?}_s1
idx=$WORK/indexes/condensate.w10.$slug.kmerseek.rocksdb

case "$STEP" in
index)
    # Reuse a finished index only if it was built from this exact targets FASTA.
    [ -f "$idx/DONE" ] && [ "$(cat "$idx/targets.md5" 2>/dev/null)" = "$(md5sum "$TARGETS")" ] \
        && { echo "357: $idx done already"; exit 0; }
    rm -rf "$idx" && mkdir -p "$(dirname "$idx")"
    kmerseek index --alphabet "$ALPHABET" --ksize "$KSIZE" --scaled 1 \
        --input "$TARGETS" --output "$idx" \
        --extend-mismatch-penalty "$C" --extend-xdrop "$X" --ka-queries 0 \
        --kmer-stats-out "$idx/spectrum.csv.gz"
    md5sum "$TARGETS" > "$idx/targets.md5"
    kmerseek --version > "$idx/DONE"
    ;;
search)
    : "${SLURM_ARRAY_TASK_ID:?}"
    fit=$NB342_WORK/indexes/whole.w10.$slug.kmerseek.rocksdb/ka_survival.C$C.csv
    [ -f "$idx/DONE" ] || { echo "357: no finished index $idx" >&2; exit 2; }
    [ -s "$fit" ] || { echo "357: nb 342's whole-protein index has no fit for $slug ($fit); no E-values" >&2; exit 4; }
    [ "$(md5sum "$QUERIES" | cut -d' ' -f1)" = $QUERIES_MD5 ] || { echo "357: $QUERIES is not nb 342's query file" >&2; exit 2; }
    # Every row of the survival CSV carries the fit's K and r_database.
    read -r KA_K KA_R < <(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
                                   NR == 2 { print $c["k"], $c["r_database"]; exit }' "$fit")
    chunk=$SLURM_ARRAY_TASK_ID
    out=$WORK/calls/$slug/chunk_$(printf %02d "$chunk").calls.tsv.gz
    [ -s "$out" ] && { echo "357: $out done already"; exit 0; }
    mkdir -p "$(dirname "$out")"
    # Query chunk: records i with i mod N_CHUNKS == chunk, so exons and decoys spread evenly.
    q=${TMPDIR:-/tmp}/357.q.$slug.$chunk.fasta
    awk -v n=$N_CHUNKS -v c="$chunk" '/^>/ { r++ } (r - 1) % n == c' "$QUERIES" > "$q"
    echo "357: $slug chunk $chunk, $(grep -c '>' "$q") queries, fit K $KA_K r_database $KA_R"
    # kmerseek writes regions 0-based and end-exclusive; the call file is 1-based and
    # inclusive (start + 1, end as written), as nb 342's call files.
    set +e
    kmerseek search --alphabet "$ALPHABET" --ksize "$KSIZE" --query "$q" --target "$idx" \
        --extend-mismatch-penalty "$C" --extend-xdrop "$X" --ka-queries 0 \
        --ka-k "$KA_K" --ka-r-database "$KA_R" $SEARCH_OPTS 2> "${out%.tsv.gz}.log" \
      | awk -F, -v OFS='\t' -v max=$MAX_EVALUE '
          NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i
                    split("query_name target_name region_start region_end target_start target_end region_evalue region_subseq target_subseq", need, " ")
                    for (n in need) if (!(need[n] in c)) { print "missing column " need[n] > "/dev/stderr"; exit 3 }
                    print "query", "qstart", "qend", "target", "tstart", "tend", "evalue", "qseq", "tseq"; next }
          ($c["region_evalue"] + 0) <= max {
                    print $c["query_name"], $c["region_start"] + 1, $c["region_end"],
                          $c["target_name"], $c["target_start"] + 1, $c["target_end"],
                          $c["region_evalue"], $c["region_subseq"], $c["target_subseq"] }' \
      | gzip -c > "$out.part"
    status=("${PIPESTATUS[@]}")
    set -e
    for s in "${status[@]}"; do [ "$s" -eq 0 ] || { echo "357: search failed (${status[*]})" >&2; exit 1; }; done
    mv "$out.part" "$out"
    echo "357: $(zcat "$out" | tail -n +2 | wc -l) regions with E <= $MAX_EVALUE"
    ;;
*)
    echo "357: unknown STEP $STEP" >&2; exit 2
    ;;
esac
