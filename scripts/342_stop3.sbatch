#!/bin/bash
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --output=342-stop3-%x-%A_%a.out
#
# Notebook 342, Stop 3: kmerseek against the regions index (PR 114), one step per job.
# scripts/342_stop3_submit.sh submits every step with its own --cpus/--mem/--time from
# tables/342/342_stop3_resources.tsv and chains them with dependencies; run that, not this.
#
#   STEP=build    build kmerseek from the tag nb342-kmerseek-9ce5865 (seanome/kmerseek):
#                 main 00395d7 + PR 146 db906a3 (aro_hyd_chg_pol4, aro_hyd_arg_chg_pol5)
#                 + PR 144 9cd7096 (--ka-r-database) + PR 152 3c23fd6 (index entries keyed
#                 by their sequence, the port of PR 141's equal-k-mer-set fix).
#   STEP=decoys   the target + decoy FASTAs of both indexes, 10-residue windows, made by PR
#                 114's make_window_decoys.py with its seed; each decoy file's md5 must equal
#                 the one PR 114 published, so the indexes hold the same decoys.
#   STEP=index    one kmerseek index (ALPHABET, KSIZE, INDEX=regions|whole). The whole-protein
#                 index is built only for its Karlin-Altschul fit.
#   STEP=search   one chunk (SLURM_ARRAY_TASK_ID) of the queries against the regions index,
#                 with the whole-protein index's fit (--ka-k, --ka-r-database), as PR 114's
#                 "regionsshared" searches. Keeps regions with E <= 10.
#   STEP=score    scripts/342_kind4_features.py: own-gene filter, decoy-matched cutoffs and
#                 the kind 4 features per exon.
#
# Every step skips work already done (a finished index has a DONE file, a finished chunk
# its .tsv.gz), so resubmitting picks up where a run stopped.
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}" "${STEP:?}"
REPO=$SLURM_SUBMIT_DIR
WORK=${NB342_WORK:-$SCRATCH/exon-function-342}
PILOT=$SCRATCH/regions-index-pilot/results
PILOT_REPO=$SCRATCH/2024-kmerseek-analysis-regions-index/nextflow-runs/regions-index-pilot
KMERSEEK_TAG=nb342-kmerseek-9ce5865
KMERSEEK_SRC=$WORK/kmerseek-src
KMERSEEK=$KMERSEEK_SRC/target/release/kmerseek
# polars 1.34.0 in the venv notebook 283 made on the python/3.12.1 module.
PY=$HOME/venvs/nb283/bin/python
QUERIES=$REPO/tables/342/342_setA_exon_queries_with_decoys.fasta
N_CHUNKS=10
DECOY_SEED=20261006
DECOY_WINDOW=10
# Search options as PR 114's pilot (BUILD_REPORT.md, "kmerseek settings").
SEARCH_OPTS="--threshold 0.0 --min-shared-kmers 2 --max-query-pvalue 0.05 --min-region-score 1.3 --chain-max-gap 30 --chain-max-shift 10"
MAX_EVALUE=10
mkdir -p "$WORK"
module load python/3.12.1
# kmerseek is built with gcc 14.2.0 (STEP=build) and needs its libstdc++ (CXXABI_1.3.15), but
# python/3.12.1 puts gcc 12.4.0's first on LD_LIBRARY_PATH; run kmerseek with gcc 14.2.0's.
GCC14_LIB=/share/software/user/open/gcc/14.2.0/lib64
kmerseek() { LD_LIBRARY_PATH=$GCC14_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "$KMERSEEK" "$@"; }

penalty() {  # c_opt and xdrop_4c of an alphabet, from the tables committed in Stop 0
    awk -F'\t' -v a="$1" '!/^#/ && $1 == a { print $7, $8 }' \
        "$REPO/tables/342/kappa_by_alphabet_new_rows.tsv" "$REPO/tables/342/dark_set_kappa_by_alphabet.ec91da5.tsv" | head -1
}

case "$STEP" in
build)
    module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0
    # The llvm module puts LLVM's libc++ headers on CPATH and RocksDB then fails to compile
    # ("'abort' has not been declared in 'std'"); keep gcc's headers (notebook 283's fix).
    LLVM_ROOT=/share/software/user/open/llvm/17.0.6
    CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v "^$LLVM_ROOT" || true; } | paste -sd: -)
    export CPATH LIBCLANG_PATH=$LLVM_ROOT/lib CC=gcc CXX=g++ CARGO_HOME=$SCRATCH/.cargo
    [ -d "$KMERSEEK_SRC/.git" ] || \
        git clone --quiet --depth 1 --branch "$KMERSEEK_TAG" https://github.com/seanome/kmerseek "$KMERSEEK_SRC"
    cd "$KMERSEEK_SRC"
    [ "$(git rev-parse --short=7 HEAD)" = 9ce5865 ] || { echo "342: $KMERSEEK_SRC is not 9ce5865" >&2; exit 2; }
    cargo build --release --bin kmerseek
    "$KMERSEEK" --version | tee "$WORK/kmerseek_version.txt"
    ;;
decoys)
    declare -A SOURCE=([regions_index]=$PILOT/regions_index/regions_index.fasta
                       [whole_protein_index]=$PILOT/whole_protein_index/reference.fasta)
    mkdir -p "$WORK/fasta"
    for name in regions_index whole_protein_index; do
        out=$WORK/fasta/$name.w$DECOY_WINDOW.target_decoy.fasta
        [ -s "$out" ] && continue
        $PY "$PILOT_REPO/bin/make_window_decoys.py" --fasta "${SOURCE[$name]}" \
            --decoy-out "$WORK/fasta/$name.w$DECOY_WINDOW.decoy.fasta" --combined-out "$out.part" \
            --window $DECOY_WINDOW --seed $DECOY_SEED
        want=$(cut -d' ' -f1 "$PILOT/$name/$name.w$DECOY_WINDOW.decoy.fasta.md5")
        got=$(md5sum "$WORK/fasta/$name.w$DECOY_WINDOW.decoy.fasta" | cut -d' ' -f1)
        [ "$want" = "$got" ] || { echo "342: $name decoys md5 $got, PR 114 published $want" >&2; exit 3; }
        mv "$out.part" "$out"
        echo "$name: decoys match PR 114 (md5 $got), $(grep -c '>' "$out") entries searched"
    done
    ;;
index)
    : "${ALPHABET:?}" "${KSIZE:?}" "${INDEX:?}"
    read -r C X < <(penalty "$ALPHABET")
    [ -n "$C" ] || { echo "342: no penalty for $ALPHABET" >&2; exit 2; }
    name=$([ "$INDEX" = whole ] && echo whole_protein_index || echo regions_index)
    kq=$([ "$INDEX" = whole ] && echo 500 || echo 2500)  # PR 114's --ka-queries per index
    idx=$WORK/indexes/$INDEX.w$DECOY_WINDOW.${ALPHABET}_k${KSIZE}_s1.kmerseek.rocksdb
    [ -f "$idx/DONE" ] && { echo "342: $idx done already"; exit 0; }
    rm -rf "$idx" && mkdir -p "$(dirname "$idx")"
    kmerseek index --alphabet "$ALPHABET" --ksize "$KSIZE" --scaled 1 \
        --input "$WORK/fasta/$name.w$DECOY_WINDOW.target_decoy.fasta" --output "$idx" \
        --extend-mismatch-penalty "$C" --extend-xdrop "$X" \
        --ka-queries $kq --ka-survival-out "$idx/ka_survival.C$C.csv" \
        --kmer-stats-out "$idx/spectrum.csv.gz"
    kmerseek --version > "$idx/DONE"
    ;;
search)
    : "${ALPHABET:?}" "${KSIZE:?}" "${SLURM_ARRAY_TASK_ID:?}"
    read -r C X < <(penalty "$ALPHABET")
    slug=${ALPHABET}_k${KSIZE}_s1
    regions=$WORK/indexes/regions.w$DECOY_WINDOW.$slug.kmerseek.rocksdb
    fit=$WORK/indexes/whole.w$DECOY_WINDOW.$slug.kmerseek.rocksdb/ka_survival.C$C.csv
    [ -f "$regions/DONE" ] || { echo "342: no finished index $regions" >&2; exit 2; }
    [ -s "$fit" ] || { echo "342: the whole-protein index has no fit for $slug ($fit); no E-values" >&2; exit 4; }
    # Every row of the survival CSV carries the fit's K and r_database.
    read -r KA_K KA_R < <(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
                                   NR == 2 { print $c["k"], $c["r_database"]; exit }' "$fit")
    chunk=$SLURM_ARRAY_TASK_ID
    out=$WORK/calls/$slug/chunk_$(printf %02d "$chunk").calls.tsv.gz
    [ -s "$out" ] && { echo "342: $out done already"; exit 0; }
    mkdir -p "$(dirname "$out")"
    # Query chunk: records i with i mod N_CHUNKS == chunk, so exons and decoys spread evenly.
    q=${TMPDIR:-/tmp}/342.q.$slug.$chunk.fasta
    awk -v n=$N_CHUNKS -v c="$chunk" '/^>/ { r++ } (r - 1) % n == c' "$QUERIES" > "$q"
    echo "342: $slug chunk $chunk, $(grep -c '>' "$q") queries, fit K $KA_K r_database $KA_R"
    # kmerseek writes regions 0-based and end-exclusive; the call file is 1-based and
    # inclusive (start + 1, end as written), as PR 114's call files.
    set +e
    kmerseek search --alphabet "$ALPHABET" --ksize "$KSIZE" --query "$q" --target "$regions" \
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
    for s in "${status[@]}"; do [ "$s" -eq 0 ] || { echo "342: search failed (${status[*]})" >&2; exit 1; }; done
    mv "$out.part" "$out"
    echo "342: $(zcat "$out" | tail -n +2 | wc -l) regions with E <= $MAX_EVALUE"
    ;;
score)
    $PY "$REPO/scripts/342_kind4_features.py" --calls "$WORK/calls" \
        --queries "$REPO/tables/342/342_setA_exon_queries.tsv" \
        --swissprot-dat "$SCRATCH/regions-index-pilot/uniprot_sprot.2026_03.dat.gz" \
        --out "$WORK/score"
    ;;
*)
    echo "342: unknown STEP $STEP" >&2; exit 2
    ;;
esac
