#!/usr/bin/env bash
#
# Assemble per-species AlphaFold structure directories for the Foldseek arm.
#
# Two sources, because AlphaFold DB does not publish a proteome archive for every
# species in this benchmark:
#
#   proteome tar   human, mouse, zebrafish, fly, worm, yeast, ecoli, arabidopsis
#                  One bulk download each. Far kinder to EBI than tens of thousands of
#                  individual requests, and much faster.
#   per accession  chicken, ciona. Not in AFDB's model-organism proteome set, so each
#                  structure is fetched on its own. ~8.4k requests combined.
#
# Structures already present in the flat local cache are linked, not re-downloaded.
#
# Usage:
#   bin/fetch_alphafold_structures.sh <structure_dir> <accession_list_dir> [species...]
#
# <accession_list_dir> holds <species>.accessions -- one UniProt accession per line,
# written by `make structure-lists`. With no species named, all ten are fetched.
#
# Resumable: re-running skips anything already on disk. A bare backgrounded curl on
# multi-GB EBI files hangs for hours after a silent stall, so every transfer here sets
# --continue-at plus a speed floor that aborts and retries a stalled connection.

set -euo pipefail

STRUCT_DIR="${1:?usage: fetch_alphafold_structures.sh <structure_dir> <accession_list_dir> [species...]}"
ACC_DIR="${2:?missing accession list dir}"
shift 2

FLAT_CACHE="${FLAT_CACHE:-$HOME/data/alphafold_structures}"
AFDB_BASE="https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"
AFDB_FILES="https://alphafold.ebi.ac.uk/files"
PARALLEL="${PARALLEL:-8}"

# Stall detection: abort a transfer holding under 10 KB/s for 120s, then resume it.
CURL_COMMON=(--fail --silent --show-error --location --continue-at -
             --speed-limit 10240 --speed-time 120 --retry 10 --retry-delay 15)

# species -> AFDB proteome archive basename. Empty means no archive; fetch per accession.
declare -A PROTEOME=(
    [human]="UP000005640_9606_HUMAN_v4"
    [mouse]="UP000000589_10090_MOUSE_v4"
    [zebrafish]="UP000000437_7955_DANRE_v4"
    [fly]="UP000000803_7227_DROME_v4"
    [worm]="UP000001940_6239_CAEEL_v4"
    [yeast]="UP000002311_559292_YEAST_v4"
    [ecoli]="UP000000625_83333_ECOLI_v4"
    [arabidopsis]="UP000006548_3702_ARATH_v4"
    [chicken]=""
    [ciona]=""
)

ALL_SPECIES=(human mouse chicken zebrafish ciona fly worm yeast arabidopsis ecoli)
TARGETS=("${@:-${ALL_SPECIES[@]}}")
[[ $# -gt 0 ]] && TARGETS=("$@")

link_cached() {
    # Link what the flat cache already has, so only genuinely missing structures are fetched.
    local species="$1" dest="$2" acc_file="$3"
    local linked=0
    while read -r acc; do
        [[ -z "$acc" ]] && continue
        [[ -e "$dest/AF-${acc}-F1.cif" ]] && continue
        for v in v6 v5 v4; do
            local src="$FLAT_CACHE/AF-${acc}-F1-model_${v}.cif"
            if [[ -f "$src" ]]; then
                ln -sf "$src" "$dest/AF-${acc}-F1.cif"
                linked=$((linked + 1))
                break
            fi
        done
    done < "$acc_file"
    echo "  linked from cache: $linked"
}

fetch_proteome_tar() {
    local species="$1" dest="$2" archive="$3"
    local tar_path="$STRUCT_DIR/_archives/${archive}.tar"

    mkdir -p "$STRUCT_DIR/_archives"
    if [[ ! -f "$tar_path.done" ]]; then
        echo "  downloading ${archive}.tar"
        curl "${CURL_COMMON[@]}" -o "$tar_path" "$AFDB_BASE/${archive}.tar"
        touch "$tar_path.done"
    else
        echo "  archive already downloaded"
    fi

    # AFDB tars hold AF-<acc>-F<n>-model_v4.cif.gz. Keep fragment F1 only: the benchmark
    # scores whole-protein domain intervals, and multi-fragment entries are split models
    # of very long proteins whose fragment coordinates do not map onto the full sequence.
    echo "  extracting"
    tar -xf "$tar_path" -C "$dest" --wildcards '*-F1-model_v4.cif.gz' 2>/dev/null || \
        tar -xf "$tar_path" -C "$dest"
    find "$dest" -name '*.cif.gz' -print0 | xargs -0 -P "$PARALLEL" -n 64 gunzip -f
    # Normalise to the AF-<acc>-F1.cif name the cache links also use.
    for f in "$dest"/AF-*-F1-model_v*.cif; do
        [[ -e "$f" ]] || continue
        mv -f "$f" "$dest/$(basename "$f" | sed -E 's/-model_v[0-9]+//')"
    done
}

fetch_per_accession() {
    local species="$1" dest="$2" acc_file="$3"
    local todo="$dest/.todo"
    : > "$todo"
    while read -r acc; do
        [[ -z "$acc" ]] && continue
        [[ -e "$dest/AF-${acc}-F1.cif" ]] || echo "$acc" >> "$todo"
    done < "$acc_file"

    local n; n=$(wc -l < "$todo" | tr -d ' ')
    echo "  fetching $n structures individually (parallel=$PARALLEL)"
    [[ "$n" -eq 0 ]] && { rm -f "$todo"; return; }

    # A 404 here is expected and not fatal: AlphaFold has no model for every UniProt
    # accession. Those proteins are simply absent from the Foldseek arm, which is the
    # coverage gap the benchmark reports rather than hides.
    export AFDB_FILES dest
    xargs -a "$todo" -P "$PARALLEL" -I{} bash -c '
        out="$dest/AF-{}-F1.cif"
        curl --fail --silent --location --speed-limit 10240 --speed-time 60 \
             --retry 5 --retry-delay 10 \
             -o "$out.part" "$AFDB_FILES/AF-{}-F1-model_v4.cif" \
          && mv "$out.part" "$out" \
          || rm -f "$out.part"
    '
    rm -f "$todo"
}

for species in "${TARGETS[@]}"; do
    acc_file="$ACC_DIR/${species}.accessions"
    [[ -f "$acc_file" ]] || { echo "!! no accession list: $acc_file -- run 'make structure-lists'"; exit 1; }

    dest="$STRUCT_DIR/$species"
    mkdir -p "$dest"
    echo "== $species ($(wc -l < "$acc_file" | tr -d ' ') annotated proteins)"

    link_cached "$species" "$dest" "$acc_file"

    archive="${PROTEOME[$species]:-}"
    if [[ -n "$archive" ]]; then
        fetch_proteome_tar "$species" "$dest" "$archive"
    else
        fetch_per_accession "$species" "$dest" "$acc_file"
    fi

    echo "  $species total structures: $(find "$dest" -name 'AF-*.cif' | wc -l | tr -d ' ')"
done

echo
echo "Done. Structure root: $STRUCT_DIR"
