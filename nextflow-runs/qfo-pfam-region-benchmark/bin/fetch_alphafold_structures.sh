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

# Archive names and the model version are RESOLVED FROM THE SERVER, never hardcoded.
#
# They were hardcoded to v4 and every download 404'd: AFDB is on v6, and the evidence was
# already on disk in the local cache's AF-*-model_v6.cif filenames. A hardcoded version is
# a guess with an expiry date -- v7 would break it again -- so both are looked up once per
# run and the script fails loudly if the lookup itself fails.
#
# species -> UniProt proteome id. Only the id is stable; the filename around it is not.
proteome_id() {
    case "$1" in
        human)       echo "UP000005640" ;;
        mouse)       echo "UP000000589" ;;
        zebrafish)   echo "UP000000437" ;;
        fly)         echo "UP000000803" ;;
        worm)        echo "UP000001940" ;;
        yeast)       echo "UP000002311" ;;
        ecoli)       echo "UP000000625" ;;
        arabidopsis) echo "UP000006548" ;;
        # Confirmed absent from AFDB's proteome archives; fetched per accession.
        chicken|ciona) echo "" ;;
        *)           echo "" ;;
    esac
}

LISTING=""
fetch_listing() {
    [[ -n "$LISTING" ]] && return 0
    LISTING="$(mktemp)"
    curl --fail --silent --show-error --location --max-time 120 "$AFDB_BASE/" \
      | tr '"' '\n' | grep -E '^UP[0-9]+_.*\.tar$' | sort -u > "$LISTING" || {
        echo "!! could not list $AFDB_BASE/ -- cannot resolve archive names" >&2
        exit 1
    }
    [[ -s "$LISTING" ]] || { echo "!! empty archive listing from $AFDB_BASE/" >&2; exit 1; }
}

proteome_archive() {
    local up; up="$(proteome_id "$1")"
    if [[ -z "$up" ]]; then echo ""; return 0; fi
    fetch_listing
    # Newest first, so a directory carrying several versions resolves to the current one.
    grep "^${up}_" "$LISTING" | sort -Vr | head -1
}

# Model version for per-accession fetches, probed once against a real accession rather
# than assumed. Ordered newest first.
MODEL_VERSION=""
resolve_model_version() {
    local probe="$1"
    [[ -n "$MODEL_VERSION" ]] && return 0
    local v
    for v in v6 v5 v4; do
        if curl --fail --silent --head --max-time 30 \
             "$AFDB_FILES/AF-${probe}-F1-model_${v}.cif" >/dev/null 2>&1; then
            MODEL_VERSION="$v"
            echo "  per-accession model version: $MODEL_VERSION"
            return 0
        fi
    done
    echo "!! no model version among v6/v5/v4 resolved for probe accession $probe" >&2
    exit 1
}

# Default target list, restored: an earlier edit replaced the block these lived in and
# deleted them, so TARGETS was never assigned and the script died at its main loop with
# "TARGETS[@]: unbound variable" under set -u. bash -n does not catch that.
ALL_SPECIES=(human mouse chicken zebrafish ciona fly worm yeast arabidopsis ecoli)
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=("${ALL_SPECIES[@]}")
fi

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
    # $archive is the full filename resolved from the listing, .tar included. Appending
    # another .tar here produced _v6.tar.tar and a 404.
    local tar_path="$STRUCT_DIR/_archives/${archive}"

    mkdir -p "$STRUCT_DIR/_archives"
    if [[ ! -f "$tar_path.done" ]]; then
        echo "  downloading ${archive}"
        curl "${CURL_COMMON[@]}" -o "$tar_path" "$AFDB_BASE/${archive}"
        touch "$tar_path.done"
    else
        echo "  archive already downloaded"
    fi

    # AFDB tars hold AF-<acc>-F<n>-model_v<N>.cif.gz. Keep fragment F1 only: the benchmark
    # scores whole-protein domain intervals, and multi-fragment entries are split models
    # of very long proteins whose fragment coordinates do not map onto the full sequence.
    echo "  extracting"
    tar -xf "$tar_path" -C "$dest" --wildcards '*-F1-model_v*.cif.gz' 2>/dev/null || \
        tar -xf "$tar_path" -C "$dest"
    find "$dest" -name '*.cif.gz' -print0 | xargs -0 -P "$PARALLEL" -n 64 gunzip -f

    # Drop every non-F1 fragment explicitly rather than trusting tar to have filtered.
    # GNU tar honours --wildcards; macOS bsdtar does not and silently falls through to the
    # unfiltered extract above, leaving F2+ behind. Those matter: accession_from_tid()
    # parses AF-<acc>-F2-... back to the same accession, so a fragment would be indexed as
    # a second copy of the protein. Fragment coordinates are also relative to the fragment,
    # not the full sequence, so their residue ranges would be wrong against Pfam anyway.
    find "$dest" -name 'AF-*.cif' ! -name 'AF-*-F1*' -delete

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
    resolve_model_version "$(head -1 "$todo")"
    export AFDB_FILES dest MODEL_VERSION
    xargs -a "$todo" -P "$PARALLEL" -I{} bash -c '
        out="$dest/AF-{}-F1.cif"
        curl --fail --silent --location --speed-limit 10240 --speed-time 60 \
             --retry 5 --retry-delay 10 \
             -o "$out.part" "$AFDB_FILES/AF-{}-F1-model_${MODEL_VERSION}.cif" \
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

    archive="$(proteome_archive "$species")"
    if [[ -n "$archive" ]]; then
        fetch_proteome_tar "$species" "$dest" "$archive"
    else
        fetch_per_accession "$species" "$dest" "$acc_file"
    fi

    echo "  $species total structures: $(find "$dest" -name 'AF-*.cif' | wc -l | tr -d ' ')"
done

echo
echo "Done. Structure root: $STRUCT_DIR"
