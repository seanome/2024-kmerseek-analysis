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
# Resumable: re-running skips anything already on disk -- a species whose proteome
# archive is already unpacked is skipped without re-extracting it (FORCE_EXTRACT=1
# overrides), and the per-accession species fetch only what is missing. A bare backgrounded curl on
# multi-GB EBI files hangs for hours after a silent stall, so every transfer here sets
# --continue-at plus a speed floor that aborts and retries a stalled connection.

set -euo pipefail

STRUCT_DIR="${1:?usage: fetch_alphafold_structures.sh <structure_dir> <accession_list_dir> [species...]}"
ACC_DIR="${2:?missing accession list dir}"
shift 2

FLAT_CACHE="${FLAT_CACHE:-$HOME/data/alphafold_structures}"
AFDB_BASE="https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"
# Proteome archives come over rsync, which resumes a partial transfer natively. These are
# ~1-25 GB each and a restart-from-zero on a dropped connection is the difference between
# an interrupted download and a lost afternoon. The HTTPS base above is still used for the
# directory listing and for per-accession fetches, which rsync does not serve.
AFDB_RSYNC="rsync://ftp.ebi.ac.uk/pub/databases/alphafold/latest"
AFDB_FILES="https://alphafold.ebi.ac.uk/files"
PARALLEL="${PARALLEL:-8}"

# Stall detection: abort a transfer holding under 10 KB/s for 120s, then resume it.
CURL_COMMON=(--fail --silent --show-error --location --continue-at -
             --speed-limit 10240 --speed-time 120 --retry 10 --retry-delay 15)

# Archive names and the model version are RESOLVED FROM THE SERVER, never hardcoded.
#
# They were hardcoded to v4 and every download 404'd: AFDB is on v6, and the evidence was
# already on disk in the local cache's AF-*-model_v6.cif filenames. A hardcoded version is
# a guess with an expiry date. v7 would break it again. so both are looked up once per
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

# Model version for per-accession fetches, probed against real accessions rather than
# assumed. Ordered newest first.
#
# Probes SEVERAL accessions, not one. AlphaFold has no model for every UniProt accession --
# the fetch loop below says so itself and treats a 404 there as expected -- but this probe
# used only the first accession in the list and exited 1 when it missed. One absent chicken
# entry (A0A165EAY6) therefore killed a ten-species download in which chicken itself was
# fully available: 12 of 12 other chicken accessions resolve to v6.
PROBE_LIMIT="${PROBE_LIMIT:-12}"
MODEL_VERSION=""
resolve_model_version() {
    local todo="$1"
    [[ -n "$MODEL_VERSION" ]] && return 0
    local v acc tried=0
    while read -r acc; do
        [[ -z "$acc" ]] && continue
        tried=$((tried + 1))
        for v in v6 v5 v4; do
            if curl --fail --silent --head --max-time 30 \
                 "$AFDB_FILES/AF-${acc}-F1-model_${v}.cif" >/dev/null 2>&1; then
                MODEL_VERSION="$v"
                echo "  per-accession model version: $MODEL_VERSION (resolved on $acc after $tried probe(s))"
                return 0
            fi
        done
        [[ "$tried" -ge "$PROBE_LIMIT" ]] && break
    done < "$todo"
    echo "!! no model version among v6/v5/v4 resolved after $tried probe accessions" >&2
    return 1
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
    # Link what the flat cache already has, so only missing structures are fetched.
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
    local species="$1" dest="$2" archive="$3" acc_file="$4"
    # $archive is the full filename resolved from the listing, .tar included. Appending
    # another .tar here produced _v6.tar.tar and a 404.
    local tar_path="$STRUCT_DIR/_archives/${archive}"
    local marker="$dest/.extracted"

    # Skip a species that is already unpacked.
    #
    # The `.done` marker below only ever guarded the DOWNLOAD. Extraction ran every time:
    # untar ~25 GB, gunzip every member, delete the PDB copies, rename every file. So
    # re-running `make fetch-structures` to pick up one missing species redid human from
    # scratch first, which is what this is fixing. The per-accession path never had the
    # problem because it builds a .todo of what is actually absent.
    # Signatures of an unfinished run. Computed BEFORE the marker is trusted, not only in
    # the adoption path below, because a marker can itself be wrong: the first version of
    # this skip adopted on a file count alone and stamped human as complete while 20_539
    # .cif.gz files were still sitting unread in its directory. A marker is a claim about
    # the directory, so the directory gets the last word.
    #
    #   *.cif.gz            the untar or the gunzip did not finish
    #   *.pdb.gz            the PDB cleanup did not run. The wildcard untar normally keeps
    #                       these out, but the no-wildcard fallback extracts them, and
    #                       leaving them gives Foldseek and Folddisco two files per protein
    #                       so every structure is counted twice.
    #   AF-*-model_v*.cif   the rename did not finish, and the fragment-offset normalizers
    #                       downstream key on the AF-<acc>-F<n>.cif form.
    local leftovers
    leftovers=$(find "$dest" \( -name '*.cif.gz' -o -name '*.pdb.gz' \
                               -o -name 'AF-*-model_v*.cif' \) -print -quit)

    if [[ -f "$marker" && "$(cat "$marker" 2>/dev/null)" == "$archive" ]]; then
        if [[ -z "$leftovers" ]]; then
            echo "  already extracted ($archive)"
            return 0
        fi
        echo "  marker says extracted but $(basename "$leftovers") is still here --"
        echo "  the directory is unfinished; re-extracting and rewriting the marker"
        rm -f "$marker"
    fi

    # Adopt an extraction that predates the marker, so this fix does not itself cost one
    # last full re-extract of everything already on disk. Only when the archive downloaded
    # completely AND the directory holds at least as many .cif files as the species has
    # annotated accessions -- a real lower bound, since the AFDB proteome archives cover
    # the whole reference proteome and add fragments on top, while the accession list is
    # only its annotated subset. An interrupted extraction lands under that bound and is
    # redone. Set FORCE_EXTRACT=1 to re-extract regardless.
    # Adopt an extraction that predates the marker, so introducing the marker did not cost
    # one last full re-extract of everything already on disk. Requires the archive to have
    # downloaded completely, no leftovers above, and at least as many .cif files as the
    # species has annotated accessions -- a real lower bound, since AFDB's proteome archives
    # cover the whole reference proteome and add fragments on top while the accession list
    # is only its annotated subset. A count ALONE is not enough: `AF-*.cif` matches the
    # un-renamed name too, so a directory interrupted partway through gunzip can pass the
    # count while being unusable. That is how human got mis-marked.
    if [[ -z "${FORCE_EXTRACT:-}" && -f "$tar_path.done" && -z "$leftovers" ]]; then
        local have want
        have=$(find "$dest" -name 'AF-*.cif' | wc -l | tr -d ' ')
        want=$(grep -c . "$acc_file" | tr -d ' ')
        if [[ "$have" -ge "$want" && "$want" -gt 0 ]]; then
            echo "  already extracted ($have cif >= $want annotated accessions); marking"
            echo "$archive" > "$marker"
            return 0
        fi
    elif [[ -n "$leftovers" ]]; then
        echo "  partial extraction detected ($(basename "$leftovers")) -- re-extracting"
    fi

    mkdir -p "$STRUCT_DIR/_archives"
    if [[ ! -f "$tar_path.done" ]]; then
        echo "  downloading ${archive} (rsync, resumable)"
        rsync -P "$AFDB_RSYNC/${archive}" "$tar_path"
        touch "$tar_path.done"
    else
        echo "  archive already downloaded"
    fi

    # AFDB tars hold AF-<acc>-F<n>-model_v<N>.cif.gz. KEEP EVERY FRAGMENT. Proteins over
    # 2700 aa are modelled only as overlapping 1400-residue fragments with a 200-residue
    # stride, so an F1-only extract silently drops every structure for the longest proteins
    # in the proteome -- 3400 fragment files in the human archive alone, and with them all
    # 3469 human Pfam domain instances (6.9%) that sit on proteins over 2700 aa.
    #
    # Fragment residue numbering is fragment-local: AF-A0A087WUL8-F2 numbers its residues
    # 1..1400 while its SIFTS xref shows UniProt 201..1600. Every downstream normalizer
    # therefore adds (n-1)*200 to a hit on F<n> before the interval means anything against
    # a full-sequence Pfam annotation -- see af_offset() in bin/normalize_reseek.awk,
    # foldseekSearch in main.nf, and accession_and_offset() in bin/folddisco_to_regions.py.
    echo "  extracting"
    tar -xf "$tar_path" -C "$dest" --wildcards '*-model_v*.cif.gz' 2>/dev/null || \
        tar -xf "$tar_path" -C "$dest"
    find "$dest" -name '*.cif.gz' -print0 | xargs -0 -P "$PARALLEL" -n 64 gunzip -f

    # AFDB proteome tars ship BOTH .cif.gz and .pdb.gz for every entry. Only the mmCIF is
    # wanted: leaving the PDB copies behind doubles disk on a ~60 GB download and, worse,
    # gives Foldseek and Folddisco two files per protein to index, so every structure is
    # counted twice. Measured on the real yeast archive: 6055 .cif alongside 6055 .pdb.gz.
    find "$dest" -name 'AF-*' ! -name '*.cif' ! -name '*.cif.gz' -delete

    # Normalise to AF-<acc>-F<n>.cif, keeping the fragment number: it is the only thing
    # that tells a downstream normalizer which offset to apply.
    for f in "$dest"/AF-*-F[0-9]*-model_v*.cif; do
        [[ -e "$f" ]] || continue
        mv -f "$f" "$dest/$(basename "$f" | sed -E 's/-model_v[0-9]+//')"
    done

    # Written only after every step above succeeded, so an interrupted run leaves no marker
    # and the next one redoes the extraction rather than trusting a half-unpacked directory.
    echo "$archive" > "$marker"
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
    # accession. Those proteins are absent from the Foldseek arm, which is the
    # coverage gap the benchmark reports rather than hides.
    # A species AFDB does not cover at all must not abort the other nine: the structure
    # arms already report per-species coverage, and a missing species is a coverage gap to
    # report, not a reason to lose an in-progress multi-species download.
    if ! resolve_model_version "$todo"; then
        echo "!! skipping per-accession fetch for $species -- AlphaFold resolved no model" >&2
        echo "   for any of its first $PROBE_LIMIT accessions. Other species continue." >&2
        rm -f "$todo"
        return 0
    fi
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
        fetch_proteome_tar "$species" "$dest" "$archive" "$acc_file"
    else
        fetch_per_accession "$species" "$dest" "$acc_file"
    fi

    echo "  $species total structures: $(find "$dest" -name 'AF-*.cif' | wc -l | tr -d ' ')"
done

echo
echo "Done. Structure root: $STRUCT_DIR"
