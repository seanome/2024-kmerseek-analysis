#!/bin/bash
# Download the Pfam 33.1 (t0), Pfam 38.2 (t1) and UniProt 2019_08 Swiss-Prot files
# for the time-split test (notebook 280). Run on sherlock-dtn. Resumable: rerun to finish.
# Checks each Pfam file against the release's md5_checksums.
set -uo pipefail
DATA=${TIME_SPLIT_DATA:-$SCRATCH/280-time-split/data}
PFAM=https://ftp.ebi.ac.uk/pub/databases/Pfam/releases
UNIPROT=https://ftp.uniprot.org/pub/databases/uniprot

MIRROR=https://bio.nic.funet.fi/pub/mirrors/ftp.sanger.ac.uk/pub/databases/Pfam/releases
NPARTS=${NPARTS:-6}

fetch() {  # url dest [range]: one connection, resumable, restarts a stalled transfer
    local url=$1 dest=$2 range=${3:-}
    # A finished small file: some EBI paths refuse a resume request, so do not ask again.
    [ -z "$range" ] && [ -s "$dest" ] && return 0
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        curl -fsSL -C - ${range:+-r $range} --speed-limit 50000 --speed-time 60 -o "$dest" "$url" && return 0
        echo "retry $attempt: $url $range" >&2; sleep 20
    done
    echo "FAILED: $url $range" >&2; return 1
}

fetch_big() {  # url dest: NPARTS byte ranges in parallel, resumable per range.
    local url=$1 dest=$2
    [ -s "$dest" ] && return 0
    local size; size=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
    local step=$(( (size + NPARTS - 1) / NPARTS )) pids=() i
    for ((i = 0; i < NPARTS; i++)); do
        local a=$(( i * step )) b=$(( (i + 1) * step - 1 ))
        (( b >= size )) && b=$(( size - 1 ))
        # Ranges come from the funet mirror (~6 MB/s per connection against ~1.5 from
        # EBI, measured 2026-10-12); every third attempt falls back to EBI. The md5 check
        # against EBI's md5_checksums covers both.
        local u=$url alt=$url
        [[ $url == $PFAM/* ]] && u=$MIRROR/${url#$PFAM/}
        local want=$(( b - a + 1 ))
        if [ "$(stat -c %s "$dest.part$i" 2>/dev/null || echo 0)" -lt "$want" ]; then
            # A resumed ranged request needs the remaining range, not -C -.
            local have; have=$(stat -c %s "$dest.part$i" 2>/dev/null || echo 0)
            ( for attempt in 1 2 3 4 5 6 7 8 9 10; do
                  have=$(stat -c %s "$dest.part$i" 2>/dev/null || echo 0)
                  (( have >= want )) && exit 0
                  src=$u; (( attempt % 3 == 0 )) && src=$alt
                  curl -fsSL -r $(( a + have ))-$b --speed-limit 50000 --speed-time 60 "$src" >> "$dest.part$i" || sleep 20
              done
              have=$(stat -c %s "$dest.part$i"); (( have == want )) ) &
            pids+=($!)
        fi
    done
    local ok=0
    for p in "${pids[@]}"; do wait "$p" || ok=1; done
    (( ok == 0 )) || { echo "FAILED: $url" >&2; return 1; }
    for ((i = 0; i < NPARTS; i++)); do cat "$dest.part$i"; done > "$dest.tmp" &&
        [ "$(stat -c %s "$dest.tmp")" -eq "$size" ] && mv "$dest.tmp" "$dest" &&
        for ((i = 0; i < NPARTS; i++)); do command rm -f "$dest.part$i"; done
}

for rel in Pfam33.1 Pfam38.2; do
    mkdir -p "$DATA/$rel"
    fetch "$PFAM/$rel/md5_checksums" "$DATA/$rel/md5_checksums.remote"
    small="Pfam-A.hmm.dat.gz Pfam-A.clans.tsv.gz Pfam-A.dead.gz relnotes.txt userman.txt Pfam.version.gz"
    big="Pfam-A.regions.tsv.gz"
    [ "$rel" = Pfam33.1 ] && big="$big pfamseq.gz Pfam-A.regions.uniprot.tsv.gz Pfam-A.hmm.gz"
    for f in $small; do [ -s "$DATA/$rel/$f" ] || fetch "$PFAM/$rel/$f" "$DATA/$rel/$f"; done
    for f in $big; do fetch_big "$PFAM/$rel/$f" "$DATA/$rel/$f" & done
    wait
    files="$small $big"
    (cd "$DATA/$rel" && for f in $files; do grep "  $f\$" md5_checksums.remote; done | md5sum -c -) \
        > "$DATA/$rel/md5_check.txt" 2>&1
done

# Pfam 38.0 pfamseq: the t1 sequences (reference proteomes 2025_03, the same as 38.2,
# which has no pfamseq). Only its MD5s are used, to tell a query that left the reference
# proteomes from one that is still there with no domain.
mkdir -p "$DATA/Pfam38.0"
fetch "$PFAM/Pfam38.0/md5_checksums" "$DATA/Pfam38.0/md5_checksums.remote"
NPARTS=16 fetch_big "$PFAM/Pfam38.0/pfamseq.gz" "$DATA/Pfam38.0/pfamseq.gz"
fetch "$PFAM/Pfam38.0/relnotes.txt" "$DATA/Pfam38.0/relnotes.txt"
(cd "$DATA/Pfam38.0" && grep "  pfamseq.gz\$" md5_checksums.remote | md5sum -c -) > "$DATA/Pfam38.0/md5_check.txt" 2>&1

mkdir -p "$DATA/uniprot_2019_08"
fetch "$UNIPROT/previous_releases/release-2019_08/relnotes.txt" "$DATA/uniprot_2019_08/relnotes.txt"
fetch_big "$UNIPROT/previous_releases/release-2019_08/knowledgebase/uniprot_sprot-only2019_08.tar.gz" \
    "$DATA/uniprot_2019_08/uniprot_sprot-only2019_08.tar.gz"
(cd "$DATA/uniprot_2019_08" && tar tzvf uniprot_sprot-only2019_08.tar.gz > tar_listing.txt)
# Species mnemonic -> kingdom code (A archaea, B bacteria, E eukaryota, V viruses, O other).
fetch "$UNIPROT/current_release/knowledgebase/complete/docs/speclist.txt" "$DATA/uniprot_2019_08/speclist_current.txt"
echo "download done $(date)"
