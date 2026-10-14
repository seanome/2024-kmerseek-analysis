#!/bin/bash
# Download Pfam 38.2 Pfam-A.regions.tsv.gz in N parallel byte ranges (EBI gives ~150 KB/s
# per connection), check it against the release md5, keep only the Swiss-Prot rows, and
# delete the 5 GB file. Usage: download_pfam_regions_parallel.sh OUT_DIR ACCESSIONS_TXT [N]
set -euo pipefail
OUT=$1; ACC=$2; N=${3:-32}
URL=https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.regions.tsv.gz
EXPECTED_MD5=f22fb1efefd1e52d6d9c64a8f10003ba
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/parts"
SIZE=$(curl -sI "$URL" | awk 'tolower($1)=="content-length:"{print $2+0}')
STEP=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*STEP)); e=$(( s+STEP-1 < SIZE-1 ? s+STEP-1 : SIZE-1 ))
  ( until [ "$(stat -f %z "$OUT/parts/$i" 2>/dev/null || echo 0)" -eq $((e-s+1)) ]; do
      have=$(stat -f %z "$OUT/parts/$i" 2>/dev/null || echo 0)
      curl -sS --speed-limit 5000 --speed-time 60 -r $((s+have))-$e "$URL" >> "$OUT/parts/$i" || sleep 5
    done ) &
done
wait
for i in $(seq 0 $((N-1))); do cat "$OUT/parts/$i"; done > "$OUT/Pfam-A.regions.tsv.gz"
GOT=$(md5 -q "$OUT/Pfam-A.regions.tsv.gz")
echo "size=$SIZE md5=$GOT expected=$EXPECTED_MD5"
[ "$GOT" = "$EXPECTED_MD5" ]
/bin/rm -rf "$OUT/parts"
gzip -dc "$OUT/Pfam-A.regions.tsv.gz" | /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 -I "$HERE/filter_pfam_regions_swissprot.py" "$ACC" "$OUT/Pfam-A.regions.swissprot_2026_03.tsv"
/bin/rm -f "$OUT/Pfam-A.regions.tsv.gz"
