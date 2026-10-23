#!/bin/bash
#SBATCH --job-name=oyster-pilot-branchwater
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="$SCRATCH/oyster-shazam-20261023"
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/test_search_extended.py"
mkdir -p "$ROOT/pilot-indexes"
cd "$ROOT/pilot-indexes"
for SAMPLE in DW1 DW2 DW3 ZW1 ZW2 ZW3; do test -s "$ROOT/index-pilot/$SAMPLE.validated.json"; done
for MOL in dna protein dayhoff hp; do
 case "$MOL" in dna) K=31; TYPE=DNA;; protein) K=7; TYPE=protein;; dayhoff) K=12; TYPE=dayhoff;; hp) K=30; TYPE=hp;; esac
 LIST="$MOL.scaled5.paths.txt"
 for SAMPLE in DW1 DW2 DW3 ZW1 ZW2 ZW3; do printf '%s\n' "$ROOT/index-pilot/$MOL/$SAMPLE.scaled5.sig.zip"; done > "$LIST"
 # The prior DNA build completed before its manifest-export command failed.
 # Reuse it only after exporting and checking its actual manifest and self-search.
 if ! test -d "$MOL.scaled5.rocksdb"; then
  /usr/bin/time -v sourmash scripts index "$LIST" -o "$MOL.scaled5.rocksdb" -k "$K" -m "$TYPE" --scaled 5 -c 4
 fi
 sourmash sig manifest "$MOL.scaled5.rocksdb" --no-rebuild-manifest -o "$MOL.manifest.csv"
 sourmash scripts manysearch "$LIST" "$MOL.scaled5.rocksdb" -o "$MOL.self.csv" -k "$K" -m "$TYPE" -s 5 -c 4 -N --ignore-abundance
 "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" - "$MOL" <<'PY'
import csv,json,sys
from pathlib import Path
mol=sys.argv[1]
with open(mol+'.manifest.csv') as f:
 rows=list(csv.DictReader(line for line in f if not line.startswith('#')))
names=sorted(r['name'] for r in rows)
assert names==['DW1','DW2','DW3','ZW1','ZW2','ZW3'],names
with open(mol+'.self.csv') as f: hits=list(csv.DictReader(f))
assert {r['query_name'] for r in hits if r['query_name']==r['match_name'] and float(r['containment'])==1.0}==set(names)
Path(mol+'.validated.json').write_text(json.dumps(dict(reference_names=names,scope='Six trimmed oyster pilot libraries; discovery study',self_search='Each run contains its own complete sketch'),indent=2)+'\n')
PY
done
