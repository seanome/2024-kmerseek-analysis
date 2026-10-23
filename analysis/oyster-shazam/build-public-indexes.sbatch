#!/bin/bash
#SBATCH --job-name=oyster-public-bivalve-index
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="$SCRATCH/oyster-shazam-20261023"
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
mkdir -p "$ROOT/public-bivalve-indexes"
cd "$ROOT/public-bivalve-indexes"
for MOL in dna protein dayhoff hp; do
 case "$MOL" in dna) K=31; TYPE=DNA;; protein) K=7; TYPE=protein;; dayhoff) K=12; TYPE=dayhoff;; hp) K=30; TYPE=hp;; esac
 "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" - "$MOL" <<'PY'
import csv,json,os,sys
from pathlib import Path
mol=sys.argv[1];root=Path(os.environ['SCRATCH'])/'oyster-shazam-20261023'
with (Path.home()/'oyster-shazam-code/public-bivalve-pilot.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
for scale in (5,100):
 paths=[]
 for r in rows:
  run=r['run_accession'];v=json.loads((root/'public-bivalve-sketches'/f'{run}.validated.json').read_text())
  assert v['input_scope'].startswith('Complete public ENA')
  paths.append(str(root/'public-bivalve-sketches'/mol/f'{run}.scaled{scale}.sig.zip'))
 Path(f'{mol}.scaled{scale}.paths').write_text('\n'.join(paths)+'\n')
PY
 sourmash scripts index "$MOL.scaled5.paths" -o "$MOL.scaled5.rocksdb" -k "$K" -m "$TYPE" --scaled 5 -c 4
 sourmash scripts manysearch "$MOL.scaled100.paths" "$MOL.scaled5.rocksdb" -o "$MOL.self.csv" -k "$K" -m "$TYPE" -s 100 -c 4 -N --ignore-abundance
 "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" - "$MOL" <<'PY'
import csv,json,sys
from pathlib import Path
mol=sys.argv[1]
with (Path.home()/'oyster-shazam-code/public-bivalve-pilot.tsv').open() as f:expected=sorted(r['run_accession'] for r in csv.DictReader(f,delimiter='\t'))
with open(mol+'.self.csv') as f:hits=list(csv.DictReader(f))
found=sorted({r['query_name'] for r in hits if r['query_name']==r['match_name'] and float(r['containment'])==1.0})
if found!=expected:raise ValueError('Public index self-search incomplete')
Path(mol+'.validated.json').write_text(json.dumps(dict(reference_names=expected,scope='Eight complete public ENA bivalve RNA-seq runs from eight studies; initial archive subset, not all 17623 runs'),indent=2)+'\n')
PY
done
