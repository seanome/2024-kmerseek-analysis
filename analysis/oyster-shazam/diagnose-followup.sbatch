#!/bin/bash
#SBATCH --job-name=oyster-followup-diagnosis
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
sourmash scripts manysearch --help
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" - <<'PY'
import os,csv,json,hashlib,pathlib
r=pathlib.Path(os.environ['SCRATCH'])/'oyster-shazam-20261023'
for i in range(22):
 b=r/'search-public-bivalve'/f'batch-{i:03d}'
 if not (b/'validated.json').exists():continue
 for mol in ['dna','protein','dayhoff','hp']:
  maps=[]
  for scale in [10,100]:
   with (b/f'{mol}.scaled{scale}.csv').open() as f: maps.append({(x['query_name'],x['match_name']):x for x in csv.DictReader(f)})
  gained=maps[1].keys()-maps[0].keys()
  if gained: print('GAINED',i,mol,len(gained),[maps[1][k] for k in sorted(gained)[:3]],flush=True)
p=pathlib.Path(os.environ['SCRATCH'])/'oyster-mvp-20261023/splash/full-memory-corrected/oyster.after_correction.scores.tsv'
a=[];rows=0
with p.open() as f:
 for x in csv.DictReader(f,delimiter='\t'):
  rows+=1
  if float(x['pval_opt_corrected'])<0.05:a.append(x['anchor'])
with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
print('FULL_SELECTION',json.dumps(dict(source_sha256=sha,score_rows=rows,selected=len(a),unique=len(set(a)))))
PY
