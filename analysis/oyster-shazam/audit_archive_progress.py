import os,json,hashlib,gzip,csv,collections
from pathlib import Path
root=Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023'
records=[];failures=collections.Counter()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p in root.glob('runs/*/*/validated.json'):
 v=json.loads(p.read_text());out=p.parent
 if v.get('containment_threshold')!=0 or v['scales']!=[10,100]:raise ValueError('Protocol mismatch')
 for mol,s in v['signatures'].items():
  if sha(out/f'{mol}.sig.zip')!=s['sha256']:raise ValueError('Signature changed')
 for s in v['searches']:
  path=out/f"{s['moltype']}.scaled{s['scaled']}.matches.csv.gz"
  if sha(path)!=s['result_sha256']:raise ValueError('Results changed')
 for c in v['comparisons']:
  if sha(out/f"{c['moltype']}.lost-at-scaled100.tsv.gz")!=c['loss_sha256']:raise ValueError('Loss table changed')
 records.append(v)
for p in root.glob('worker-49241661-*.log'):
 lines=p.read_text(errors='replace').splitlines()
 for line in lines:
  if line.startswith(('TypeError:','RuntimeError:','ValueError:','subprocess.CalledProcessError:')):failures[line[:350]]+=1
out=dict(validated_runs=len(records),validated_studies=len({v['metadata']['study_accession'] for v in records}),runs=records,failure_messages=dict(failures),verification='Rehashed retained signatures, both match tables and loss tables for every completion record; no partial directory counted',snapshot='2026-10-24')
(root/'archive-progress.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='runs'}))
