"""Independently check the completed trimmed archive pilot before releasing workers."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


root=Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023'
out=root/'runs/DRR350/DRR350416'
meta=json.loads((out/'validated.json').read_text())
queries=json.loads((root/'queries/validated.json').read_text())
if not (meta['scales']==[10,100] and meta['containment_threshold']==0.0):raise ValueError('Independent archive validation failed')
if not (meta['query_source_sha256']==queries['source_sha256']):raise ValueError('Independent archive validation failed')
checks=[]
for mol,k in [('dna',31),('protein',7),('dayhoff',12),('hp',30)]:
    sig=out/f'{mol}.sig.zip'
    if not (sha(sig)==meta['signatures'][mol]['sha256']):raise ValueError('Independent archive validation failed')
    pairs={}
    for scale in [10,100]:
        q=next(s for s in queries['searches'] if s['moltype']==mol and s['scaled']==scale)
        if not (sha(root/'queries'/f'{mol}.scaled{scale}.sig.gz')==q['sha256']):raise ValueError('Independent archive validation failed')
        result=next(s for s in meta['searches'] if s['moltype']==mol and s['scaled']==scale)
        path=out/f'{mol}.scaled{scale}.matches.csv.gz'
        if not (sha(path)==result['result_sha256']):raise ValueError('Independent archive validation failed')
        with gzip.open(path,'rt') as f:
            rows=list(csv.DictReader(f))
        names={r['query_name'] for r in rows}
        if not (all(r['match_name']==meta['run'] and float(r['intersect_hashes'])>0 for r in rows)):raise ValueError('Independent archive validation failed')
        if not (len(names)==len(rows)==result['match_rows']):raise ValueError('Independent archive validation failed')
        pairs[scale]=names
    if not (not pairs[100]-pairs[10]):raise ValueError('Independent archive validation failed')
    comp=next(x for x in meta['comparisons'] if x['moltype']==mol)
    loss=out/f'{mol}.lost-at-scaled100.tsv.gz'
    if not (sha(loss)==comp['loss_sha256']):raise ValueError('Independent archive validation failed')
    with gzip.open(loss,'rt') as f:
        lost={r['query_id'] for r in csv.DictReader(f,delimiter='\t')}
    if not (lost==pairs[10]-pairs[100] and len(lost)==comp['lost_matches']):raise ValueError('Independent archive validation failed')
    self_csv=out/f'{mol}.independent-self.csv'
    subprocess.run(['sourmash','scripts','manysearch',str(sig),str(out/f'{mol}.rocksdb'),'-o',str(self_csv),'-k',str(k),'-m','DNA' if mol=='dna' else mol,'-s','100','-c','4','-N','--ignore-abundance'],check=True)
    with self_csv.open() as f:
        hits=list(csv.DictReader(f))
    if not (len(hits)==1 and hits[0]['query_name']==hits[0]['match_name']==meta['run'] and float(hits[0]['containment'])==1.0):raise ValueError('Independent archive validation failed')
    checks.append(dict(moltype=mol,pairs10=len(pairs[10]),pairs100=len(pairs[100]),lost=len(lost),self_containment=1.0))
(root/'independent-pilot-validation.json').write_text(json.dumps(dict(run=meta['run'],scales=[10,100],checks=checks),indent=2)+'\n')
print('Independent archive checks passed',json.dumps(checks),flush=True)
