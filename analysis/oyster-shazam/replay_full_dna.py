"""Replay full-depth SPLASH and DE-kupl DNA queries on frozen public scaled100 shards."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sourmash
from sourmash.save_load import SaveSignaturesToLocation

ROOT=Path(os.environ['GROUP_SCRATCH'])/'oyster-full-public-replay-20261024'
ARCHIVE=Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023'
FULL=Path(os.environ['SCRATCH'])/'oyster-full-extensions-20261023'
DE=Path(os.environ['SCRATCH'])/'oyster-dekupl-queries-20261024'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def command(args,log):
    with log.open('w') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)

def save_query(rows,name,out):
    path=out/f'{name}.sig.zip';total=0;empty=0
    with SaveSignaturesToLocation(str(path)) as save:
        for key,dna in rows:
            if set(dna.upper())-set('ACGTN'):raise ValueError('Invalid query DNA')
            mh=sourmash.MinHash(n=0,ksize=31,scaled=100);mh.add_sequence(dna,force=True)
            save.add(sourmash.SourmashSignature(mh,name=key));total+=1;empty+=not len(mh)
    return dict(file=path.name,queries=total,empty_sketches=empty,sha256=sha(path),query_set=name)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    out=ROOT/'queries'
    out.mkdir()  # Refuse reuse without inspection.
    progress=json.loads((FULL/'summary-progress.json').read_text())
    if progress['source_sha256']!='4312b75c216d9f62767c2457b531e80d0bbb1343a013634c44f850751e8e5f9e':raise ValueError('Wrong selection')
    collections=[]
    for b in progress['completed_batches']:
        p=FULL/'splash-trimmed-all'/f"batch-{b['batch']:03d}"/'compactors.tsv'
        if sha(p)!=b['output_sha256']:raise ValueError('Extension changed')
        with p.open() as f:
            rows=[(f"splash:full:{b['batch']}:{i}:{r['anchor']}",r['compactor']) for i,r in enumerate(csv.DictReader(f,delimiter='\t')) if len(r['compactor'])>len(r['anchor'])]
        entry=save_query(rows,f"splash-full-{b['batch']:03d}",out)
        entry['source_sha256']=b['output_sha256'];collections.append(entry)
    selection=json.loads((DE/'selection.json').read_text())
    if sha(Path(selection['source']))!=selection['source_sha256']:raise ValueError('DE source changed')
    n=0;chunk=[];part=0
    with gzip.open(DE/'all-significant-contigs.tsv.gz','rt') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if not float(r['pvalue'])<.05:raise ValueError('Non-significant DE query')
            chunk.append((r['query_id'],r['contig']));n+=1
            if len(chunk)==10000:
                collections.append(save_query(chunk,f'dekupl-full-{part:03d}',out));chunk=[];part+=1
    if chunk:collections.append(save_query(chunk,f'dekupl-full-{part:03d}',out))
    if n!=selection['selected_contigs']:raise ValueError('Incomplete DE export')
    audit=json.loads((ARCHIVE/'archive-progress.json').read_text())
    runs=sorted(audit['runs'],key=lambda r:r['run'])
    meta=dict(collections=collections,reference_runs=runs,encoding='DNA',k=31,scaled=100,
              full_depth_batches=len(progress['completed_batches']),dekupl_records=n,
              rule='All nonzero shared hashes, explicit -t0 -A',
              limitations='DNA scaled100 only. No scaled10 loss inference; translated stop-free references require rebuilding.')
    (ROOT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(query_records=sum(x['queries'] for x in collections),collections=len(collections),runs=len(runs))))

def search(shard):
    manifest=ROOT/'manifest.json';m=json.loads(manifest.read_text())
    runs=m['reference_runs'][shard*8:(shard+1)*8]
    if not runs:raise ValueError('Empty shard')
    out=ROOT/'dna100'/f'shard-{shard:03d}';out.mkdir(parents=True)
    paths=[];names={r['run'] for r in runs}
    for r in runs:
        p=ARCHIVE/'runs'/r['run'][:6]/r['run']/'dna.sig.zip'
        if sha(p)!=r['signatures']['dna']['sha256']:raise ValueError('Reference changed')
        paths.append(p)
    pathlist=out/'reference-paths.txt';pathlist.write_text(''.join(str(p)+'\n' for p in paths))
    index=out/'dna.rocksdb'
    command(['sourmash','scripts','index',pathlist,'-o',index,'-k','31','-m','DNA','--scaled','100','-c','4'],out/'index.log')
    results=[]
    for q in m['collections']:
        query=ROOT/'queries'/q['file']
        if sha(query)!=q['sha256']:raise ValueError('Query changed')
        dest=out/f"{q['query_set']}.csv";log=out/f"{q['query_set']}.log"
        command(['sourmash','scripts','manysearch',query,index,'-o',dest,'-k','31','-m','DNA','-s','100','-t','0','-A','-N','--ignore-abundance','-c','4'],log)
        done=re.findall(r'DONE\. Processed ([\d,]+) search sigs',log.read_text())
        if not done or int(done[-1].replace(',',''))!=q['queries']:raise ValueError('Incomplete query processing')
        expected_names={sig.name for sig in sourmash.load_file_as_signatures(str(query))}
        if len(expected_names)!=q['queries']:raise ValueError('Duplicate query IDs')
        pairs=set();count=0
        with dest.open() as f:
            for row in csv.DictReader(f):
                if row['match_name'] not in names or row['query_name'] not in expected_names or int(row['intersect_hashes'])<1:raise ValueError('Invalid hit')
                key=(row['query_name'],row['match_name'])
                if key in pairs:raise ValueError('Duplicate pair')
                pairs.add(key);count+=1
        packed=dest.with_suffix('.csv.gz')
        with dest.open('rb') as src,gzip.open(packed,'wb') as target:
            import shutil
            shutil.copyfileobj(src,target)
        dest.unlink()
        results.append(dict(query_set=q['query_set'],queries=q['queries'],empty_sketches=q['empty_sketches'],match_pairs=count,file=packed.name,sha256=sha(packed)))
    (out/'validated.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),references=sorted(names),encoding='DNA',scaled=100,results=results),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','search']);p.add_argument('--shard',type=int);a=p.parse_args()
    prepare() if a.mode=='prepare' else search(a.shard)
