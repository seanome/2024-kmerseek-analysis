"""Trim complete public runs, compare scaled10 and scaled100 sketches, and search each immediately."""
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import sourmash
from sourmash.save_load import SaveSignaturesToLocation

SETTINGS=[('dna',31,{}),('protein',7,{'is_protein':True}),('dayhoff',12,{'dayhoff':True}),('hp',30,{'hp':True})]
CODE=Path(__file__).parent
ROOT=Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023'
DISCOVERY=Path(os.environ['SCRATCH'])/'oyster-shazam-20261023'


def sha(path,algorithm='sha256'):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()


def run(args,log):
    with Path(log).open('w') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)


def prepare():
    source=DISCOVERY/'splash-trimmed-all/all-extended-sequences.tsv.gz'
    expected=json.loads((DISCOVERY/'splash-trimmed-all/summary.json').read_text())
    if sha(source)!=expected['download_sha256']:raise ValueError('Extended sequence export changed')
    out=ROOT/'queries';out.mkdir(parents=True,exist_ok=True)
    if (out/'validated.json').exists():
        prior=json.loads((out/'validated.json').read_text())
        if prior['source_sha256']!=sha(source):raise ValueError('Cached query source changed')
        if {(s['moltype'],s['scaled']) for s in prior['searches']} != {(m,k) for m,_,_ in SETTINGS for k in (10,100)}:raise ValueError('Missing cached setting')
        for entry in prior['searches']:
            if sha(out/f"{entry['moltype']}.scaled{entry['scaled']}.sig.gz")!=entry['sha256']:raise ValueError('Cached query sketch changed')
        print('Verified cached paired query sketches',flush=True)
        return
    with gzip.open(source,'rt') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    if len(rows)!=expected['extended_records']:raise ValueError('Query count changed')
    stats=[]
    for mol,k,flags in SETTINGS:
        paths={scale:out/f'{mol}.scaled{scale}.sig.gz' for scale in (10,100)}
        empty={scale:0 for scale in paths}
        with SaveSignaturesToLocation(str(paths[10])) as save10, SaveSignaturesToLocation(str(paths[100])) as save100:
            for r in rows:
                mh=sourmash.MinHash(n=0,ksize=k,scaled=10,**flags);mh.add_sequence(r['dna'],force=True)
                name=f"all:{r['batch']}:{r['anchor']}:{r['compactor_id']}"
                for scale,save in [(10,save10),(100,save100)]:
                    sketch=mh.downsample(scaled=scale)
                    empty[scale]+=not len(sketch)
                    save.add(sourmash.SourmashSignature(sketch,name=name))
        for scale,path in paths.items():
            stats.append(dict(moltype=mol,k=k,scaled=scale,queries=len(rows),empty_sketches=empty[scale],sha256=sha(path)))
        print('Prepared paired queries',mol,flush=True)
    (out/'validated.json').write_text(json.dumps(dict(source_sha256=sha(source),searches=stats),indent=2)+'\n')


def process(row):
    accession=row['run_accession'];out=ROOT/'runs'/accession[:6]/accession
    if (out/'validated.json').exists():
        prior=json.loads((out/'validated.json').read_text())
        if prior['metadata']!=row or prior['scales']!=[10,100]:raise ValueError('Checkpoint protocol changed')
        for mol,_,_ in SETTINGS:
            if sha(out/f'{mol}.sig.zip')!=prior['signatures'][mol]['sha256']:raise ValueError('Sketch changed')
        return prior
    if out.exists():raise RuntimeError('Incomplete prior run; inspect before retry')
    out.mkdir(parents=True)
    urls=row['fastq_ftp'].split(';');md5s=row['fastq_md5'].split(';')
    if len(urls)!=len(md5s) or not all(md5s):raise ValueError('Missing source MD5')
    size=sum(map(int,row['fastq_bytes'].split(';')))
    local=Path(os.environ['L_SCRATCH_JOB'])
    if shutil.disk_usage(local).free<size*3:
        local=ROOT/'temporary';local.mkdir(exist_ok=True)
    if shutil.disk_usage(ROOT).free<5_000_000_000_000:raise RuntimeError('Less than 5 TB storage headroom; stop for review')
    with tempfile.TemporaryDirectory(prefix=accession+'-',dir=local) as temp:
        temp=Path(temp);trimmed=[];sources=[]
        for i,(url,md5) in enumerate(zip(urls,md5s)):
            if not url.startswith('ftp.sra.ebi.ac.uk/'):raise ValueError('Unexpected ENA host')
            raw=temp/f'reads-{i}.fastq.gz'
            run(['curl','--fail','--location','--retry','5','--retry-delay','10','--speed-limit','1024','--speed-time','60','--continue-at','-','--output',raw,'https://'+url],out/f'download-{i}.log')
            if sha(raw,'md5')!=md5:raise ValueError('ENA MD5 mismatch')
            sources.append(dict(url='https://'+url,md5=md5,sha256=sha(raw)))
            # Each listed FASTQ is trimmed independently; mates/records are never joined.
            run(['trim_galore','--cores','1','--quality','20','--length','20','--gzip','--output_dir',temp,raw],out/f'trim-{i}.log')
            result=temp/f'reads-{i}_trimmed.fq.gz'
            if not result.exists():raise RuntimeError('Missing trimmed FASTQ')
            trimmed.append(result)
            for report in temp.glob('*trimming_report.txt'):shutil.copy2(report,out/report.name)
        def sketch(setting):
            mol,k,_=setting;dense=temp/f'{mol}.scaled10.sig.zip';sparse=out/f'{mol}.sig.zip'
            run(['sourmash','sketch','dna' if mol=='dna' else 'translate',*trimmed,'--merge',accession,'-p',f'{mol},k={k},scaled=10','-o',dense],out/f'{mol}.sketch.log')
            sigs=list(sourmash.load_file_as_signatures(str(dense)))
            if len(sigs)!=1:raise ValueError('Expected one run signature')
            mh=sigs[0].minhash
            if mh.ksize!=k or mh.scaled!=10 or mh.moltype.lower()!=mol:raise ValueError('Wrong signature setting')
            with SaveSignaturesToLocation(str(sparse)) as save:
                save.add(sourmash.SourmashSignature(mh.downsample(scaled=100),name=accession))
            return mol,dict(sha256=sha(sparse),bytes=sparse.stat().st_size,hashes=len(mh.downsample(scaled=100)),scaled10_bytes=dense.stat().st_size,scaled10_sha256=sha(dense))
        # Bound simultaneous dense signature memory. Four hashing jobs are not required.
        signatures=dict(sketch(setting) for setting in SETTINGS)
        searches=[];comparisons=[];query_meta=json.loads((ROOT/'queries/validated.json').read_text())
        for mol,k,_ in SETTINGS:
            pair_sets={};empty100=set()
            for scale in (10,100):
                query=ROOT/'queries'/f'{mol}.scaled{scale}.sig.gz'
                expected=next(x for x in query_meta['searches'] if x['moltype']==mol and x['scaled']==scale)
                if sha(query)!=expected['sha256']:raise ValueError('Query sketches changed')
                if scale==100:
                    empty100={sig.name for sig in sourmash.load_file_as_signatures(str(query)) if not len(sig.minhash)}
                target=temp/f'{mol}.scaled10.sig.zip' if scale==10 else out/f'{mol}.sig.zip'
                index=temp/f'{mol}.scaled10.rocksdb' if scale==10 else out/f'{mol}.rocksdb'
                paths=temp/f'{mol}.paths';paths.write_text(str(target)+'\n')
                run(['sourmash','scripts','index',paths,'-o',index,'-k',k,'-m','DNA' if mol=='dna' else mol,'--scaled',str(scale),'-c','4'],out/f'{mol}.scaled{scale}.index.log')
                csvpath=temp/f'{mol}.scaled{scale}.csv'
                run(['sourmash','scripts','manysearch',query,index,'-o',csvpath,'-k',k,'-m','DNA' if mol=='dna' else mol,'-s',str(scale),'-t','0','-c','4','-A','-N','--ignore-abundance'],out/f'{mol}.scaled{scale}.search.log')
                names=set();count=0;dest=out/f'{mol}.scaled{scale}.matches.csv.gz'
                with csvpath.open() as src,gzip.open(dest,'wt') as saved:
                    reader=csv.DictReader(src);writer=csv.DictWriter(saved,fieldnames=reader.fieldnames);writer.writeheader()
                    for hit in reader:
                        if hit['match_name']!=accession:raise ValueError('Wrong matched run')
                        names.add(hit['query_name']);count+=1;writer.writerow(hit)
                pair_sets[scale]=names
                searches.append(dict(moltype=mol,k=k,scaled=scale,queries=expected['queries'],empty_sketches=expected['empty_sketches'],queries_with_overlap=len(names),match_rows=count,result_sha256=sha(dest)))
            lost=pair_sets[10]-pair_sets[100];gained=pair_sets[100]-pair_sets[10]
            if gained:raise ValueError('Unexpected gained matches in nested all-overlap search')
            loss_path=out/f'{mol}.lost-at-scaled100.tsv.gz'
            with gzip.open(loss_path,'wt') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['query_id','run','reason'])
                for q in sorted(lost):w.writerow([q,accession,'empty query sketch' if q in empty100 else 'no retained shared hash'])
            comparisons.append(dict(moltype=mol,lost_matches=len(lost),retained_matches=len(pair_sets[100]),lost_empty_queries=len(lost&empty100),loss_sha256=sha(loss_path)))
        info=dict(run=accession,metadata=row,sources=sources,scales=[10,100],containment_threshold=0.0,signatures=signatures,searches=searches,comparisons=comparisons,
                  retention='Scaled100 signature ZIPs and RocksDB retained; scaled10 signature/index temporary; both match tables and exact loss IDs retained',
                  preprocessing='Trim Galore 0.6.10, Q20, length20; each listed FASTQ independently; no SPLASH or DE-kupl',query_source_sha256=query_meta['source_sha256'])
        (out/'validated.json').write_text(json.dumps(info,indent=2)+'\n')
        print('VALIDATED PUBLIC MATCHES',accession,flush=True)
        return info


def worker(identifier):
    with gzip.open(CODE/'public-bivalve-all.tsv.gz','rt') as f:rows={r['run_accession']:r for r in csv.DictReader(f,delimiter='\t')}
    plan=json.loads((CODE/'archive-worker-plan.json').read_text())
    selected=[identifier] if identifier in rows else sorted(plan['workers'][identifier]['runs'],key=lambda run:sum(map(int,rows[run]['fastq_bytes'].split(';'))))
    start=time.monotonic();limit=float(os.environ.get('ARCHIVE_WORKER_HOURS','44'))*3600
    completed=[];deferred=[]
    for accession in selected:
        size=sum(map(int,rows[accession]['fastq_bytes'].split(';')))
        # Checkpoint between complete runs; do not knowingly start beyond the allocation.
        predicted=0 if (ROOT/'runs'/accession[:6]/accession/'validated.json').exists() else size/1e6*12+1800
        if time.monotonic()-start+predicted>limit:
            deferred.append(accession)
            continue
        process(rows[accession]);completed.append(accession)
    progress=ROOT/'workers';progress.mkdir(exist_ok=True)
    (progress/f'{identifier}.json').write_text(json.dumps(dict(completed=completed,deferred=deferred),indent=2)+'\n')


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    if sys.argv[1]=='prepare':prepare()
    else:worker(sys.argv[1])
