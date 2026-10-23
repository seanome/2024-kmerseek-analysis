"""Search every extended DNA record in four alphabets against declared indexes."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import sourmash
from sourmash.save_load import SaveSignaturesToLocation

SETTINGS = [('dna',31,{}),('protein',7,{'is_protein':True}),
            ('dayhoff',12,{'dayhoff':True}),('hp',30,{'hp':True})]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    a=p.parse_args()
    source=a.root/'splash-trimmed-all'/f'batch-{a.batch:03d}'
    ext=json.loads((source/'validated.json').read_text())
    path=source/'compactors.tsv'
    if sha256(path) != ext['output_sha256']:
        raise ValueError('Extension output hash differs from validation')
    out=a.root/'search-trimmed-all'/f'batch-{a.batch:03d}'
    out.mkdir(parents=True,exist_ok=True)
    # Refuse an unfinished attempt; no stale result may become a successful search.
    if any(out.iterdir()):
        raise RuntimeError('Search output exists; inspect before retry')
    with path.open() as f:
        rows=[r for r in csv.DictReader(f,delimiter='\t') if len(r['compactor'])>len(r['anchor'])]
    with (out/'queries.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['query_id','anchor','compactor_id','dna','length'])
        for i,r in enumerate(rows):
            r['query_id']=f"splash:all:{a.batch}:{i}:{r['anchor']}"
            w.writerow([r['query_id'],r['anchor'],r['id'],r['compactor'],len(r['compactor'])])
    settings=[]
    for mol,k,flags in SETTINGS:
        index=a.root/'pilot-indexes'/f'{mol}.scaled5.rocksdb'
        validated=json.loads((a.root/'pilot-indexes'/f'{mol}.validated.json').read_text())
        if validated['reference_names'] != ['DW1','DW2','DW3','ZW1','ZW2','ZW3']:
            raise ValueError('Unexpected reference library names')
        sigpath=out/f'{mol}.sig.zip'
        empty={s:0 for s in (5,10,100)}
        with SaveSignaturesToLocation(str(sigpath)) as save:
            for r in rows:
                mh=sourmash.MinHash(n=0,ksize=k,scaled=5,**flags)
                mh.add_sequence(r['compactor'],force=True)
                save.add(sourmash.SourmashSignature(mh,name=r['query_id']))
                for scale in empty:
                    empty[scale]+=not len(mh.downsample(scaled=scale))
        for scale in (5,10,100):
            csvpath=out/f'{mol}.scaled{scale}.csv'
            if rows and empty[scale]<len(rows):
                subprocess.run(['sourmash','scripts','manysearch',str(sigpath),str(index),
                                '-o',str(csvpath),'-k',str(k),'-m','DNA' if mol=='dna' else mol,
                                '-s',str(scale),'-c','4','-A','-N','--ignore-abundance'],check=True)
                with csvpath.open() as f:
                    hits=list(csv.DictReader(f))
                queries={h['query_name'] for h in hits}
                if not queries <= {r['query_id'] for r in rows}:
                    raise ValueError('Search returned an unknown query')
            else:
                csvpath.write_text('query_name,match_name,containment,intersect_hashes\n')
                hits=[];queries=set()
            settings.append(dict(moltype=mol,k=k,scaled=scale,queries=len(rows),
                                 empty_sketches=empty[scale],queries_with_overlap=len(queries),
                                 match_rows=len(hits),output=csvpath.name,
                                 output_sha256=sha256(csvpath)))
    (out/'validated.json').write_text(json.dumps(dict(batch=a.batch,
        input_sha256=ext['output_sha256'],extended_records=len(rows),
        reference_scope='Six oyster first-million-pair trimmed discovery libraries; self-study context only',
        translation='Six-frame sourmash translation of each separate DNA record',
        search_rule='All comparisons with at least one shared hash; candidate occurrence evidence only',
        searches=settings),indent=2)+'\n')


if __name__=='__main__':main()
