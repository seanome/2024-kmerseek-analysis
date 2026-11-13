"""Verify completed full-depth extension batches without treating pending batches as zero."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def audit(root):
    selection=json.loads((root/'selection.json').read_text())
    completed=[]
    for p in sorted((root/'splash-trimmed-all').glob('batch-*/validated.json')):
        meta=json.loads(p.read_text());batch=p.parent
        if meta['total_selected']!=selection['selected'] or meta['batch_size']!=10000 or batch.name!=f"batch-{meta['batch']:03d}":
            raise ValueError('Invalid full-depth batch identity')
        if meta['source_sha256']!=selection['source_sha256'] or meta['threshold']!=.05 or meta['comparison']!='<':
            raise ValueError('Wrong full-depth source or selection rule')
        with (batch/'selected_anchors.tsv').open() as f:
            anchors=[r['anchor'] for r in csv.DictReader(f,delimiter='\t')]
        if len(anchors)!=len(set(anchors)) or len(anchors)!=meta['selected_anchors']:
            raise ValueError('Invalid batch selection')
        path=batch/'compactors.tsv'
        if sha(path)!=meta['output_sha256']:
            raise ValueError('Extension output hash changed')
        expected=set(anchors);extended=set();count=0
        with path.open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                a=r['anchor'];dna=r['compactor'];count+=1
                if a not in expected or a not in dna or set(dna.upper())-set('ACGTN'):
                    raise ValueError('Invalid extension sequence or anchor')
                if len(dna)>len(a):extended.add(a)
        if count!=meta['output_rows'] or len(extended)!=meta['anchors_extended']:
            raise ValueError('Extension counts disagree')
        completed.append(dict(batch=meta['batch'],selected_anchors=len(anchors),anchors_extended=len(extended),output_rows=count,output_sha256=meta['output_sha256'],selection=anchors))
    # Read the frozen global sorted selection once and verify disjoint batch membership.
    wanted={i*10000+j:a for b in completed for i in [b['batch']] for j,a in enumerate(b['selection'])}
    if wanted:
        with (root/'selected-anchors.tsv').open() as f:
            for j,r in enumerate(csv.DictReader(f,delimiter='\t')):
                if j in wanted and wanted.pop(j)!=r['anchor']:
                    raise ValueError('Batch does not match full-depth selected slice')
                if not wanted:break
        if wanted:raise ValueError('Missing full-depth selection positions')
    for b in completed:b.pop('selection')
    return dict(selected_total=selection['selected'],total_batches=selection['batches'],source_sha256=selection['source_sha256'],completed_batches=completed,
                completed_selected_anchors=sum(b['selected_anchors'] for b in completed),anchors_extended=sum(b['anchors_extended'] for b in completed),
                extension_records=sum(b['output_rows'] for b in completed),scope='Full-depth read1 SPLASH selection; both complete trimmed mates used for extension context',
                verification='Output hashes, sequence membership, independent counts and exact frozen selection slices checked',complete=len(completed)==selection['batches'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    result=audit(args.root);(args.root/'summary-progress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
