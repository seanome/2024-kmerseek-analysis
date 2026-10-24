"""Validate completed public-pilot searches and compare exact query/run match pairs."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict
import sourmash


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=Path(os.environ['SCRATCH'])/'oyster-shazam-20261023'
    out=root/'public-match-summary-zero';out.mkdir(exist_ok=True)
    with (Path.home()/'oyster-shazam-code/public-bivalve-pilot.tsv').open() as f:metadata={r['run_accession']:r for r in csv.DictReader(f,delimiter='\t')}
    totals=defaultdict(lambda:defaultdict(int));completed=[];examples=[];queries=0
    path=out/'matches-lost-at-scaled100.tsv.gz'
    with gzip.open(path,'wt') as dest:
        fields=['batch','query_id','run','study','species','encoding','shared_hashes_scaled10','containment_scaled10','reason_at_scaled100']
        writer=csv.DictWriter(dest,fieldnames=fields,delimiter='\t');writer.writeheader()
        for i in range(22):
            batch=root/'search-public-bivalve-zero'/f'batch-{i:03d}'
            if not (batch/'validated.json').exists():continue
            v=json.loads((batch/'validated.json').read_text())
            if v.get('containment_threshold')!=0.0:raise ValueError('Search does not declare explicit zero threshold')
            source=root/'splash-trimmed-all'/f'batch-{i:03d}'/'compactors.tsv'
            if sha(source)!=v['input_sha256']:raise ValueError('Extension input changed')
            completed.append(i);queries+=v['extended_records']
            for mol in ['dna','protein','dayhoff','hp']:
                pairs={};empty100=set()
                for sig in sourmash.load_file_as_signatures(str(batch/f'{mol}.sig.zip')):
                    if not len(sig.minhash.downsample(scaled=100)):empty100.add(sig.name)
                for scale in (10,100):
                    s=next(s for s in v['searches'] if s['moltype']==mol and s['scaled']==scale)
                    csvpath=batch/s['output']
                    if sha(csvpath)!=s['output_sha256']:raise ValueError('Search result changed')
                    with csvpath.open() as f:rows=list(csv.DictReader(f))
                    mapping={(r['query_name'],r['match_name']):r for r in rows}
                    if len(rows)!=s['match_rows'] or len(mapping)!=len(rows):raise ValueError('Pair-count mismatch or duplicate')
                    if len({q for q,_ in mapping})!=s['queries_with_overlap']:raise ValueError('Query-count mismatch')
                    pairs[scale]=mapping
                    totals[mol][f'matches_{scale}']+=len(mapping)
                    totals[mol][f'queries_with_overlap_{scale}']+=s['queries_with_overlap']
                    totals[mol][f'empty_{scale}']+=s['empty_sketches']
                lost=pairs[10].keys()-pairs[100].keys();gained=pairs[100].keys()-pairs[10].keys()
                if gained:raise ValueError('Unexpected gained pairs under nested all-overlap search')
                totals[mol]['lost_pairs']+=len(lost);totals[mol]['retained_pairs']+=len(pairs[100])
                for q,run in sorted(lost):
                    hit=pairs[10][q,run];meta=metadata[run]
                    row=dict(batch=i,query_id=q,run=run,study=meta['study_accession'],species=meta['scientific_name'],encoding=mol,
                             shared_hashes_scaled10=hit.get('intersect_hashes',''),containment_scaled10=hit['containment'],
                             reason_at_scaled100='empty query sketch' if q in empty100 else 'no retained shared hash')
                    writer.writerow(row)
                    if len(examples)<30:examples.append(row)
    summary=dict(completed_batches=completed,total_batches=22,extended_records_searched=queries,reference_runs=len(metadata),
                 reference_studies=len({r['study_accession'] for r in metadata.values()}),
                 scope='Eight complete public runs, untrimmed pilot; '+('all 22 query batches complete' if len(completed)==22 else f'partial query coverage, {len(completed)} of 22 batches')+'. New trimmed full-archive results are separate.',
                 rule='Any nonzero shared-hash match; candidate occurrence only, not validated homology or function.',
                 encodings=dict(totals),lost_examples=examples,loss_table=path.name,loss_table_sha256=sha(path))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='lost_examples'}))


if __name__=='__main__':main()
