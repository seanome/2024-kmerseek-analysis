"""Count completed run manifests; this is not an independent file-hash audit."""
import collections
import datetime
import json
import os
from pathlib import Path
root=Path(os.environ['GROUP_SCRATCH'])/'oyster-bivalve-s10-s100-zero-20261023'
records=[]
enc={m:dict(matches_10=0,matches_100=0,lost_pairs=0) for m in ('dna','protein','dayhoff','hp')}
seen=set()
for p in root.glob('runs/*/*/validated.json'):
    r=json.loads(p.read_text())
    if r['run'] in seen: raise ValueError('Duplicate accession')
    seen.add(r['run'])
    if r['scales']!=[10,100] or r['containment_threshold']!=0: raise ValueError('Wrong protocol')
    settings={(s['moltype'],s['scaled']) for s in r['searches']}
    if settings!={(m,s) for m in enc for s in (10,100)}: raise ValueError('Incomplete settings')
    counts={}
    for s in r['searches']:
        if s['queries']!=354316: raise ValueError('Wrong query denominator')
        counts[s['moltype'],s['scaled']]=s['match_rows']
        enc[s['moltype']][f"matches_{s['scaled']}"]+=s['match_rows']
    for c in r['comparisons']:
        if counts[c['moltype'],10]-counts[c['moltype'],100]!=c['lost_matches']:raise ValueError('Loss count mismatch')
        enc[c['moltype']]['lost_pairs']+=c['lost_matches']
    records.append(dict(run=r['run'],study=r['metadata']['study_accession'],species=r['metadata'].get('scientific_name'),metadata=r['metadata']))
out=dict(validated_runs=len(records),validated_studies=len({r['study'] for r in records}),
 reference_inventory_runs=17623,query_records=354316,query_scope='Prefix-derived extensions, not full-depth extensions',
 encodings=enc,runs=records,snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 verification='Completion-manifest settings, query denominators, unique accessions and loss-count arithmetic checked; fresh independent output-file hash audit49387728 pending')
(root/'archive-progress-records.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='runs'}))
