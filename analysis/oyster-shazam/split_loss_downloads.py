"""Publish the verified loss export as smaller downloads, one per encoding."""
import argparse
import csv
import gzip
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

r=Path(__file__).resolve().parent/'results'
parser=argparse.ArgumentParser();parser.add_argument('--full-depth',action='store_true');args=parser.parse_args()
p=r/('full-public-match-summary.json' if args.full_depth else 'public-match-summary.json');data=json.loads(p.read_text());source=r/data['loss_table']
prefix='full-' if args.full_depth else ''
with source.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!=data['loss_table_sha256']:
        raise ValueError('Full loss export hash changed')
counts={m:0 for m in data['encodings']};paths={m:r/f'{prefix}matches-lost-at-scaled100-{m}.tsv.gz' for m in counts}
with ExitStack() as stack:
    streams={m:stack.enter_context(gzip.open(path,'wt')) for m,path in paths.items()}
    with gzip.open(source,'rt') as f:
        reader=csv.DictReader(f,delimiter='\t')
        writers={m:csv.DictWriter(s,fieldnames=reader.fieldnames,delimiter='\t') for m,s in streams.items()}
        for w in writers.values():w.writeheader()
        for row in reader:
            m=row['encoding'];writers[m].writerow(row);counts[m]+=1
files={}
for m,path in paths.items():
    if counts[m]!=data['encodings'][m]['lost_pairs']:
        raise ValueError('Split loss count disagrees')
    if path.stat().st_size>25*1024**2:
        raise ValueError('Split still too large for report hosting')
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    files[m]=dict(file=path.name,sha256=digest,rows=counts[m],bytes=path.stat().st_size)
data['loss_tables']=files;p.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(files))
