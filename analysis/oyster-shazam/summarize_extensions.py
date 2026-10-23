"""Independently validate the complete extension partition and export all records."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
from extend_all_significant import select


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root=Path(sys.argv[1]);folder=root/'splash-trimmed-all'
    significant=set(select(root/'splash-trimmed/splash.after_correction.scores.tsv'))
    expected_hash=digest(root/'splash-trimmed/splash.after_correction.scores.tsv')
    anchors=set();extended=set();seqs=set();lengths=Counter();records=0;all_rows=0
    dest=folder/'all-extended-sequences.tsv.gz'
    partial=dest.with_suffix(dest.suffix+'.partial')
    with gzip.open(partial,'wt') as out:
        w=csv.writer(out,delimiter='\t');w.writerow(['batch','anchor','compactor_id','dna','length'])
        for i in range(22):
            batch=folder/f'batch-{i:03d}';v=json.loads((batch/'validated.json').read_text())
            require(v['batch']==i and v['source_sha256']==expected_hash, 'Batch or source mismatch')
            require(digest(batch/'compactors.tsv')==v['output_sha256'], 'Compactors hash mismatch')
            with (batch/'selected_anchors.tsv').open() as f:
                chosen={r['anchor'] for r in csv.DictReader(f,delimiter='\t')}
            require(len(chosen)==v['selected_anchors'] and not chosen&anchors, 'Selection overlap or size mismatch')
            anchors.update(chosen);count=0;batch_ext=set()
            with (batch/'compactors.tsv').open() as f:
                for r in csv.DictReader(f,delimiter='\t'):
                    a=r['anchor'];seq=r['compactor'];count+=1;all_rows+=1
                    require(a in chosen and a in seq and not set(seq)-set('ACGTN'), 'Invalid extension sequence')
                    if len(seq)>len(a):
                        records+=1;extended.add(a);batch_ext.add(a);seqs.add(seq);lengths[len(seq)]+=1
                        w.writerow([i,a,r['id'],seq,len(seq)])
            require(count==v['output_rows'] and len(batch_ext)==v['anchors_extended'], 'Extension count mismatch')
    require(anchors==significant and len(anchors)==213658, 'Incomplete significant-anchor partition')
    partial.replace(dest)
    result=dict(batches=22,selected_anchors=len(anchors),compactor_rows=all_rows,
                extended_records=records,anchors_extended=len(extended),
                unique_dna_sequences=len(seqs),anchors_without_extension=len(anchors-extended),
                length_histogram=dict(sorted(lengths.items())),source_sha256=expected_hash,
                download=dest.name,download_sha256=digest(dest),
                note='Exact DNA duplicates removed only for unique-DNA count; overlapping fragments are not grouped into genes.')
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='length_histogram'}))


if __name__=='__main__':main()
