"""Extend every significant trimmed-read SPLASH anchor in disjoint batches."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from run_splash import command, table


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def select(path):
    anchors = []
    with path.open() as source:
        for row in csv.DictReader(source, delimiter='\t'):
            anchor = row['anchor']
            if not anchor or set(anchor) - set('ACGT'):
                raise ValueError('Invalid DNA anchor')
            q = float(row['pval_opt_corrected'])
            if not math.isfinite(q) or not 0 <= q <= 1:
                raise ValueError('Invalid adjusted P value')
            if q < 0.05:
                anchors.append(anchor)
    if len(anchors) != len(set(anchors)):
        raise ValueError('Duplicate anchor')
    return sorted(anchors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--batch', required=True, type=int)
    p.add_argument('--batch-size', default=10000, type=int)
    p.add_argument('--scores', type=Path)
    p.add_argument('--reads', type=Path)
    p.add_argument('--compactors', type=Path)
    args = p.parse_args()
    source = args.scores or args.root / 'splash-trimmed/splash.after_correction.scores.tsv'
    anchors = select(source)
    if args.batch_size <= 0 or not 0 <= args.batch < math.ceil(len(anchors)/args.batch_size):
        raise ValueError('Invalid batch')
    selected = anchors[args.batch*args.batch_size:(args.batch+1)*args.batch_size]
    out = args.root / 'splash-trimmed-all' / f'batch-{args.batch:03d}'
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_sha256=sha256(source),
                    threshold=0.05, comparison='<', total_selected=len(anchors),
                    batch=args.batch, batch_size=args.batch_size, selected_anchors=len(selected))
    done = out / 'validated.json'
    if done.exists():
        saved = json.loads(done.read_text())
        if not all(saved.get(k) == v for k, v in metadata.items()):
            raise ValueError('Validated metadata does not match this selection')
        if saved.get('output_sha256') != sha256(out/'compactors.tsv'):
            raise ValueError('Validated output hash does not match')
        return
    if (out/'compactors.command.json').exists() or (out/'compactors.tsv').exists():
        raise RuntimeError('Unvalidated previous attempt; inspect before retry')
    (out/'selection.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with (out/'selected_anchors.tsv').open('w') as f:
        w=csv.writer(f, delimiter='\t'); w.writerow(['anchor']); w.writerows([a] for a in selected)
    command([args.compactors or args.root/'software/compactors','--num_threads','8',
             args.reads or args.root/'splash-trimmed/reads.txt',out/'selected_anchors.tsv',out/'compactors.tsv'], out, 'compactors')
    expected=set(selected); extended=set(); count=0
    with (out/'compactors.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            a=row['anchor']; seq=row['compactor']
            if a not in expected or a not in seq or set(seq.upper()) - set('ACGTN'):
                raise ValueError('Invalid Compactors output sequence or anchor')
            count+=1
            if len(seq)>len(a): extended.add(a)
    metadata.update(output_rows=count, anchors_extended=len(extended),
                    output_sha256=sha256(out/'compactors.tsv'))
    done.write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
