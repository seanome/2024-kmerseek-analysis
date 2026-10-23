import argparse
import csv
import gzip
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from .core import analyze, build_index, read_tsv, sha256


def run_splash(samples, splash_bin, output, top=100, threads=2):
    """Run SPLASH then Compactors on the q-ranked anchors; no condition-effect inference."""
    if threads < 1 or top < 1:
        raise ValueError('threads and top must be positive')
    if sys.platform.startswith('linux') and not Path('/usr/bin/time').exists():
        raise ValueError('SPLASH requires GNU time at /usr/bin/time; install the time system package')
    root = Path(output).resolve()
    if any(c.isspace() for c in str(root)):
        raise ValueError('SPLASH working directory must not contain whitespace')
    if root.exists() and any(root.iterdir()):
        raise ValueError('SPLASH output must be new or empty')
    bins = Path(splash_bin).resolve()
    for name in ['splash', 'compactors']:
        if not (bins/name).is_file():
            raise ValueError(f'Missing executable: {bins/name}')
    rows = read_tsv(samples)
    if len(rows) < 2 or any(not {'sample','read1'} <= r.keys() for r in rows):
        raise ValueError('Sample sheet needs at least two rows with sample,read1; optional read2,condition')
    if len({r['sample'] for r in rows}) != len(rows) or any(not r['sample'] for r in rows):
        raise ValueError('Sample names must be nonempty and unique')
    root.mkdir(parents=True, exist_ok=True)
    records, provenance = [], []
    for i, row in enumerate(rows):
        target = root/f'sample_{i:04d}.fastq.gz'
        sources = []
        with gzip.open(target, 'wb', compresslevel=1) as dest:
            for col in ['read1','read2']:
                if not row.get(col):
                    continue
                source = Path(row[col]).expanduser()
                if not source.is_absolute():
                    source = Path(samples).resolve().parent/source
                sources.append(dict(path=str(source.resolve()), sha256=sha256(source)))
                opener = gzip.open if source.suffix == '.gz' else open
                with opener(source, 'rb') as f:
                    shutil.copyfileobj(f, dest)
        records.append(f'sample_{i:04d} {target}\n')
        provenance.append(dict(internal_sample=f'sample_{i:04d}', sample=row['sample'], condition=row.get('condition',''), sources=sources))
    (root/'input.txt').write_text(''.join(records))
    (root/'reads.txt').write_text(''.join(str(root/f'sample_{i:04d}.fastq.gz')+'\n' for i in range(len(rows))))
    commands=[]
    def execute(command, log):
        commands.append(command)
        with (root/log).open('w') as handle:
            subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)
    start=time.monotonic()
    execute([str(bins/'splash'), '--bin_path',str(bins),'--outname_prefix','splash',
             '--without_compactors','--n_threads_stage_1','1', '--n_threads_stage_1_internal',str(threads),
             '--n_threads_stage_2',str(threads),'--n_bins','8',
             '--dump_sample_anchor_target_count_txt','--satc_merge_dump_format','satc',str(root/'input.txt')], 'splash.log')
    scores=root/'splash.after_correction.scores.tsv'
    rowscores=read_tsv(scores)
    qcol='pval_opt_corrected'
    if rowscores and qcol not in rowscores[0]:
        raise ValueError(f'Unexpected SPLASH score columns: {list(rowscores[0])}; see splash.log')
    chosen=sorted(rowscores,key=lambda r:(float(r[qcol]),r['anchor']))[:top]
    with (root/'selected_anchors.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['anchor']);w.writerows([r['anchor']] for r in chosen)
    compactors=root/'compactors.tsv'
    if chosen:
        execute([str(bins/'compactors'), '--num_threads', str(threads), str(root/'reads.txt'),
                 str(root/'selected_anchors.tsv'),str(compactors)], 'compactors.log')
    else:
        compactors.write_text('anchor\tcompactor\n')
    # Native SATC text format is sample-name, anchor, target, count (no header).
    names={r['internal_sample']:r for r in provenance}
    anchors={r['anchor'] for r in chosen}
    with (root/'counts.tsv').open('w',newline='') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerow(['anchor','sample','condition','target','count'])
        for dump in sorted((root/'splash_dumps').glob('*.satc.dump')):
            with dump.open() as handle:
                for line in handle:
                    parts=line.split()
                    if len(parts)!=4:
                        raise ValueError(f'Unexpected SATC dump row in {dump}')
                    sample,anchor,target,count=parts
                    if anchor in anchors:
                        original=names[sample]
                        writer.writerow([anchor,original['sample'],original['condition'],target,count])
    (root/'run.json').write_text(json.dumps(dict(samples=provenance,commands=commands,
        splash_executable_sha256=sha256(bins/'splash'), compactors_executable_sha256=sha256(bins/'compactors'),
        seconds=round(time.monotonic()-start,3), note='Mates pooled within each sample; pairing is not used. Conditions retained as metadata only.'),indent=2))
    return dict(scores=str(scores),compactors=str(compactors),q_column=qcol,counts=str(root/'counts.tsv'))


def main(argv=None):
    parser=argparse.ArgumentParser(description='Clam Context: SPLASH to sourmash protein / Dayhoff / HP evidence')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('index',help='Build per-protein sourmash index from local FASTA')
    p.add_argument('--fasta',required=True);p.add_argument('--out',required=True)
    p.add_argument('--protein-k',type=int,default=7);p.add_argument('--dayhoff-k',type=int,default=12)
    p.add_argument('--hp-k',type=int,default=30);p.add_argument('--scaled',type=int,default=10)
    p.add_argument('--label',default='User protein FASTA')
    p=sub.add_parser('analyze',help='Import SPLASH scores and extended sequences; search and report')
    for name in ['scores','compactors','database','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--q-column',default='pval_opt_corrected');p.add_argument('--sequence-column',default='compactor')
    p.add_argument('--effect-column',default='effect_size_bin');p.add_argument('--counts')
    p.add_argument('--top',type=int,default=100);p.add_argument('--fdr',type=float,default=.05)
    p.add_argument('--min-aa',type=int,default=20);p.add_argument('--translation-table',type=int,default=1)
    p.add_argument('--min-shared',type=int,default=2);p.add_argument('--min-containment',type=float,default=.1)
    p.add_argument('--top-hits',type=int,default=5);p.add_argument('--scales',type=int,nargs='+',default=[10,100])
    p.add_argument('--demo',action='store_true',help='Label report as synthetic demonstration')
    p=sub.add_parser('splash',help='Run external SPLASH and Compactors on local FASTQs')
    p.add_argument('--samples',required=True);p.add_argument('--splash-bin',required=True);p.add_argument('--out',required=True)
    p.add_argument('--top',type=int,default=100);p.add_argument('--threads',type=int,default=2)
    p=sub.add_parser('demo',help='Generate deterministic synthetic fixtures and run actual sourmash')
    p.add_argument('--out',default='demo-run')
    args=vars(parser.parse_args(argv));command=args.pop('command')
    try:
        if command=='index':
            args['output']=args.pop('out');print(json.dumps(build_index(**args),indent=2))
        elif command=='analyze':
            args['output']=args.pop('out');r=analyze(**args)
            print(f"Analyzed {len(r['events'])} events. Open {Path(args['output'])/'report.html'}")
        elif command=='splash':
            args['output']=args.pop('out');print(json.dumps(run_splash(**args),indent=2))
        else:
            from .demo import demo
            print(demo(args['out']))
    except (ValueError, OSError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'Error: {exc}\n')

if __name__=='__main__':main()
